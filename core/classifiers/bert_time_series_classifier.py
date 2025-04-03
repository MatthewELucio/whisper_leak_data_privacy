# bert_time_series_classifier.py

import ast
import math
import json
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, logging as hf_logging
import warnings
from peft import LoraConfig, get_peft_model, TaskType # pip install peft

# Suppress excessive warnings from transformers library (optional)
hf_logging.set_verbosity_error() 

# Assuming base_classifier.py is in the same directory or accessible via python path
from .base_classifier import BaseClassifier 
from core.utils import PrintUtils # Assuming PrintUtils exists for parameter printing

class BERTTimeSeriesClassifier(BaseClassifier):
    """
    A classifier using a pre-trained BERT model adapted for time series data, 
    optionally using LoRA for parameter-efficient fine-tuning.
    It buckets normalized time_diffs and data_lengths, converts them to new tokens,
    interleaves them (optional), and feeds them into the transformer model for 
    binary classification.
    """

    def __init__(self, 
                 normalization_params,
                 # Bucket boundaries for *normalized* data
                 time_boundaries_norm, 
                 len_boundaries_norm,
                 num_buckets=50,
                 # Pre-trained model config
                 model_name='distilbert-base-uncased', # Good balance of size/performance
                 # Classifier head config
                 fc_dims=[64], 
                 dropout_rate=0.3,
                 interleave=True,
                 # --- LoRA Configuration ---
                 use_lora: bool = True,          # Set to False to disable LoRA (standard fine-tuning)
                 lora_r: int = 32,                # LoRA rank
                 lora_alpha: int = 16,           # LoRA alpha (scaling factor)
                 lora_dropout: float = 0.1,      # LoRA dropout
                 # --- End LoRA Configuration ---
                 ):
        """
        Creates an instance of the BERTTimeSeriesClassifier.

        Args:
            normalization_params (dict): Dictionary containing normalization parameters like 
                                         'time_mean', 'time_std', 'size_mean', 'size_std', 'max_len'.
            time_boundaries_norm (list/np.array): Quantile boundaries for normalized time_diffs. 
                                                  Should have num_buckets - 1 elements.
            len_boundaries_norm (list/np.array): Quantile boundaries for normalized data_lengths.
                                                 Should have num_buckets - 1 elements.
            num_buckets (int): The number of buckets to divide time and length data into.
            model_name (str): The name of the pre-trained model from Hugging Face Hub.
            fc_dims (list): List of dimensions for fully connected layers in the classification head.
            dropout_rate (float): Dropout rate for regularization in the classification head.
            interleave (bool): Whether to interleave time and length tokens or append them.
            use_lora (bool): Whether to use LoRA for fine-tuning.
            lora_r (int): Rank for LoRA matrices.
            lora_alpha (int): Alpha scaling factor for LoRA.
            lora_dropout (float): Dropout for LoRA layers.
        """
        super().__init__(normalization_params)
        self.class_name = self.__class__.__name__

        # --- Parameter Validation ---
        if not (isinstance(time_boundaries_norm, (list, np.ndarray)) and \
                len(time_boundaries_norm) == num_buckets - 1):
             raise ValueError(f"time_boundaries_norm must be a list/array of length num_buckets-1 ({num_buckets-1})")
        if not (isinstance(len_boundaries_norm, (list, np.ndarray)) and \
                len(len_boundaries_norm) == num_buckets - 1):
             raise ValueError(f"len_boundaries_norm must be a list/array of length num_buckets-1 ({num_buckets-1})")

        # --- Store Config ---
        self.time_boundaries_norm = np.array(time_boundaries_norm)
        self.len_boundaries_norm = np.array(len_boundaries_norm)
        self.num_buckets = num_buckets
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.fc_dims = fc_dims
        self.interleave = interleave
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Original max sequence length (from normalization params)
        self.original_max_len = self.max_len 
        # Max length for BERT input (depends on interleaving + [CLS] + [SEP])
        # Note: Actual BERT input length determined dynamically based on self.bert_max_len_limit
        self.bert_max_len_limit = 512 # Standard limit for many BERT models
        
        self.args = {
            'model_name': self.model_name,
            'num_buckets': self.num_buckets,
            # Store boundaries as lists for JSON serialization
            'time_boundaries_norm': self.time_boundaries_norm.tolist(), 
            'len_boundaries_norm': self.len_boundaries_norm.tolist(),
            'fc_dims': self.fc_dims,
            'dropout_rate': self.dropout_rate,
            'interleave': self.interleave,
            'use_lora': self.use_lora,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            # Note: normalization_params are handled by the BaseClassifier save/load
        }

        # --- Load Tokenizer and Base Model ---
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Loading model: {model_name}")
        # Load the base model weights. PEFT will be applied later if enabled.
        self.bert_model = AutoModel.from_pretrained(model_name)

        # --- Add New Tokens ---
        self.len_tokens = [f"[LEN_{i}]" for i in range(num_buckets)]
        self.time_tokens = [f"[TIME_{i}]" for i in range(num_buckets)]
        special_tokens_to_add = self.len_tokens + self.time_tokens
        
        num_added_toks = self.tokenizer.add_tokens(special_tokens_to_add, special_tokens=True)
        print(f"Added {num_added_toks} new tokens to tokenizer.")
        
        # --- Resize Model Embeddings ---
        # IMPORTANT: Resize embeddings *before* applying PEFT
        self.bert_model.resize_token_embeddings(len(self.tokenizer))
        print(f"Resized model embeddings to: {len(self.tokenizer)}")

        # --- Get Token IDs (after adding new tokens) ---
        self.len_token_ids = self.tokenizer.convert_tokens_to_ids(self.len_tokens)
        self.time_token_ids = self.tokenizer.convert_tokens_to_ids(self.time_tokens)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        
        if self.pad_token_id is None:
             print("Warning: Tokenizer has no default pad token. Adding '[PAD]'.")
             self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
             self.pad_token_id = self.tokenizer.pad_token_id
             # Resize again if PAD was added
             self.bert_model.resize_token_embeddings(len(self.tokenizer))

        # --- Apply LoRA (PEFT) if enabled ---
        if self.use_lora:
            print(f"Applying LoRA with r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}")
            # Common target modules for DistilBERT. Might need adjustment for other models.
            # Inspect model structure (e.g., print(self.bert_model)) if unsure.
            #target_modules = ["q_lin", "v_lin"] 
            #target_modules = ["attention", "embeddings"]
            target_modules=[
                 # Target the linear layers within the attention blocks
                "q_lin",    # Query projection layer in DistilBERT attention
                #"k_lin",    # Key projection layer
                "v_lin"    # Value projection layer
                #"out_lin",  # Output projection layer in attention

                # Target the specific embedding layers
                #"word_embeddings",      # The main token embeddings
                #"position_embeddings"   # Positional embeddings (optional but can be included)
            ],
            target_modules = ["q_lin", "k_lin", "v_lin", "out_lin", "word_embeddings", "position_embeddings"] 
            
            # Check if target modules exist in the model
            found_modules = []
            for name, module in self.bert_model.named_modules():
                 # Check the final part of the name
                 if name.split('.')[-1] in target_modules:
                      found_modules.append(name.split('.')[-1])
            
            unique_found = sorted(list(set(found_modules)))
            if not unique_found:
                 warnings.warn(f"LoRA target modules {target_modules} not found in {self.model_name}. LoRA may not be applied effectively.")
                 # Fallback or raise error? For now, continue with empty targets (won't apply LoRA).
                 target_modules = [] 
            elif set(unique_found) != set(target_modules):
                 warnings.warn(f"Found LoRA target modules {unique_found}, but expected {target_modules}. Using found modules: {unique_found}")
                 target_modules = unique_found


            lora_config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=target_modules, 
                lora_dropout=self.lora_dropout,
                bias="none", # Common setting: 'none', 'all', 'lora_only'
                task_type=None
                #task_type=TaskType.SEQ_CLS # Indicate task type (might help PEFT optimize)
                #                         # Set to None if using a custom head setup like here might be okay too.
            )
            self.bert_model = get_peft_model(self.bert_model, lora_config)
            
            # --- Print Trainable Parameters (Good for verification) ---
            # try: # Use try-except as PrintUtils might not be available everywhere
            #     PrintUtils.print_trainable_parameters(self.bert_model)
            # except NameError:
            #     print("Skipping trainable parameter count printout (PrintUtils not found).")
            # --- End Print Trainable Parameters ---
        else:
             print("LoRA is disabled. Using standard fine-tuning.")


        # --- Classification Head ---
        # This head is always trained, whether using LoRA or full fine-tuning.
        bert_output_dim = self.bert_model.config.hidden_size
        
        fc_layers = []
        input_dim = bert_output_dim
        
        for dim in fc_dims:
            fc_layers.extend([
                nn.Linear(input_dim, dim),
                nn.LayerNorm(dim), 
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = dim
            
        # Final output layer
        fc_layers.append(nn.Linear(input_dim, 1))
        
        # Combine all FC layers
        self.fc_layers = nn.Sequential(*fc_layers)

        # Initialize weights for the classification head
        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Initialize weights for the classification head."""
        for m in self.fc_layers.modules():
            if isinstance(m, nn.Linear):
                # Consider Glorot/Xavier initialization
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                 # Default LayerNorm initialization is usually fine (weights=1, bias=0)
                 nn.init.constant_(m.weight, 1.0)
                 nn.init.constant_(m.bias, 0.0)

    def _bucket_and_tokenize_sample(self, norm_time_vals, norm_size_vals):
        """Buckets, tokenizes, optionally interleaves, and pads a single sample."""
        
        seq_len = len(norm_time_vals) # Assuming input arrays are already trimmed

        # Bucket the normalized values
        time_bucket_indices = np.digitize(norm_time_vals[:seq_len], self.time_boundaries_norm)
        len_bucket_indices = np.digitize(norm_size_vals[:seq_len], self.len_boundaries_norm)

        # Map bucket indices to token IDs
        time_tok_ids = [self.time_token_ids[idx] for idx in time_bucket_indices]
        len_tok_ids = [self.len_token_ids[idx] for idx in len_bucket_indices]

        # Combine tokens (Interleave or Append)
        if self.interleave:
            combined_tokens = []
            for i in range(seq_len):
                combined_tokens.append(len_tok_ids[i])
                combined_tokens.append(time_tok_ids[i])
            max_combined_len = 2 * self.original_max_len 
        else:
            # Append: [LEN tokens] + [TIME tokens]
            combined_tokens = len_tok_ids + time_tok_ids
            max_combined_len = 2 * self.original_max_len # Still potential for 2*N tokens

        # Add special tokens ([CLS] at start, [SEP] at end)
        # BERT expects: [CLS] sequence [SEP]
        final_token_ids = [self.cls_token_id] + combined_tokens + [self.sep_token_id]
        
        # Calculate max length for *this specific input type* based on the limit
        current_max_len = self.bert_max_len_limit

        # Pad or truncate sequence
        padding_len = current_max_len - len(final_token_ids)
        
        if padding_len < 0:
            # Truncate if sequence exceeds the limit
            # Keep [CLS] and ensure [SEP] is the last token
            # print(f"Warning: Sequence truncated from {len(final_token_ids)} to {current_max_len}")
            final_token_ids = final_token_ids[:current_max_len-1] + [self.sep_token_id]
            attention_mask = [1] * current_max_len
        else:
            # Pad if sequence is shorter than the limit
            final_token_ids = final_token_ids + ([self.pad_token_id] * padding_len)
            attention_mask = ([1] * (current_max_len - padding_len)) + ([0] * padding_len)
            
        # Sanity check length
        assert len(final_token_ids) == current_max_len, f"Length mismatch: {len(final_token_ids)} vs {current_max_len}"
        assert len(attention_mask) == current_max_len, f"Mask length mismatch: {len(attention_mask)} vs {current_max_len}"

        return final_token_ids, attention_mask

    def forward(self, x):
        """
        Implements the forward pass: bucketing, tokenization, BERT (w/ or w/o LoRA), classification head.
        Returns raw logits for loss calculation.

        Args:
            x (torch.Tensor): Input tensor from Loader of shape 
                              [batch_size, 2, original_max_len], containing
                              *normalized* and *padded* time_diffs and data_lengths.
                              x[:, 0, :] = normalized time_diffs
                              x[:, 1, :] = normalized data_lengths
        
        Returns:
            torch.Tensor: Logits output from the final classification layer. 
                          Shape: [batch_size, 1]
        """
        batch_size = x.size(0)
        device = x.device

        all_input_ids = []
        all_attention_masks = []

        # Process each sample in the batch
        for i in range(batch_size):
            # Extract normalized, padded sequences for the i-th sample
            norm_time_padded = x[i, 0, :].cpu().numpy()
            norm_size_padded = x[i, 1, :].cpu().numpy()

            # --- Find original length before padding ---
            non_zero_indices = np.where((norm_time_padded != 0) | (norm_size_padded != 0))[0]
            if len(non_zero_indices) == 0:
                 original_len = 0 # Handle completely zero sequences
            else:
                 original_len = non_zero_indices[-1] + 1
                 
            # Ensure we don't exceed the expected max_len from normalization_params
            original_len = min(original_len, self.original_max_len) 

            # Handle empty sequences after length calculation
            if original_len == 0:
                 input_ids = [self.cls_token_id, self.sep_token_id] + \
                             [self.pad_token_id] * (self.bert_max_len_limit - 2)
                 attention_mask = [1, 1] + [0] * (self.bert_max_len_limit - 2)
                 
            else:
                norm_time_vals = norm_time_padded[:original_len]
                norm_size_vals = norm_size_padded[:original_len]
                input_ids, attention_mask = self._bucket_and_tokenize_sample(norm_time_vals, norm_size_vals)
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

        # Convert lists to tensors
        input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long).to(device)
        attention_mask_tensor = torch.tensor(all_attention_masks, dtype=torch.long).to(device)

        # --- Pass through BERT model (PEFT handles LoRA automatically if applied) ---
        outputs = self.bert_model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor,
            return_dict=True 
        )

        # --- Get representation for classification ---
        cls_output = outputs.last_hidden_state[:, 0, :] 

        # --- Pass through classification head ---
        # Output raw logits (before sigmoid)
        logits = self.fc_layers(cls_output) # Shape: [batch_size, 1]
        
        # Return logits directly. Loss calculation (e.g., with BCEWithLogitsLoss) 
        # should happen in the training loop.
        return logits # Shape: [batch_size, 1]


    # --- calculate_boundaries remains unchanged ---
    @staticmethod
    def calculate_boundaries(df, num_buckets, norm):
        """
        Calculates quantile boundaries for normalized time_diffs and data_lengths.

        (Implementation is identical to the previous version, so omitted here for brevity) 
        Args:
            df (pd.DataFrame): DataFrame containing 'time_diffs' and 'data_lengths' columns.
                               Values can be lists or string representations of lists.
            num_buckets (int): The number of buckets to create (N). This method will
                               calculate N-1 quantile boundaries.
            normalization_params (dict): A dictionary containing the pre-calculated normalization
                                          parameters: 'time_mean', 'time_std', 'size_mean', 'size_std'.

        Returns:
            tuple: A tuple containing two numpy arrays:
                   (time_boundaries_norm, len_boundaries_norm).
                   Each array contains num_buckets - 1 boundary values.
        Raises:
            ValueError, TypeError: As documented previously.
        """
        # --- Input Validation ---
        if 'time_diffs' not in df.columns or 'data_lengths' not in df.columns:
            raise ValueError("DataFrame must contain 'time_diffs' and 'data_lengths' columns.")
        if not isinstance(num_buckets, int) or num_buckets < 2:
             raise ValueError("num_buckets must be an integer greater than or equal to 2.")
        if not isinstance(norm, (dict)):
            raise TypeError("Normalization parameters must be a dictionary.")
        required_keys = ["time_mean", "time_std", "size_mean", "size_std"]
        if not all(key in norm for key in required_keys):
            missing = [k for k in required_keys if k not in norm]
            raise ValueError(f"Normalization parameters dictionary is missing keys: {missing}")
        
        time_mean = norm["time_mean"]
        time_std = norm["time_std"]
        size_mean = norm["size_mean"]
        size_std = norm["size_std"]

        # Ensure standard deviations are positive for safe division
        time_std_safe = time_std if time_std > 1e-6 else 1.0 # Use small epsilon
        size_std_safe = size_std if size_std > 1e-6 else 1.0

        # --- Data Preparation and Normalization ---
        all_norm_time_vals = []
        all_norm_size_vals = []

        for _, row in df.iterrows():
            try:
                # Attempt to evaluate if string, otherwise assume list/iterable
                time_vals = ast.literal_eval(row['time_diffs']) if isinstance(row['time_diffs'], str) else list(row['time_diffs'])
                size_vals = ast.literal_eval(row['data_lengths']) if isinstance(row['data_lengths'], str) else list(row['data_lengths'])
            except (ValueError, SyntaxError, TypeError) as e:
                 print(f"Warning: Skipping row due to parsing/type error: {e}. Row content:\n{row}")
                 continue 

            if not isinstance(time_vals, list) or not isinstance(size_vals, list):
                 print(f"Warning: Skipping row - expected lists after processing. Row content:\n{row}")
                 continue 

            # Normalize and add to flattened lists (only non-empty lists)
            if time_vals:
                 all_norm_time_vals.extend([(float(val) - time_mean) / time_std_safe for val in time_vals])
            if size_vals:
                 all_norm_size_vals.extend([(float(val) - size_mean) / size_std_safe for val in size_vals])


        if not all_norm_time_vals or not all_norm_size_vals:
             # It's possible one list might be empty if all inputs for that feature were empty lists
             if not all_norm_time_vals: print("Warning: No valid time values found after processing.")
             if not all_norm_size_vals: print("Warning: No valid size values found after processing.")
             # Decide how to handle this: raise error or return default boundaries?
             # Returning default boundaries (e.g., zeros) might be safer for robustness.
             default_boundaries = np.zeros(num_buckets - 1)
             time_boundaries_norm = default_boundaries if not all_norm_time_vals else None
             len_boundaries_norm = default_boundaries if not all_norm_size_vals else None
             if time_boundaries_norm is not None and len_boundaries_norm is not None:
                  return time_boundaries_norm, len_boundaries_norm
             # If only one is missing, calculate the other
             # raise ValueError("After processing, no valid time or size values were found in the DataFrame.")


        # --- Quantile Calculation ---
        quantiles = np.linspace(0, 1, num_buckets + 1)[1:-1] 
        
        if all_norm_time_vals:
             time_boundaries_norm = np.quantile(all_norm_time_vals, quantiles)
             # Check for duplicate boundaries
             if len(np.unique(time_boundaries_norm)) != num_buckets - 1:
                  print(f"Warning: Duplicate time boundaries found ({len(np.unique(time_boundaries_norm))} unique out of {num_buckets - 1}). This might occur with skewed data or low num_buckets. Consider adjusting num_buckets or data preprocessing.")
                  # Simple fallback: use linspace over the range if quantiles fail badly
                  if len(np.unique(time_boundaries_norm)) < 2: 
                       min_t, max_t = np.min(all_norm_time_vals), np.max(all_norm_time_vals)
                       if max_t > min_t:
                            time_boundaries_norm = np.linspace(min_t, max_t, num_buckets + 1)[1:-1]
                       else: # All values are the same
                            time_boundaries_norm = np.full(num_buckets - 1, min_t)


        if all_norm_size_vals:
             len_boundaries_norm = np.quantile(all_norm_size_vals, quantiles)
             if len(np.unique(len_boundaries_norm)) != num_buckets - 1:
                  print(f"Warning: Duplicate length boundaries found ({len(np.unique(len_boundaries_norm))} unique out of {num_buckets - 1}). See advice for time boundaries.")
                  if len(np.unique(len_boundaries_norm)) < 2:
                       min_s, max_s = np.min(all_norm_size_vals), np.max(all_norm_size_vals)
                       if max_s > min_s:
                            len_boundaries_norm = np.linspace(min_s, max_s, num_buckets + 1)[1:-1]
                       else:
                           len_boundaries_norm = np.full(num_buckets - 1, min_s)


        return time_boundaries_norm, len_boundaries_norm
