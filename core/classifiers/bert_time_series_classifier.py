import ast
import math
import json
import os
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModel, logging as hf_logging
import warnings
from peft import LoraConfig, get_peft_model, TaskType # pip install peft

# Suppress excessive warnings from transformers library (optional)
hf_logging.set_verbosity_error()

# Assuming base_classifier.py is in the same directory or accessible via python path
from .base_classifier import BaseClassifier
# from core.utils import PrintUtils # Assuming PrintUtils exists for parameter printing (Optional)

class BERTTimeSeriesClassifier(BaseClassifier):
    """
    A classifier using a pre-trained BERT model adapted for time series data,
    using continuous-valued embeddings projected from normalized time and size values,
    optionally using LoRA for parameter-efficient fine-tuning.

    Instead of bucketing, it uses placeholder tokens ([TIME_VAL], [SIZE_VAL])
    and adds learned projections of the actual normalized continuous values
    to the standard token embeddings before feeding them to the BERT encoder.
    """

    def __init__(self,
                 normalization_params,
                 time_boundaries_norm,
                 len_boundaries_norm,
                 num_buckets=50,
                 # Pre-trained model config
                 model_name='distilbert-base-uncased', # Good balance of size/performance
                 # Classifier head config
                 fc_dims=[128, 64],
                 dropout_rate=0.3,
                 # --- LoRA Configuration ---
                 use_lora: bool = False,          # Set to False to disable LoRA (standard fine-tuning)
                 lora_r: int = 16,                # LoRA rank (reduced default, tune as needed)
                 lora_alpha: int = 16,           # LoRA alpha (scaling factor)
                 lora_dropout: float = 0.1,      # LoRA dropout
                 # --- End LoRA Configuration ---
                 ):
        """
        Creates an instance of the BERTContinuousClassifier.

        Args:
            normalization_params (dict): Dictionary containing normalization parameters like
                                         'time_mean', 'time_std', 'size_mean', 'size_std', 'max_len'.
            model_name (str): The name of the pre-trained model from Hugging Face Hub.
            fc_dims (list): List of dimensions for fully connected layers in the classification head.
            dropout_rate (float): Dropout rate for regularization in the classification head.
            use_lora (bool): Whether to use LoRA for fine-tuning.
            lora_r (int): Rank for LoRA matrices.
            lora_alpha (int): Alpha scaling factor for LoRA.
            lora_dropout (float): Dropout for LoRA layers.
        """
        super().__init__(normalization_params) # normalization_params stored in self.norm
        self.class_name = self.__class__.__name__

        # --- Store Config ---
        self.model_name = model_name
        self.dropout_rate = dropout_rate
        self.fc_dims = fc_dims
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

        # Original max sequence length for time/size pairs (from normalization params)
        if 'max_len' not in self.norm:
             raise ValueError("normalization_params dictionary must contain 'max_len'")
        self.original_max_len = self.norm['max_len']

        # Max length for BERT input: [CLS] + N * ([TIME_VAL] + [SIZE_VAL]) + [SEP]
        # We need to ensure this doesn't exceed the model's absolute limit (e.g., 512)
        self.bert_model_max_len = AutoConfig.from_pretrained(model_name).max_position_embeddings
        self.calculated_input_len = 1 + self.original_max_len * 2 + 1 # CLS + pairs + SEP
        
        if self.calculated_input_len > self.bert_model_max_len:
            # Adjust original_max_len to fit within the BERT limit
            allowed_pairs = (self.bert_model_max_len - 2) // 2
            warnings.warn(
                f"Original max_len ({self.original_max_len}) leads to BERT input length "
                f"({self.calculated_input_len}) exceeding model max length ({self.bert_model_max_len}). "
                f"Truncating effective sequence length to {allowed_pairs} time/size pairs."
            )
            self.original_max_len = allowed_pairs
            self.input_seq_len = 1 + self.original_max_len * 2 + 1
        else:
            self.input_seq_len = self.calculated_input_len
            
        print(f"Effective sequence length for time/size pairs: {self.original_max_len}")
        print(f"Resulting BERT input sequence length: {self.input_seq_len}")


        self.args = {
            'model_name': self.model_name,
            'fc_dims': self.fc_dims,
            'dropout_rate': self.dropout_rate,
            'use_lora': self.use_lora,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            # 'original_max_len': self.original_max_len # Store the possibly adjusted value
            # Note: normalization_params are handled by the BaseClassifier save/load
        }

        # --- Load Tokenizer and Base Model ---
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Loading model: {model_name}")
        # Load the base model weights. PEFT will be applied later if enabled.
        self.bert_model = AutoModel.from_pretrained(model_name)
        bert_hidden_size = self.bert_model.config.hidden_size

        # --- Add New Special Tokens for Continuous Placeholders ---
        self.time_val_token = "[TIME_VAL]"
        self.size_val_token = "[SIZE_VAL]"
        special_tokens_to_add = [self.time_val_token, self.size_val_token]

        num_added_toks = self.tokenizer.add_tokens(special_tokens_to_add, special_tokens=True)
        print(f"Added {num_added_toks} new special tokens: {special_tokens_to_add}")

        # --- Resize Model Embeddings ---
        # IMPORTANT: Resize embeddings *before* applying PEFT
        self.bert_model.resize_token_embeddings(len(self.tokenizer))
        print(f"Resized model embeddings to: {len(self.tokenizer)}")

        # --- Get Token IDs (after adding new tokens) ---
        self.time_val_token_id = self.tokenizer.convert_tokens_to_ids(self.time_val_token)
        self.size_val_token_id = self.tokenizer.convert_tokens_to_ids(self.size_val_token)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        if self.pad_token_id is None:
             print("Warning: Tokenizer has no default pad token. Adding '[PAD]'.")
             self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
             self.pad_token_id = self.tokenizer.pad_token_id
             # Resize again if PAD was added
             self.bert_model.resize_token_embeddings(len(self.tokenizer))

        # --- Linear Projection Layers for Continuous Values ---
        self.time_embed_proj = nn.Linear(1, bert_hidden_size)
        self.size_embed_proj = nn.Linear(1, bert_hidden_size)
        # Initialize projection layers (optional, but can help)
        nn.init.xavier_uniform_(self.time_embed_proj.weight)
        nn.init.constant_(self.time_embed_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.size_embed_proj.weight)
        nn.init.constant_(self.size_embed_proj.bias, 0.0)
        print(f"Created projection layers (1 -> {bert_hidden_size}) for time and size.")

        # --- Apply LoRA (PEFT) if enabled ---
        if self.use_lora:
            print(f"Applying LoRA with r={self.lora_r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}")
            # Define target modules (check model architecture if needed)
            target_modules = ["q_lin", "k_lin", "v_lin", "out_lin"] # Common for attention layers
            # Optionally target embedding layers as well, but start with attention
            # target_modules.extend(["word_embeddings", "position_embeddings"])

            # Verify target modules exist
            found_modules = set()
            for name, module in self.bert_model.named_modules():
                 module_name = name.split('.')[-1]
                 if module_name in target_modules:
                      found_modules.add(module_name)
            
            verified_target_modules = list(found_modules)
            if not verified_target_modules:
                 warnings.warn(f"LoRA target modules {target_modules} not found in {self.model_name}. LoRA may not be applied effectively.")
            elif set(verified_target_modules) != set(target_modules):
                 warnings.warn(f"Specified LoRA target modules {target_modules}, but only found {verified_target_modules}. Using found modules.")

            if verified_target_modules:
                 lora_config = LoraConfig(
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    target_modules=verified_target_modules,
                    lora_dropout=self.lora_dropout,
                    bias="none", # Common setting: 'none', 'all', 'lora_only'
                    task_type=None # Using custom head, so None might be appropriate
                 )
                 self.bert_model = get_peft_model(self.bert_model, lora_config)
                 # Optional: Print trainable parameters to verify LoRA application
                 # try:
                 #      PrintUtils.print_trainable_parameters(self.bert_model)
                 # except NameError:
                 #      pass # Or use peft's built-in print function
                 self.bert_model.print_trainable_parameters()
            else:
                 # No modules found, LoRA cannot be applied
                 print("Skipping LoRA application as no target modules were found.")
                 self.use_lora = False # Update flag to reflect reality
                 self.args['use_lora'] = False # Update args as well

        else:
             print("LoRA is disabled. Using standard fine-tuning.")


        # --- Classification Head ---
        # This head is always trained, whether using LoRA or full fine-tuning.
        # bert_output_dim = self.bert_model.config.hidden_size # Redundant, already got bert_hidden_size

        fc_layers = []
        input_dim = bert_hidden_size

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
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                 nn.init.constant_(m.weight, 1.0)
                 nn.init.constant_(m.bias, 0.0)


    def forward(self, x):
        """
        Implements the forward pass: creates input sequence with placeholders,
        projects continuous values, adds them to embeddings, runs through BERT
        (w/ or w/o LoRA), and passes through the classification head.

        Args:
            x (torch.Tensor): Input tensor from Loader of shape
                              [batch_size, 2, sequence_length_from_loader], containing
                              *normalized* and *padded* time_diffs and data_lengths.
                              x[:, 0, :] = normalized time_diffs
                              x[:, 1, :] = normalized data_lengths

        Returns:
            torch.Tensor: Logits output from the final classification layer.
                          Shape: [batch_size, 1]
        """
        batch_size = x.size(0)
        device = x.device
        bert_hidden_size = self.bert_model.config.hidden_size # Get hidden size

        all_input_ids = []
        all_continuous_values = {'time': [], 'size': []} # Store values aligned with placeholders
        all_original_lens = []

        # --- 1. Prepare Input IDs and Extract Continuous Values ---
        for i in range(batch_size):
            norm_time_padded = x[i, 0, :].cpu().numpy()
            norm_size_padded = x[i, 1, :].cpu().numpy()

            # Find original length before padding (based on non-zero in *either* channel)
            non_zero_indices = np.where((norm_time_padded != 0) | (norm_size_padded != 0))[0]
            original_len = 0
            if len(non_zero_indices) > 0:
                original_len = non_zero_indices[-1] + 1

            # Crucially, cap original_len at the effective max_len determined during init
            original_len = min(original_len, self.original_max_len)
            all_original_lens.append(original_len)

            # Build the token ID sequence for this sample
            input_ids_sample = [self.cls_token_id]
            time_vals_sample = []
            size_vals_sample = []

            for j in range(original_len):
                 input_ids_sample.extend([self.time_val_token_id, self.size_val_token_id])
                 time_vals_sample.append(norm_time_padded[j])
                 size_vals_sample.append(norm_size_padded[j])

            input_ids_sample.append(self.sep_token_id)

            # Pad the sequence to the fixed input_seq_len determined in __init__
            padding_len = self.input_seq_len - len(input_ids_sample)
            input_ids_sample.extend([self.pad_token_id] * padding_len)

            all_input_ids.append(input_ids_sample)
            # Store continuous values, padding them to original_max_len for easier processing later
            all_continuous_values['time'].append(
                np.pad(time_vals_sample, (0, self.original_max_len - len(time_vals_sample)))
            )
            all_continuous_values['size'].append(
                 np.pad(size_vals_sample, (0, self.original_max_len - len(size_vals_sample)))
            )


        # Convert lists to tensors
        input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long).to(device)
        time_values_tensor = torch.tensor(np.array(all_continuous_values['time']), dtype=torch.float32).to(device)
        size_values_tensor = torch.tensor(np.array(all_continuous_values['size']), dtype=torch.float32).to(device)

        # Create attention mask based on input_ids_tensor
        attention_mask = (input_ids_tensor != self.pad_token_id).long()

        # --- 2. Get Base Embeddings (Token + Position) ---
        # Standard BERT embedding lookup requires input_ids
        # We need word and position embeddings separately to modify them.
        word_embeddings = self.bert_model.base_model.embeddings.word_embeddings(input_ids_tensor)
        
        # Create position_ids explicitly (0 to seq_len-1)
        position_ids = torch.arange(self.input_seq_len, dtype=torch.long, device=device).unsqueeze(0).expand_as(input_ids_tensor)
        position_embeddings = self.bert_model.base_model.embeddings.position_embeddings(position_ids)
        
        # --- 3. Calculate Continuous Embedding Projections and Add Offsets ---
        # Initialize offsets tensor
        continuous_embed_offset = torch.zeros_like(word_embeddings) # Shape: [batch, seq_len, hidden_size]

        # Iterate through the sequence pairs (up to original_max_len)
        for j in range(self.original_max_len):
             # Calculate the sequence index for the j-th time and size tokens
             # [CLS] T0 S0 T1 S1 ... Tj Sj ... SEP PAD...
             # Index of Tj = 1 (for CLS) + j * 2
             # Index of Sj = 1 (for CLS) + j * 2 + 1
             time_token_seq_idx = 1 + j * 2
             size_token_seq_idx = 1 + j * 2 + 1

             # Check if these indices are within the actual sequence length (self.input_seq_len)
             # This check is important if original_max_len was truncated during init
             if time_token_seq_idx >= self.input_seq_len or size_token_seq_idx >= self.input_seq_len:
                 break # Stop if we exceed the allowed sequence length

             # Get the j-th time/size values for the whole batch
             current_time_vals = time_values_tensor[:, j].unsqueeze(-1) # Shape: [batch, 1]
             current_size_vals = size_values_tensor[:, j].unsqueeze(-1) # Shape: [batch, 1]

             # Project values
             time_proj = self.time_embed_proj(current_time_vals) # Shape: [batch, hidden_size]
             size_proj = self.size_embed_proj(current_size_vals) # Shape: [batch, hidden_size]

             # Add projections to the corresponding positions in the offset tensor
             # We only add if the original length of the sample was >= j+1
             # Create a mask for active samples at step j
             active_mask = torch.tensor([l > j for l in all_original_lens], dtype=torch.float32, device=device).unsqueeze(-1) # [batch, 1]

             continuous_embed_offset[:, time_token_seq_idx, :] += time_proj * active_mask
             continuous_embed_offset[:, size_token_seq_idx, :] += size_proj * active_mask


        # --- 4. Combine Embeddings and Pass Through BERT Encoder ---
        # Combine base embeddings with the continuous offsets
        inputs_embeds = word_embeddings + position_embeddings + continuous_embed_offset

        # Apply embedding LayerNorm and Dropout (mimicking BERT's embedding layer output)
        inputs_embeds = self.bert_model.base_model.embeddings.LayerNorm(inputs_embeds)
        inputs_embeds = self.bert_model.base_model.embeddings.dropout(inputs_embeds)

        # --- Pass through BERT model using inputs_embeds ---
        # PEFT-wrapped model should accept inputs_embeds
        outputs = self.bert_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True
        )

        # --- 5. Get Representation for Classification ---
        # Use the output corresponding to the [CLS] token (index 0)
        cls_output = outputs.last_hidden_state[:, 0, :]

        # --- 6. Pass Through Classification Head ---
        logits = self.fc_layers(cls_output) # Shape: [batch_size, 1]

        # Return raw logits (loss calculation should happen outside)
        return logits
    
    def calculate_boundaries(df_train, num_buckets=50, norm={}):
        """
        Calculate boundaries for time and size values based on the training data.
        """
        return [], []