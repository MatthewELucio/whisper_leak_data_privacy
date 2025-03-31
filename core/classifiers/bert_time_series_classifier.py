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

# Suppress excessive warnings from transformers library (optional)
hf_logging.set_verbosity_error() 

# Assuming base_classifier.py is in the same directory or accessible via python path
from .base_classifier import BaseClassifier 

class BERTTimeSeriesClassifier(BaseClassifier):
    """
    A classifier using a pre-trained BERT model adapted for time series data.
    It buckets normalized time_diffs and data_lengths, converts them to new tokens,
    interleaves them, and feeds them into the transformer model for binary classification.
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
                 interleave=True
                 ):
        """
        Creates an instance of the BERTTimeSeriesClassifier.

        Args:
            normalization_params (tuple): A tuple containing 
                (time_mean, time_std, size_mean, size_std, max_len).
            time_boundaries_norm (list/np.array): Quantile boundaries for normalized time_diffs. 
                                                  Should have num_buckets - 1 elements.
            len_boundaries_norm (list/np.array): Quantile boundaries for normalized data_lengths.
                                                 Should have num_buckets - 1 elements.
            num_buckets (int): The number of buckets to divide time and length data into.
            model_name (str): The name of the pre-trained model from Hugging Face Hub.
            fc_dims (list): List of dimensions for fully connected layers in the classification head.
            dropout_rate (float): Dropout rate for regularization in the classification head.
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
        
        # Original max sequence length (from normalization params)
        self.original_max_len = self.max_len 
        # Max length for BERT input (interleaved + [CLS] + [SEP])
        if self.interleave:
            self.bert_max_len = 2 * self.original_max_len + 2 
        else:
            self.bert_max_len = self.original_max_len + 2
        
        if self.bert_max_len > 512:
            print(f"Warning: BERT max input length is 512. Reducing max_len to 255.")
            self.bert_max_len = 512
        
        self.args = {
            'model_name': self.model_name,
            'num_buckets': self.num_buckets,
            # Store boundaries as lists for JSON serialization
            'time_boundaries_norm': self.time_boundaries_norm.tolist(), 
            'len_boundaries_norm': self.len_boundaries_norm.tolist(),
            'fc_dims': self.fc_dims,
            'dropout_rate': self.dropout_rate,
            # Note: normalization_params are handled by the BaseClassifier save/load
        }

        # --- Load Tokenizer and Model ---
        # Use cache_dir to potentially control download location
        # cache_dir = "./hf_cache" 
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        # self.bert_model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Default behavior: downloads to ~/.cache/huggingface/hub
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Loading model: {model_name}")
        self.bert_model = AutoModel.from_pretrained(model_name)

        # --- Add New Tokens ---
        self.len_tokens = [f"[LEN_{i}]" for i in range(num_buckets)]
        self.time_tokens = [f"[TIME_{i}]" for i in range(num_buckets)]
        special_tokens_to_add = self.len_tokens + self.time_tokens
        
        num_added_toks = self.tokenizer.add_tokens(special_tokens_to_add, special_tokens=True)
        print(f"Added {num_added_toks} new tokens to tokenizer.")
        
        # --- Resize Model Embeddings ---
        self.bert_model.resize_token_embeddings(len(self.tokenizer))
        print(f"Resized model embeddings to: {len(self.tokenizer)}")

        # --- Get Token IDs ---
        self.len_token_ids = self.tokenizer.convert_tokens_to_ids(self.len_tokens)
        self.time_token_ids = self.tokenizer.convert_tokens_to_ids(self.time_tokens)
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        
        if self.pad_token_id is None:
             # Some models might not have a default pad token (like GPT2)
             # Add one if missing, although DistilBERT should have it.
             print("Warning: Tokenizer has no default pad token. Adding '[PAD]'.")
             self.tokenizer.add_tokens(['[PAD]'], special_tokens=True)
             self.pad_token_id = self.tokenizer.pad_token_id
             self.bert_model.resize_token_embeddings(len(self.tokenizer))


        # --- Classification Head ---
        bert_output_dim = self.bert_model.config.hidden_size
        
        fc_layers = []
        input_dim = bert_output_dim
        
        for dim in fc_dims:
            fc_layers.extend([
                nn.Linear(input_dim, dim),
                # Consider BatchNorm or LayerNorm depending on performance
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

    def _bucket_and_tokenize_sample(self, norm_time_vals, norm_size_vals):
        """Buckets, tokenizes, and interleaves a single sample."""
        
        # 1. Determine actual sequence length (before padding)
        # Assumes padding is 0. Find the effective length.
        # A safer way might be to pass lengths/mask from Loader if possible.
        seq_len = len(norm_time_vals) # Assuming input arrays are already trimmed

        # 2. Bucket the normalized values
        # np.digitize returns indices starting from 1. We subtract 1 for 0-based index.
        # Bins: (-inf, b0], (b0, b1], ..., (bN-2, inf)
        # Indices:     1          2      ...     N
        time_bucket_indices = np.digitize(norm_time_vals[:seq_len], self.time_boundaries_norm)
        len_bucket_indices = np.digitize(norm_size_vals[:seq_len], self.len_boundaries_norm)

        # 3. Map bucket indices to token IDs
        time_tok_ids = [self.time_token_ids[idx] for idx in time_bucket_indices]
        len_tok_ids = [self.len_token_ids[idx] for idx in len_bucket_indices]

        # 4. Interleave tokens
        if self.interleave:
            interleaved_tokens = []
            for i in range(seq_len):
                interleaved_tokens.append(len_tok_ids[i])
                interleaved_tokens.append(time_tok_ids[i])
        else:
            interleaved_tokens = len_tok_ids

        # 5. Add special tokens ([CLS] at start, [SEP] at end)
        final_token_ids = [self.cls_token_id] + interleaved_tokens + [self.sep_token_id]
        
        # 6. Pad sequence to self.bert_max_len
        padding_len = self.bert_max_len - len(final_token_ids)
        if padding_len < 0:
            # If interleaved sequence exceeds max length, truncate
            # Truncating from the end might lose the most recent info.
            # Consider alternative strategies if this happens often.
            # print(f"Warning: Sequence truncated from {len(final_token_ids)} to {self.bert_max_len}")
            final_token_ids = final_token_ids[:self.bert_max_len-1] + [self.sep_token_id]
            attention_mask = [1] * self.bert_max_len
        else:
            final_token_ids = final_token_ids + ([self.pad_token_id] * padding_len)
            attention_mask = ([1] * (self.bert_max_len - padding_len)) + ([0] * padding_len)
            
        return final_token_ids, attention_mask

    def forward(self, x):
        """
        Implements the forward pass: bucketing, tokenization, BERT, classification head.
        
        Args:
            x (torch.Tensor): Input tensor from Loader of shape 
                              [batch_size, 2, original_max_len], containing
                              *normalized* and *padded* time_diffs and data_lengths.
                              x[:, 0, :] = normalized time_diffs
                              x[:, 1, :] = normalized data_lengths
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
            # This assumes padding is zero. If Loader provides masks, use them instead.
            # Find the indices of the last non-zero element. Max length if all non-zero.
            non_zero_time_idx = np.where(norm_time_padded != 0)[0]
            non_zero_size_idx = np.where(norm_size_padded != 0)[0]
            
            # Use the maximum length found in either sequence as the original length
            original_len = 0
            if len(non_zero_time_idx) > 0:
                original_len = max(original_len, non_zero_time_idx[-1] + 1)
            if len(non_zero_size_idx) > 0:
                original_len = max(original_len, non_zero_size_idx[-1] + 1)
                
            # If all elements were zero (empty sequence?), handle gracefully.
            # This case might need specific handling depending on data.
            if original_len == 0:
                # Option 1: Treat as length 1 with zero values (might lead to bucket 0)
                # original_len = 1 
                # Option 2: Create a minimal sequence [CLS] [SEP] + padding
                 input_ids = [self.cls_token_id, self.sep_token_id] + [self.pad_token_id] * (self.bert_max_len - 2)
                 attention_mask = [1, 1] + [0] * (self.bert_max_len - 2)
                 
            else:
                # Ensure we don't exceed the expected max_len from normalization_params
                original_len = min(original_len, self.original_max_len) 

                # Extract the actual data (un-pad)
                norm_time_vals = norm_time_padded[:original_len]
                norm_size_vals = norm_size_padded[:original_len]

                # Bucket, tokenize, interleave, pad for this sample
                input_ids, attention_mask = self._bucket_and_tokenize_sample(norm_time_vals, norm_size_vals)
            
            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)

        # Convert lists to tensors
        input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long).to(device)
        attention_mask_tensor = torch.tensor(all_attention_masks, dtype=torch.long).to(device)

        # --- Pass through BERT model ---
        outputs = self.bert_model(
            input_ids=input_ids_tensor,
            attention_mask=attention_mask_tensor
        )

        # --- Get representation for classification ---
        # Use the output of the [CLS] token (first token)
        cls_output = outputs.last_hidden_state[:, 0, :] 
        # Alternative: Pool the sequence outputs (e.g., mean pooling)
        # sequence_output = outputs.last_hidden_state
        # pooled_output = torch.mean(sequence_output * attention_mask_tensor.unsqueeze(-1), dim=1)

        # --- Pass through classification head ---
        logits = self.fc_layers(cls_output) # Shape: [batch_size, 1]
        probs = torch.sigmoid(logits)       # Shape: [batch_size, 1]
        
        return probs # Return shape [batch_size, 1]
    
    @staticmethod
    def calculate_boundaries(df, num_buckets, norm):
        """
        Calculates quantile boundaries for normalized time_diffs and data_lengths.

        This method processes a DataFrame, normalizes the specified columns using
        the provided parameters, and computes quantile boundaries suitable for 
        bucketing in the BERTTimeSeriesClassifier.

        Args:
            df (pd.DataFrame): DataFrame containing 'time_diffs' and 'data_lengths' columns.
                               Values can be lists or string representations of lists.
            num_buckets (int): The number of buckets to create (N). This method will
                               calculate N-1 quantile boundaries.
            normalization_params (tuple): A tuple containing the pre-calculated normalization
                                          parameters: (time_mean, time_std, size_mean, size_std, max_len).

        Returns:
            tuple: A tuple containing two numpy arrays:
                   (time_boundaries_norm, len_boundaries_norm).
                   Each array contains num_buckets - 1 boundary values.

        Raises:
            ValueError: If columns 'time_diffs' or 'data_lengths' are missing,
                        if num_buckets is less than 2, or if normalization_params
                        are invalid.
            TypeError: If normalization_params is not a tuple/list of length 5.
        """
        # --- Input Validation ---
        if 'time_diffs' not in df.columns or 'data_lengths' not in df.columns:
            raise ValueError("DataFrame must contain 'time_diffs' and 'data_lengths' columns.")
        if not isinstance(num_buckets, int) or num_buckets < 2:
             raise ValueError("num_buckets must be an integer greater than or equal to 2.")
        if not isinstance(norm, (dict)):
            raise TypeError("Normalization parameters must be a dictionary.")
        if not all(key in norm for key in ["time_mean", "time_std", "size_mean", "size_std"]):
            raise ValueError("Normalization parameters must include 'time_mean', 'time_std', 'size_mean', and 'size_std'.")
        
        time_mean = norm["time_mean"]
        time_std = norm["time_std"]
        size_mean = norm["size_mean"]
        size_std = norm["size_std"]

        # Ensure standard deviations are positive for safe division
        time_std_safe = time_std if time_std > 0 else 1.0
        size_std_safe = size_std if size_std > 0 else 1.0

        # --- Data Preparation and Normalization ---
        all_norm_time_vals = []
        all_norm_size_vals = []

        for _, row in df.iterrows():
            # Handle string representations safely
            try:
                time_vals = ast.literal_eval(row['time_diffs']) if isinstance(row['time_diffs'], str) else row['time_diffs']
                size_vals = ast.literal_eval(row['data_lengths']) if isinstance(row['data_lengths'], str) else row['data_lengths']
            except (ValueError, SyntaxError) as e:
                 print(f"Warning: Skipping row due to parsing error: {e}. Row content:\n{row}")
                 continue # Skip rows that cannot be parsed

            if not isinstance(time_vals, list) or not isinstance(size_vals, list):
                 print(f"Warning: Skipping row due to unexpected data type. Expected lists. Row content:\n{row}")
                 continue # Skip rows with unexpected types

            # Normalize and add to flattened lists
            all_norm_time_vals.extend([(val - time_mean) / time_std_safe for val in time_vals])
            all_norm_size_vals.extend([(val - size_mean) / size_std_safe for val in size_vals])

        if not all_norm_time_vals or not all_norm_size_vals:
             raise ValueError("After processing, no valid time or size values were found in the DataFrame.")

        # --- Quantile Calculation ---
        # Calculate N-1 quantiles to define N buckets
        # Quantiles range from 1/N to (N-1)/N
        quantiles = np.linspace(0, 1, num_buckets + 1)[1:-1] 
        
        # Use np.quantile on the *normalized* flattened data
        time_boundaries_norm = np.quantile(all_norm_time_vals, quantiles)
        len_boundaries_norm = np.quantile(all_norm_size_vals, quantiles)

        # Ensure boundaries are unique (can happen with sparse data or low num_buckets)
        # If duplicates found, small adjustments might be needed, but np.quantile usually handles this.
        # For simplicity, we'll rely on np.quantile here. Consider adding checks/adjustments if needed.
        if len(np.unique(time_boundaries_norm)) != num_buckets - 1:
             print(f"Warning: Duplicate time boundaries found. This might occur with skewed data or low num_buckets.")
        if len(np.unique(len_boundaries_norm)) != num_buckets - 1:
             print(f"Warning: Duplicate length boundaries found. This might occur with skewed data or low num_buckets.")


        return time_boundaries_norm, len_boundaries_norm
