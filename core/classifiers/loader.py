import torch
from core.utils import PrintUtils

import ast
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
import random

class Loader(Dataset):
    """
        Dataset class for preprocessed time series data.
    """

    def __init__(self, df):
        """
            Creates an instance.
        """

        # Save members
        self.df = df.reset_index(drop=True)

        # Validate columns time_diffs, data_lengths, and target
        if 'time_diffs' not in df.columns or 'data_lengths' not in df.columns or 'target' not in df.columns:
            raise ValueError('Dataframe must contain columns: time_diffs, data_lengths, and target.')

        self.normalized = False

    def __len__(self):
        """
            Length override.
        """

        # Return the dataframe length
        return len(self.df)
    
    def normalize(self, normalization_params=None):
        """
        Normalizes time_diffs and data_lengths uses Z-score normalization (standardization).

        Returns: (time_mean, time_std, size_mean, size_std, max_len)
        """
        time_mean, time_std, size_mean, size_std, max_len = normalization_params if normalization_params else (None, None, None, None, None)
        if max_len is not None:
            self.max_len = max_len
        
        if time_mean is None:
            # Preprocess datasets
            self.df.loc[:, 'time_diffs'] = self.df['time_diffs'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            self.df.loc[:, 'data_lengths'] = self.df['data_lengths'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            self.max_len = int(np.percentile(self.df['data_lengths'].apply(len), 95))
            
            # Flatten the lists to calculate global statistics
            all_time_vals = [val for row in self.df['time_diffs'] for val in row]
            all_size_vals = [val for row in self.df['data_lengths'] for val in row]
            
            # Calculate mean and standard deviation for time_diffs
            time_mean = np.mean(all_time_vals)
            time_std = np.std(all_time_vals)
            if time_std <= 0:
                time_std = 1.0  # Avoid division by zero
            
            # Calculate mean and standard deviation for data_lengths
            size_mean = np.mean(all_size_vals)
            size_std = np.std(all_size_vals)
            if size_std <= 0:
                size_std = 1.0  # Avoid division by zero
            
            # Log the normalization parameters
            PrintUtils.print_extra(f'Global time normalization: mean=*{time_mean:.4f}*, std=*{time_std:.4f}*')
            PrintUtils.print_extra(f'Global size normalization: mean=*{size_mean:.4f}*, std=*{size_std:.4f}*')
        
        # Normalize the dataframe
        self._apply_normalization(time_mean, time_std, size_mean, size_std)

        return (time_mean, time_std, size_mean, size_std, self.max_len)
    
    def _apply_normalization(self, time_mean, time_std, size_mean, size_std):
        """
        Normalize time_diffs and data_lengths in the dataframe using Z-score normalization.
        Applies a single normalization strategy for each column.
        """
        
        # Create deep copy to avoid modifying the original dataframe
        df_normalized = self.df.copy()
        
        # Apply normalization
        normalized_time_diffs = []
        normalized_data_lengths = []
        
        for idx, row in self.df.iterrows():
            time_vals, size_vals = row['time_diffs'], row['data_lengths']
            
            # Normalize time_diffs
            norm_time = [(val - time_mean) / time_std for val in time_vals]
            normalized_time_diffs.append(norm_time)
            
            # Normalize data_lengths
            norm_size = [(val - size_mean) / size_std for val in size_vals]
            normalized_data_lengths.append(norm_size)
        
        # Add normalized features to the dataframe
        df_normalized['normalized_time_diffs'] = normalized_time_diffs
        df_normalized['normalized_data_lengths'] = normalized_data_lengths
        self.df = df_normalized
        self.normalized = True
        
        PrintUtils.end_stage()
        return

    def __getitem__(self, idx):
        """
            Accessor override.
        """
        if not self.normalized:
            raise ValueError('Data must be normalized before accessing items.')

        # Get the time and data lengths
        row = self.df.iloc[idx]
        time_vals = row['normalized_time_diffs']
        size_vals = row['normalized_data_lengths']

        # Pad sequences to max_len
        time_padded = np.zeros(self.max_len)
        size_padded = np.zeros(self.max_len)
        time_padded[:len(time_vals)] = time_vals[:self.max_len]
        size_padded[:len(size_vals)] = size_vals[:self.max_len]

        # Return the tensor
        sample = np.stack([time_padded, size_padded], axis=0)
        target = row['target']
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
