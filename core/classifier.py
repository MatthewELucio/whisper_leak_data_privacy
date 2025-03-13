import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
import logging
from tqdm import tqdm
import random

class CNNBinaryClassifier(nn.Module):
    """CNN model for binary classification of sequential data."""
    def __init__(self, kernel_width, max_len):
        super().__init__()
        # More layers for better reasoning
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(2, kernel_width), padding=(0, kernel_width // 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, kernel_width), padding=(0, kernel_width // 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(1, kernel_width), padding=(0, kernel_width // 2))
        self.bn3 = nn.BatchNorm2d(128)
        
        # Calculate output size after convolutions (accounting for padding)
        self.pool = nn.AdaptiveAvgPool2d((1, max_len // 2))
        
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * (max_len // 2), 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Input shape: [batch_size, 2, max_len]
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, 2, max_len]
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        
        # Pooling
        x = self.pool(x)
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        
        # Fully connected layers
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        
        return x


def save_model(model, filepath, normalization_params=None):
    """Save model state dictionary and optionally normalization parameters."""
    torch.save(model.state_dict(), filepath)
    logging.info(f"Model saved to {filepath}")
    
    if normalization_params:
        time_norm_params, size_norm_params, max_len = normalization_params
        norm_filepath = filepath.replace('.pth', '_norm_params.npz')
        save_normalization_params(time_norm_params, size_norm_params, max_len, norm_filepath)


def load_model(filepath, kernel_width, max_len, device):
    """Load model from file."""
    model = CNNBinaryClassifier(kernel_width, max_len).to(device)
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.eval()
    logging.info(f"Model loaded from {filepath}")
    return model


class PreprocessedTextDataset(Dataset):
    """Dataset class for preprocessed time series data."""
    def __init__(self, df, max_len):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        time_vals = row['normalized_time_diffs']
        size_vals = row['normalized_data_lengths']

        # Pad sequences to max_len
        time_padded = np.zeros(self.max_len)
        size_padded = np.zeros(self.max_len)
        
        time_padded[:len(time_vals)] = time_vals[:self.max_len]
        size_padded[:len(size_vals)] = size_vals[:self.max_len]

        sample = np.stack([time_padded, size_padded], axis=0)
        target = row['target']

        return torch.tensor(sample, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


def prepare_data(df):
    """Load and prepare datasets from a dataframe files."""
    logging.info("Loading datasets...")

    # Preprocess datasets
    logging.info("Preprocessing datasets...")
    df['time_diffs'] = df['time_diffs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['data_lengths'] = df['data_lengths'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    len_90p = int(np.percentile(df['data_lengths'].apply(len), 90))
    logging.info(f"Sequence length (90th percentile): {len_90p}")

    return df, len_90p


def calculate_norm_params(data, max_len):
    """Calculate normalization parameters for each position in the sequence."""
    logging.info("Calculating normalization parameters...")
    time_norm_params, size_norm_params = [], []
    
    for i in range(max_len):
        time_vals = [row[i] for row in data['time_diffs'] if len(row) > i]
        size_vals = [row[i] for row in data['data_lengths'] if len(row) > i]

        if time_vals:
            time_mean, time_std = np.mean(time_vals), np.std(time_vals)
            time_norm_params.append((time_mean, time_std if time_std > 0 else 1.0))
            logging.debug(f"Position {i}: time_mean={time_mean:.4f}, time_std={time_std:.4f}")
        else:
            time_norm_params.append((0, 1))
            
        if size_vals:
            size_mean, size_std = np.mean(size_vals), np.std(size_vals)
            size_norm_params.append((size_mean, size_std if size_std > 0 else 1.0))
            logging.debug(f"Position {i}: size_mean={size_mean:.4f}, size_std={size_std:.4f}")
        else:
            size_norm_params.append((0, 1))

    return time_norm_params, size_norm_params


def normalize_dataframe(df, time_norm_params, size_norm_params, max_len):
    """Normalize time_diffs and data_lengths in the dataframe."""
    logging.info("Normalizing dataframe features...")
    
    # Create deep copies to avoid modifying the original series
    df_normalized = df.copy()
    normalized_time_diffs = []
    normalized_data_lengths = []
    
    for idx, row in df.iterrows():
        time_vals, size_vals = row['time_diffs'], row['data_lengths']
        
        # Normalize time_diffs
        norm_time = []
        for i, val in enumerate(time_vals):
            if i < max_len:
                mean, std = time_norm_params[i]
                norm_time.append((val - mean) / std)
        normalized_time_diffs.append(norm_time)
        
        # Normalize data_lengths
        norm_size = []
        for i, val in enumerate(size_vals):
            if i < max_len:
                mean, std = size_norm_params[i]
                norm_size.append((val - mean) / std)
        normalized_data_lengths.append(norm_size)
    
    df_normalized['normalized_time_diffs'] = normalized_time_diffs
    df_normalized['normalized_data_lengths'] = normalized_data_lengths
    
    return df_normalized


def save_normalization_params(time_norm_params, size_norm_params, max_len, filename="normalization_params.npz"):
    """Save normalization parameters to a file."""
    np.savez(
        filename, 
        time_norm_params=np.array(time_norm_params, dtype=object),
        size_norm_params=np.array(size_norm_params, dtype=object),
        max_len=np.array([max_len])
    )
    logging.info(f"Normalization parameters saved to {filename}")


def load_normalization_params(filename="normalization_params.npz"):
    """Load normalization parameters from file."""
    loaded = np.load(filename, allow_pickle=True)
    time_norm_params = loaded['time_norm_params']
    size_norm_params = loaded['size_norm_params']
    max_len = int(loaded['max_len'][0])
    return time_norm_params, size_norm_params, max_len


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            verbose (bool): If True, prints a message for each improvement
            delta (float): Minimum change to qualify as an improvement
            path (str): Path for the checkpoint to be saved to
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0
        self.delta = delta
        self.path = path
        
    def __call__(self, val_acc, model):
        score = val_acc
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            logging.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0
            
    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation accuracy improves.'''
        if self.verbose:
            logging.info(f'Validation accuracy improved ({self.val_acc_max:.6f} --> {val_acc:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_acc_max = val_acc


def set_seed(seed=42):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Random seed set to {seed}")


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for X, y in progress_bar:
        X, y = X.to(device), y.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
        pred = (output > 0.5).float()
        correct += (pred == y).sum().item()
        total += y.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})
    
    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc


def eval_epoch(model, dataloader, criterion, device):
    """Evaluate the model on the validation set."""
    model.eval()
    total_loss = 0
    preds, labels = [], []
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).unsqueeze(1)
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item() * X.size(0)
            preds += (output > 0.5).cpu().numpy().astype(int).flatten().tolist()
            labels += y.cpu().numpy().astype(int).flatten().tolist()
    
    accuracy = accuracy_score(labels, preds)
    return total_loss / len(dataloader.dataset), accuracy


def get_prediction_scores(model, dataloader, device):
    """Get raw prediction scores and true labels from dataloader."""
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X)
            all_scores.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())
    
    return np.array(all_scores), np.array(all_labels)