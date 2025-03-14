from core.utils import PrintUtils

import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
import random
import os

class BaseClassifier(nn.Module):
    """
        Base classifier class that can be extended by different model architectures.
    """

    def __init__(self, kernel_width=3, max_len=100):
        """
            Creates an instance.
        """

        # Save members
        super().__init__()
        self.kernel_width = kernel_width
        self.max_len = max_len
        
    def forward(self, x):
        """
            Implements a forward pass.
        """

        # Not implemented
        raise NotImplementedError('Subclasses must implement forward method')
    
    def save(self, filepath, normalization_params=None):
        """
            Save model state dictionary and optionally normalization parameters.
        """

        # Saves a model
        torch.save(self.state_dict(), filepath)
        PrintUtils.print_extra(f'Model saved to file *{os.path.basename(filepath)}*')
        
        # Optionally saves normalization parameters
        if normalization_params:
            time_norm_params, size_norm_params, max_len = normalization_params
            norm_filepath = filepath.replace('.pth', '_norm_params.npz')
            self.save_normalization_params(time_norm_params, size_norm_params, max_len, norm_filepath)
    
    @classmethod
    def load(cls, filepath, kernel_width, max_len, device):
        """
            Load a model from a given file.
        """

        # Loads the model
        model = cls(kernel_width, max_len).to(device)
        model.load_state_dict(torch.load(filepath, map_location=device))
        model.eval()
        PrintUtils.print_extra(f'Model loaded from file *{os.path.basename(filepath)}*')
        return model
    
    def inference(self, input_data, device, normalization_params=None):
        """
            Runs inference on the input data.
            The input data can be a DataFrame or a tuple of (time_diffs, data_lengths).
            Returns the predicted probabilities and binary predictions.
        """

        # Evaluate
        self.eval()
        
        # Handle DataFrames
        if isinstance(input_data, pd.DataFrame):
            
            # If DataFrame is provided, normalize it and create a dataset
            if normalization_params is None:
                raise Exception('Normalization parameters must be provided for a DataFrame input')
            
            # Normalize
            time_norm_params, size_norm_params, max_len = normalization_params
            df_normalized = normalize_dataframe(input_data, time_norm_params, size_norm_params, max_len)
            dataset = PreprocessedTextDataset(df_normalized, max_len)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
           
            # Get all probabilities
            all_probs = []
            with torch.no_grad():
                for X, _ in dataloader:
                    X = X.to(device)
                    output = self(X)
                    all_probs.extend(output.cpu().numpy().flatten())
            
            # Get the prediction based on the probabilities and return them
            all_probs = np.array(all_probs)
            predictions = (all_probs > 0.5).astype(int)
            return all_probs, predictions
        
        # Handle a 2-tuple
        elif isinstance(input_data, tuple) and len(input_data) == 2:
            
            # If tuple of arrays is provided, normalize directly
            time_diffs, data_lengths = input_data
            if normalization_params is None:
                raise Exception('Normalization parameters must be provided for raw input')
            time_norm_params, size_norm_params, max_len = normalization_params
            
            # Normalize and prepare input
            normalized_time = []
            normalized_size = []
            for i, val in enumerate(time_diffs):
                if i < max_len:
                    mean, std = time_norm_params[i]
                    normalized_time.append((val - mean) / std)
            for i, val in enumerate(data_lengths):
                if i < max_len:
                    mean, std = size_norm_params[i]
                    normalized_size.append((val - mean) / std)
            
            # Pad sequences
            time_padded = np.zeros(max_len)
            size_padded = np.zeros(max_len)
            time_padded[:len(normalized_time)] = normalized_time[:max_len]
            size_padded[:len(normalized_size)] = normalized_size[:max_len]
            
            # Prepare tensor
            sample = np.stack([time_padded, size_padded], axis=0)
            tensor_input = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Run inference
            with torch.no_grad():
                output = self(tensor_input)
                prob = output.cpu().numpy().flatten()[0]
                prediction = 1 if prob > 0.5 else 0
            
            # Return the probability and prediction
            return prob, prediction

        # Unsupported format
        else:
            raise Exception(f'Input must be either a DataFrame or a tuple of (time_diffs, data_lengths)')
    
    @staticmethod
    def save_normalization_params(time_norm_params, size_norm_params, max_len, filename='normalization_params.npz'):
        """
            Saves the normalization parameters to a file.
        """

        # Save data
        np.savez(
            filename, 
            time_norm_params=np.array(time_norm_params, dtype=object),
            size_norm_params=np.array(size_norm_params, dtype=object),
            max_len=np.array([max_len])
        )
        PrintUtils.print_extra(f'Normalization parameters saved to {os.path.basename(filename)}')
    
    @staticmethod
    def load_normalization_params(filename='normalization_params.npz'):
        """
            Loads the normalization parameters from a file.
        """

        # Load data and return as a Tuple
        loaded = np.load(filename, allow_pickle=True)
        time_norm_params = loaded['time_norm_params']
        size_norm_params = loaded['size_norm_params']
        max_len = int(loaded['max_len'][0])
        return time_norm_params, size_norm_params, max_len

class CNNBinaryClassifier(BaseClassifier):
    """
        CNN model for binary classification of sequential data.
    """

    def __init__(self, kernel_width, max_len):
        """
            Creates an instance.
        """

        # Initialize
        super().__init__(kernel_width, max_len)
        
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
        """
            Implements a forward pass.
        """

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
       
        # Return result
        return x

class RNNBinaryClassifier(BaseClassifier):
    """
        Enhanced RNN model for binary classification of sequential data.
    """

    def __init__(self, kernel_width, max_len, hidden_size=128, num_layers=2, dropout_rate=0.3):
        """
            Creates an instance.
        """

        # Initialize
        super().__init__(kernel_width, max_len)

        # Saves members
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len
        
        # Feature embedding layers to better represent the inputs
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )
        self.size_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )
        
        # Combined feature dimension after embedding
        self.feature_dim = 64  # 32 + 32
        
        # Main LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Attention mechanism for focusing on important parts of the sequence
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),  # *2 for bidirectional
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Multiple fully connected layers with batch normalization
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
            Initialize weights for faster convergence.
        """

        # Initialize the network
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def _apply_attention(self, lstm_output, mask=None):
        """
            Apply attention mechanism to LSTM output.
        """
        
        # The lstm_output shape: [batch_size, seq_len, hidden_size*2]
        attention_weights = self.attention(lstm_output)  # [batch_size, seq_len, 1]
        
        # Apply mask if provided (for padded sequences)
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask.unsqueeze(2) == 0, float('-inf'))
        
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention weights to LSTM output
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # [batch_size, hidden_size*2]
        return context_vector, attention_weights
        
    def forward(self, x):
        """
            Implements a forward pass.
        """

        # Initialize the batch size
        batch_size = x.size(0)
        
        # Create a mask for padding (1 for real values, 0 for padding)
        # Assuming padding values are exactly 0
        mask = (x.sum(dim=1) != 0).float()  # [batch_size, max_len]
        
        # Split time and size features
        time_features = x[:, 0, :].unsqueeze(2)  # [batch_size, max_len, 1]
        size_features = x[:, 1, :].unsqueeze(2)  # [batch_size, max_len, 1]
        
        # Apply embedding layers
        time_embedded = self.time_embedding(time_features)  # [batch_size, max_len, 32]
        size_embedded = self.size_embedding(size_features)  # [batch_size, max_len, 32]
        
        # Concatenate embedded features
        embedded = torch.cat([time_embedded, size_embedded], dim=2)  # [batch_size, max_len, 64]
        
        # Pack the padded sequence for LSTM efficiency
        # This helps the LSTM ignore padded time steps
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            lengths=mask.sum(dim=1).cpu().int(),
            batch_first=True,
            enforce_sorted=False
        )
        
        # Pass through LSTM
        packed_output, _ = self.lstm(packed_embedded)
        
        # Unpack the sequence
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, 
            batch_first=True,
            total_length=self.max_len
        )
        
        # Apply attention mechanism
        context_vector, _ = self._apply_attention(lstm_output, mask)
        
        # Pass through fully connected layers with batch normalization
        output = self.fc_layers(context_vector)
        
        # Apply sigmoid for binary classification
        output = torch.sigmoid(output)
        
        # Return result
        return output

class PreprocessedTextDataset(Dataset):
    """
        Dataset class for preprocessed time series data.
    """

    def __init__(self, df, max_len):
        """
            Creates an instance.
        """

        # Save members
        self.df = df.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self):
        """
            Length override.
        """

        # Return the dataframe length
        return len(self.df)

    def __getitem__(self, idx):
        """
            Accessor override.
        """

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

def prepare_data(df):
    """
        Load and prepare datasets from a dataframe files.
    """

    # Preprocess datasets
    df['time_diffs'] = df['time_diffs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['data_lengths'] = df['data_lengths'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    len_90p = int(np.percentile(df['data_lengths'].apply(len), 90))
    return df, len_90p

def calculate_norm_params(data, max_len):
    """
        Calculate normalization parameters for each position in the sequence.
    """

    # Log
    PrintUtils.start_stage('Calculating normalization parameters')
    time_norm_params, size_norm_params = [], []
    
    # Iterate all rows
    for i in range(max_len):

        # Extract the values
        time_vals = [row[i] for row in data['time_diffs'] if len(row) > i]
        size_vals = [row[i] for row in data['data_lengths'] if len(row) > i]
        if time_vals:
            time_mean, time_std = np.mean(time_vals), np.std(time_vals)
            time_norm_params.append((time_mean, time_std if time_std > 0 else 1.0))
            PrintUtils.print_extra(f'Position *{i}*: time_mean=*{time_mean:.4f}*, time_std=*{time_std:.4f}*')
        else:
            time_norm_params.append((0, 1))
        if size_vals:
            size_mean, size_std = np.mean(size_vals), np.std(size_vals)
            size_norm_params.append((size_mean, size_std if size_std > 0 else 1.0))
            PrintUtils.print_extra(f'Position *{i}*: size_mean=*{size_mean:.4f}*, size_std=*{size_std:.4f}*')
        else:
            size_norm_params.append((0, 1))

    # Return the normalized parameters
    PrintUtils.end_stage()
    return time_norm_params, size_norm_params

def normalize_dataframe(df, time_norm_params, size_norm_params, max_len):
    """
        Normalize time_diffs and data_lengths in the dataframe.
    """

    # Log
    PrintUtils.start_stage('Normalizing dataframe features')
    
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
    
    # Return as a tuple
    PrintUtils.end_stage()
    df_normalized['normalized_time_diffs'] = normalized_time_diffs
    df_normalized['normalized_data_lengths'] = normalized_data_lengths
    return df_normalized

class EarlyStopping(object):
    """
        Early stopping to prevent overfitting.
    """

    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        """
            Creates an instance.
            The patience is an integer that specifies how many epochs to wait after last improvement.
            The delta is a floating point number that containsthe minimum change to qualify as an improvement.
        """

        # Save members
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = 0
        self.delta = delta
        self.path = path
        
    def __call__(self, val_acc, model):
        """
            Call override.
        """

        # Performs the early stopping
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            PrintUtils.print_extea(f'EarlyStopping counter: *{self.counter}* out of *{self.patience}*')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0
            
    def save_checkpoint(self, val_acc, model):
        """
            Saves model when validation accuracy improves.
        """
        
        # Save data
        if self.verbose:
            PrintUtils.print_extra(f'Validation accuracy improved (*{self.val_acc_max:.6f}* --> *{val_acc:.6f}*). Saving model.')
        torch.save(model.state_dict(), self.path)

        # Override accuracy value 
        self.val_acc_max = val_acc

def set_seed(seed=42):
    """
        Set the random seed for reproducibility.
    """

    # Set the seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    PrintUtils.print_extra(f'Random seed set to *{seed}*')

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
        Train the model for one epoch.
    """

    # Train model
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    PrintUtils.start_stage('Training')
    for X, y in dataloader:
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
        PrintUtils.start_stage(f'Training (loss = {loss.item():.4f}, accuracy = {correct/total:.4f})', override_prev=True)
   
    # Return epoch loss and accuracy
    PrintUtils.end_stage()
    epoch_loss = total_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def eval_epoch(model, dataloader, criterion, device):
    """
        Evaluate the model on the validation set.
    """

    # Evaluate model
    model.eval()
    total_loss = 0
    preds, labels = [], []
   
    # Conclude accuracy
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
    """
        Get raw prediction scores and true labels from dataloader.
    """

    # Evaluate model
    model.eval()
    all_scores = []
    all_labels = []

    # Iterate all data in dataloader
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X)
            all_scores.extend(outputs.cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())
    return np.array(all_scores), np.array(all_labels)

