"""
This script demonstrates:
1. How to normalize variable‐length arrays ("time" and "size") in a dataset based on index‐specific 
   5th and 95th percentiles.
2. How to pad the arrays to a fixed maximum length N.
3. How to build a PyTorch CNN model that takes a 2×N tensor as input (with the two rows corresponding 
   to the normalized "time" and "size" arrays concatenated) and outputs multi-class predictions.
4. How to perform N-fold cross validation (here using KFold) to tune a hyperparameter (the kernel width).
5. How to load the test set, normalize it in the same way, run inference with the final model, and 
   output the results to a CSV file.

Assumptions:
• The CSV file "train_set.csv" is produced by your preprocessing code.
• The CSV file "test_set.csv" is produced similarly, containing the same columns: "prompt", "time", "size".
• The "time" and "size" columns are stored as string representations of lists (e.g. "[0.1, 0.2, ...]")
• The class label is given by the "prompt" column.
• The normalization parameters (5th and 95th percentiles) and max sequence length are computed from the 
  training data, and then applied to the test data.
"""

import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import random

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

############################################
# 0. Convert data from pickel              #
############################################
# Load the data
with open('geoff2.pickle', 'rb') as fp:
    data = pickle.load(fp)

# Convert it into a dict to a list of dicts
data_list = []
for prompt, data in data.items():
    for i in range(len(data)):
        data_list.append({
            'prompt': prompt,
            'time': [x[0] for x in data[i]],
            'size': [x[1] for x in data[i]]
        })

# Convert to dataframe
df = pd.DataFrame(data_list)

# Summarize count by prompt
print(df['prompt'].value_counts())

# Split the data into train and test sets, stratifying by prompt
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['prompt'], random_state=42)

# Print stats
print("Train set size:", len(train_df))
print("Test set size:", len(test_df))

print(train_df.head())

# Iterate over the train set, and study the count of each size by prompt
for prompt in train_df['prompt'].unique():
    prompt_df = train_df[train_df['prompt'] == prompt]
    print(f"Prompt: {prompt}")
    print(prompt_df['size'].value_counts())
    print()

# Save the train and test sets
train_df.to_csv('train_set.csv', index=False)
test_df.to_csv('test_set.csv', index=False)

############################################
# 1. Data Preprocessing for Training Set   #
############################################

# Load the training CSV file (assumed to be created earlier)
df = pd.read_csv('train_set.csv')

# Convert string representations of lists to actual lists (if needed)
df['time'] = df['time'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['size'] = df['size'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Determine maximum length N across both "time" and "size" in the training set.
max_len_time = df['time'].apply(len).max()
max_len_size = df['size'].apply(len).max()
max_len = int(max(max_len_time, max_len_size))
print("Training Max sequence length (N):", max_len)

# For each index position from 0 to N-1, compute the 5th and 95th percentiles 
# (separately for time and size) over all rows that have a value at that index.
time_norm_params = []
size_norm_params = []
for i in range(max_len):
    # Aggregate the i-th element from rows where the array length > i
    time_vals = []
    size_vals = []
    for _, row in df.iterrows():
        if len(row['time']) > i:
            time_vals.append(row['time'][i])
        if len(row['size']) > i:
            size_vals.append(row['size'][i])
    # For time: if no values are found at index i, default to (0,1)
    if time_vals:
        p5_time = np.percentile(time_vals, 5)
        p95_time = np.percentile(time_vals, 95)
    else:
        p5_time, p95_time = 0, 1
    time_norm_params.append((p5_time, p95_time))
    
    # For size:
    if size_vals:
        p5_size = np.percentile(size_vals, 5)
        p95_size = np.percentile(size_vals, 95)
    else:
        p5_size, p95_size = 0, 1
    size_norm_params.append((p5_size, p95_size))

#############################################
# 2. Dataset Preparation for Train/Test     #
#############################################

# Create a PyTorch Dataset that normalizes and pads each sample.
class TextDataset(Dataset):
    def __init__(self, df, time_norm_params, size_norm_params, max_len, prompt2label=None):
        self.data = df.reset_index(drop=True)
        # For train dataset, we create a mapping; for test dataset, we can pass an existing mapping.
        if prompt2label is None:
            prompts = self.data['prompt'].unique()
            self.prompt2label = {prompt: idx for idx, prompt in enumerate(prompts)}
        else:
            self.prompt2label = prompt2label
        self.time_norm_params = time_norm_params
        self.size_norm_params = size_norm_params
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Get the raw lists
        time_list = row['time']
        size_list = row['size']
        
        # Normalize and pad each list to length max_len.
        time_norm = []
        size_norm = []
        for i in range(self.max_len):
            # Normalize time values if available, else pad with 0.
            if i < len(time_list):
                t_val = time_list[i]
                p5, p95 = self.time_norm_params[i]
                # Linear normalization: values at 5th percentile become 0.0 and at 95th become 1.0.
                t_norm = (t_val - p5) / (p95 - p5) if (p95 - p5) != 0 else 0.0
            else:
                t_norm = 0.0
            time_norm.append(t_norm)

            # Normalize size values if available, else pad with 0.
            if i < len(size_list):
                s_val = size_list[i]
                p5, p95 = self.size_norm_params[i]
                s_norm = (s_val - p5) / (p95 - p5) if (p95 - p5) != 0 else 0.0
            else:
                s_norm = 0.0
            size_norm.append(s_norm)
        
        # Combine the normalized lists into a 2xN numpy array.
        sample = np.stack([time_norm, size_norm], axis=0)  # shape (2, max_len)
        # Convert prompt to label.
        label = self.prompt2label[row['prompt']]
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# Create the training dataset.
train_dataset = TextDataset(df, time_norm_params, size_norm_params, max_len)
num_classes = len(train_dataset.prompt2label)
print("Number of classes:", num_classes)

#################################
# 3. Define the CNN Model       #
#################################

class CNNClassifier(nn.Module):
    def __init__(self, kernel_width, num_classes, max_len):
        """
        The model architecture:
         - Input: tensor of shape (batch, 2, max_len). Treated as a single-channel "image" of size 2 x N.
         - Convolution: a conv2d layer with kernel size (2, kernel_width) so that the kernel spans across
           both sequences (time and size).
         - Fully-connected layer for classification.
        """
        super(CNNClassifier, self).__init__()
        self.kernel_width = kernel_width
        # Input is unsqueezed to shape (batch, 1, 2, max_len)
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, kernel_width))
        # Output width after convolution: W_out = max_len - kernel_width + 1
        conv_out_width = max_len - kernel_width + 1
        self.fc = nn.Linear(16 * conv_out_width, num_classes)
    
    def forward(self, x):
        # x: (batch, 2, max_len) -> unsqueeze to (batch, 1, 2, max_len)
        x = x.unsqueeze(1)
        x = self.conv(x)  # shape: (batch, 16, 1, conv_out_width)
        x = torch.relu(x)
        x = x.squeeze(2)  # shape: (batch, 16, conv_out_width)
        x = x.view(x.size(0), -1)  # flatten to (batch, 16 * conv_out_width)
        logits = self.fc(x)
        return logits

#############################################
# 4. Training with Cross-Validation         #
#############################################

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def eval_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

# Hyperparameter grid: trying different kernel widths.
kernel_width_list = [3, 5, 7]
learning_rate = 0.001
num_epochs = 10  # Adjust the number of epochs as needed.
batch_size = 32
num_folds = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on device:", device)

# KFold Cross Validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=SEED)
hyperparam_results = {}

for kernel_width in kernel_width_list:
    fold_accuracies = []
    print(f"Evaluating kernel_width = {kernel_width}")
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(train_dataset)))):
        print(f" Fold {fold + 1}/{num_folds}")
        # Create subset datasets for this fold.
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        
        # Initialize the model for this fold
        model = CNNClassifier(kernel_width=kernel_width, num_classes=num_classes, max_len=max_len)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop for current fold
        for epoch in range(num_epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
            # Uncomment the following line to see per-epoch stats:
            # print(f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        
        # Evaluate on validation set after training
        _, val_acc = eval_epoch(model, val_loader, criterion, device)
        fold_accuracies.append(val_acc)
        print(f"   Fold {fold + 1} Val Acc: {val_acc:.4f}")
    avg_acc = np.mean(fold_accuracies)
    hyperparam_results[kernel_width] = avg_acc
    print(f"Avg Val Accuracy for kernel_width {kernel_width}: {avg_acc:.4f}\n")

# Select the kernel_width with the highest average validation accuracy.
best_kernel_width = max(hyperparam_results, key=hyperparam_results.get)
print("Best kernel_width selected:", best_kernel_width)

#############################################
# 5. Final Training on Full Training Dataset#
#############################################

# (Optionally, perform a final training on the entire training dataset using the best hyperparameter.)
final_model = CNNClassifier(kernel_width=best_kernel_width, num_classes=num_classes, max_len=max_len)
final_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(final_model.parameters(), lr=learning_rate)
final_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print("Starting final training on the full training dataset...")
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(final_model, final_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs}: Loss={train_loss:.4f}, Accuracy={train_acc:.4f}")

# Save the final trained model.
torch.save(final_model.state_dict(), "cnn_classifier_final.pth")
print("Final model saved as cnn_classifier_final.pth")

##########################################
# 6. Load the Test Set and Run Inference #
##########################################

# Load the test set CSV.
df_test = pd.read_csv('test_set.csv')

# Convert string representations of lists to actual lists.
df_test['time'] = df_test['time'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df_test['size'] = df_test['size'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Note: In practice, one should use the normalization parameters computed from the training set.
# Here we reuse time_norm_params, size_norm_params, and max_len computed earlier.
# Also, we pass the same prompt2label mapping from the training dataset to ensure consistency.
test_dataset = TextDataset(df_test, time_norm_params, size_norm_params, max_len, prompt2label=train_dataset.prompt2label)

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create reverse mapping for labels.
label2prompt = {v: k for k, v in train_dataset.prompt2label.items()}

final_model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        outputs = final_model(X_batch)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Calculate test accuracy:
from sklearn.metrics import accuracy_score
test_accuracy = accuracy_score(all_labels, all_preds)
print("Final Test Accuracy: {:.4f}".format(test_accuracy))

# Convert numeric predictions to corresponding prompt text.
predicted_prompts = [label2prompt[p] for p in all_preds]

# Add the predictions as a new column to the test dataframe.
df_test['predicted_prompt'] = predicted_prompts

# Save the results to CSV.
cols = df_test.columns.tolist()
cols.insert(0, cols.pop(cols.index('predicted_prompt')))
df_test = df_test[cols]
df_test.to_csv('test_results.csv', index=False)
print("Test results saved as test_results.csv")
