import time
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from core.utils import PrintUtils # Assuming this import works in your environment
import numpy as np
from sklearn.metrics import accuracy_score
import random
import sys
from collections import defaultdict

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
        self.val_loss_min = 0
        self.delta = delta
        self.path = path
        
    def __call__(self, val_loss, model):
        """
            Call override.
        """

        # Performs the early stopping
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score > self.best_score + self.delta:
            self.counter += 1
            PrintUtils.print_extra(f'EarlyStopping counter: *{self.counter}* out of *{self.patience}*')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        """
            Saves model when validation accuracy improves.
        """
        
        # Save data
        if self.verbose:
            PrintUtils.print_extra(f'Validation loss improved (*{self.val_loss_min:.6f}* --> *{val_loss:.6f}*). Saving model.')
        torch.save(model.state_dict(), self.path)

        # Override accuracy value 
        self.val_loss_min = val_loss


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

import time
# import numpy as np # Assuming numpy is available if needed for other parts
# from utils import PrintUtils # Assuming PrintUtils is available if needed for other parts
import sys # Needed for sys.exit()
from collections import defaultdict # Useful for storing timing data

# Define NUM_ITERATIONS_TO_TIME constant
NUM_ITERATIONS_TO_TIME = 20


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, max_epochs):
    """
        Train the model for one epoch.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    seconds_per_steps = []

    # Determine if the criterion expects logits (common case now)
    criterion_expects_logits = isinstance(criterion, torch.nn.BCEWithLogitsLoss)

    for i, (X, y) in enumerate(dataloader): # Use enumerate for progress tracking
        start_time_epoch = time.time()
        X, y = X.to(device), y.to(device).float().unsqueeze(1) # Ensure y is float and correct shape

        optimizer.zero_grad()
        output = model(X) # Output should be logits if using BCEWithLogitsLoss

        loss = criterion(output, y) # Calculate loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

        # Calculate predictions based on whether output is logits or probabilities
        with torch.no_grad():
            if criterion_expects_logits:
                # If criterion expects logits, output is logits. Apply sigmoid for prediction.
                pred = (torch.sigmoid(output) > 0.5).float()
            else:
                # If criterion expects probabilities (e.g., BCELoss), output is already probabilities.
                pred = (output > 0.5).float()

        correct += (pred == y).sum().item()
        total += y.size(0)

        # Update progress bar
        seconds_per_step = (time.time() - start_time_epoch)
        seconds_per_steps.append(seconds_per_step)
        # Use a moving average window (e.g., last 10 steps) for smoother time estimate
        if len(seconds_per_steps) > 10: 
             seconds_per_steps.pop(0) 
        avg_seconds_per_step = np.mean(seconds_per_steps) if seconds_per_steps else 0
        
        current_batch_loss = loss.item()
        current_batch_accuracy = (pred == y).sum().item() / y.size(0) if y.size(0) > 0 else 0
        
        progress = (i + 1) / len(dataloader)
        PrintUtils.start_stage(
            f'Training (epoch {epoch+1}/{max_epochs}): {progress*100:.1f}% '
            f'(s/iter={avg_seconds_per_step:.3f}, loss={current_batch_loss:.4f}, acc={current_batch_accuracy:.3f})', 
            override_prev=True
        )

    # Return epoch loss and accuracy
    epoch_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    return epoch_loss, epoch_acc


def eval_epoch(model, dataloader, criterion, device):
    """
        Evaluate the model on the validation set.
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    # Determine if the criterion expects logits
    criterion_expects_logits = isinstance(criterion, torch.nn.BCEWithLogitsLoss)

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device).float().unsqueeze(1) # Ensure y is float and correct shape
            output = model(X) # Output should be logits if using BCEWithLogitsLoss

            loss = criterion(output, y) # Calculate loss
            total_loss += loss.item() * X.size(0)

            # Calculate predictions based on whether output is logits or probabilities
            if criterion_expects_logits:
                pred = (torch.sigmoid(output) > 0.5).float()
            else:
                pred = (output > 0.5).float()

            all_preds.extend(pred.cpu().numpy().flatten())
            all_labels.extend(y.cpu().numpy().flatten())

    epoch_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0
    # Calculate accuracy using all collected predictions and labels
    accuracy = accuracy_score(all_labels, all_preds) if all_labels else 0
    return epoch_loss, accuracy


def get_prediction_scores(model, dataloader, device, return_probs=True):
    """
        Get raw prediction scores (logits or probabilities) and true labels from dataloader.

        Args:
            model: The trained model.
            dataloader: DataLoader for the dataset.
            device: The device to run inference on.
            return_probs (bool): If True, applies sigmoid to model output assuming 
                                 output are logits, returning probabilities. 
                                 If False, returns raw model output (logits).
                                 Set to False only if downstream code explicitly handles logits.

        Returns:
            tuple (np.array, np.array): A tuple containing scores and true labels.
    """
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            outputs = model(X) # Raw model output (likely logits)

            if return_probs:
                # Assume outputs are logits if return_probs is True, apply sigmoid
                scores = torch.sigmoid(outputs) 
            else:
                # Return raw logits
                scores = outputs 

            all_scores.extend(scores.cpu().numpy().flatten())
            all_labels.extend(y.numpy().flatten()) # y comes from dataloader, usually on CPU already

    return np.array(all_scores), np.array(all_labels)

def split_data(df, seed, test_size=0.2, valid_size=0.1):
    """
    Split data into train, validation, and test sets
    
    Args:
        df: DataFrame to split
        seed: Random seed for reproducibility
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Split into train and test sets preserving prompt distribution
    unique_prompts = df.drop_duplicates(subset=['prompt'])[['prompt', 'target']]
    train_and_val_prompts, test_prompts = train_test_split(
        unique_prompts['prompt'],
        test_size=test_size,
        random_state=seed,  # Use trial-specific seed
        stratify=unique_prompts['target']
    )
    test_prompts = set(test_prompts)
    
    # Split train into train and validation
    df_train_val = df[df['prompt'].isin(train_and_val_prompts)]
    df_train, df_val = train_test_split(
        df_train_val,
        test_size=valid_size,
        random_state=seed,  # Use trial-specific seed
        stratify=df_train_val['target']
    )
    
    df_test = df[df['prompt'].isin(test_prompts)]
    
    return df_train, df_val, df_test

def oversample(df, neg_to_pos, random_state=None):
    """
    Samples the negative class (target=0) in a DataFrame to achieve a
    specified ratio of negative to positive samples.

    This function will either:
    1. Oversample the negative class (sample with replacement) if its
       current count is lower than required by the ratio.
    2. Undersample the negative class (sample without replacement) if its
       current count is higher than required by the ratio.
    3. Keep the negative class as is if the count already matches the requirement.

    The positive class (target=1) remains unchanged. The final DataFrame is shuffled.

    Args:
        df (pd.DataFrame): The input DataFrame. Must contain a 'target'
                           column with binary values (0 for negative,
                           1 for positive).
        neg_to_pos (float): The desired ratio of negative samples count to positive
                            samples count (e.g., 10.0 means 10 negative samples
                            for every 1 positive sample). Must be positive.
        random_state (int, optional): Controls the randomness of sampling and
                                      shuffling for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: A new DataFrame with the negative class sampled
                      (either over- or under-sampled) to meet the ratio
                      and the rows shuffled.

    Raises:
        ValueError: If 'target' column is missing, neg_to_pos is not positive,
                    or if oversampling is required but the negative class is empty.
    """
    # --- Input Validation ---
    if 'target' not in df.columns:
        raise ValueError("DataFrame must have a 'target' column.")
    if not isinstance(neg_to_pos, (int, float)) or neg_to_pos <= 0:
        raise ValueError("neg_to_pos must be a positive number.")
    if not df['target'].isin([0, 1]).all():
        print("Warning: 'target' column contains values other than 0 and 1. Assuming 0=negative, 1=positive.")

    # --- Separate classes ---
    pos_df = df[df['target'] == 1]
    neg_df = df[df['target'] == 0]

    n_pos = len(pos_df)
    n_neg = len(neg_df)

    # --- Handle edge case: No positive samples ---
    if n_pos == 0:
        print("Warning: No positive samples (target=1) found. Cannot apply ratio. Returning original data shuffled.")
        # If no positives, the concept of neg_to_pos ratio is undefined.
        # Return original shuffled.
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # --- Calculate target negative count ---
    # Use round to get the closest integer count for the target ratio.
    n_neg_desired = int(round(neg_to_pos * n_pos))

    # --- Perform Sampling (Over, Under, or No Change) ---
    if n_neg_desired > n_neg:
        # --- Oversample ---
        if n_neg == 0:
            # Cannot oversample if there are no negative samples to begin with
            raise ValueError("Cannot oversample the negative class (target=0) as it is initially empty and positives exist.")

        n_samples_to_add = n_neg_desired - n_neg
        # Sample *with replacement* from the existing negative samples
        neg_samples_added = neg_df.sample(n=n_samples_to_add, replace=True, random_state=random_state)
        # Combine original negatives with the new samples
        neg_sampled_df = pd.concat([neg_df, neg_samples_added], ignore_index=True)

    elif n_neg_desired < n_neg:
        # --- Undersample ---
        # Sample *without replacement* from the existing negative samples
        neg_sampled_df = neg_df.sample(n=n_neg_desired, replace=False, random_state=random_state)

    else:
        # --- No Change Needed ---
        neg_sampled_df = neg_df # Use original negative samples

    # --- Combine positive and sampled negative classes ---
    result_df = pd.concat([pos_df, neg_sampled_df], ignore_index=True)

    # --- Shuffle the final DataFrame ---
    shuffled_df = result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return shuffled_df