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


def get_prediction_scores(model, dataloader, device, criterion=None, return_probs=True, neg_to_pos_ratio=None):
    """
        Get raw prediction scores (logits or probabilities), true labels and losses from dataloader.

        Args:
            model: The trained model.
            dataloader: DataLoader for the dataset.
            device: The device to run inference on.
            criterion: Loss function to calculate per-batch loss. If None, losses won't be computed.
            return_probs (bool): If True, applies sigmoid to model output assuming 
                                 output are logits, returning probabilities. 
                                 If False, returns raw model output (logits).
                                 Set to False only if downstream code explicitly handles logits.
            neg_to_pos_ratio: Ratio of negative to positive samples (for imbalanced datasets).

        Returns:
            tuple (np.array, np.array, float): A tuple containing scores, true labels, and total loss (if criterion provided).
    """
    model.eval()
    all_scores = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y_device = y.to(device).float().unsqueeze(1) if criterion else y  # Only move to device if needed
            outputs = model(X)  # Raw model output (likely logits)

            # Calculate loss if criterion is provided
            if criterion:
                loss = criterion(outputs, y_device)
                total_loss += loss.item() * X.size(0)
            
            if return_probs:
                # Assume outputs are logits if return_probs is True, apply sigmoid
                scores = torch.sigmoid(outputs) 
            else:
                # Return raw logits
                scores = outputs 

            all_scores.extend(scores.cpu().numpy().flatten())
            all_labels.extend(y.numpy().flatten())  # y comes from dataloader, usually on CPU already
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    # If neg_to_pos_ratio is provided, adjust the arrays
    if neg_to_pos_ratio is not None:
        # Get masks for positive and negative samples
        pos_mask = all_labels == 1
        neg_mask = all_labels == 0
        
        # Count current positives and negatives
        n_pos = np.sum(pos_mask)
        n_neg = np.sum(neg_mask)
        
        if n_pos > 0:  # Only proceed if we have positive samples
            # Calculate target number of negatives
            n_neg_desired = int(round(neg_to_pos_ratio * n_pos))
            
            if n_neg_desired > n_neg:  # Only oversample if we need more negatives
                # Calculate how many times to duplicate the negatives
                n_samples_to_add = n_neg_desired - n_neg
                
                if n_neg > 0:  # Only proceed if we have some negatives to duplicate
                    # Get indices of negative samples
                    neg_indices = np.where(neg_mask)[0]
                    
                    # Randomly select indices to duplicate (with replacement if needed)
                    indices_to_duplicate = np.random.choice(neg_indices, size=n_samples_to_add, replace=True)
                    
                    # Extract the scores and labels for the selected indices
                    additional_scores = all_scores[indices_to_duplicate]
                    additional_labels = all_labels[indices_to_duplicate]  # Will all be 0
                    
                    # Concatenate with original arrays
                    all_scores = np.concatenate([all_scores, additional_scores])
                    all_labels = np.concatenate([all_labels, additional_labels])

        # Create a permutation index array
        perm = np.random.permutation(len(all_scores))
        # Apply the same permutation to both arrays
        all_scores = all_scores[perm]
        all_labels = all_labels[perm]

    epoch_loss = total_loss / len(dataloader.dataset) if criterion and len(dataloader.dataset) > 0 else 0
    return all_scores, all_labels, epoch_loss


def eval_epoch(model, dataloader, criterion, device, neg_to_pos_ratio=None):
    """
        Evaluate the model on the validation set.
    """
    model.eval()
    
    # Determine if the criterion expects logits
    criterion_expects_logits = isinstance(criterion, torch.nn.BCEWithLogitsLoss)
    
    # Use get_prediction_scores to get predictions and loss
    scores, labels, epoch_loss = get_prediction_scores(
        model, 
        dataloader, 
        device, 
        criterion=criterion,
        return_probs=criterion_expects_logits,  # If using BCEWithLogitsLoss, we want probabilities
        neg_to_pos_ratio=neg_to_pos_ratio
    )
    
    # Convert scores to binary predictions
    predictions = (scores > 0.5).astype(float)
    
    # Calculate accuracy using all collected predictions and labels
    accuracy = accuracy_score(labels, predictions) if len(labels) > 0 else 0
    
    return epoch_loss, accuracy


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

def calculate_sampling_details(df, neg_to_pos):
    """
    Calculates the number of positive and negative samples, the desired
    number of negative samples based on the ratio, and the current ratio.

    Args:
        df (pd.DataFrame): Input DataFrame with a 'target' column.
        neg_to_pos (float): The desired ratio of negative to positive samples.

    Returns:
        tuple: Contains:
            - n_pos (int): Count of positive samples (target=1).
            - n_neg (int): Count of negative samples (target=0).
            - n_neg_desired (int): Target count for negative samples based on ratio.
                                    Returns current n_neg if n_pos is 0.
            - current_ratio (float): Current neg/pos ratio (n_neg / n_pos).
                                     Returns np.inf if n_pos=0 and n_neg>0,
                                     np.nan if n_pos=0 and n_neg=0.
    Raises:
        ValueError: If 'target' column is missing or neg_to_pos is not positive.
    """
    if 'target' not in df.columns:
        raise ValueError("DataFrame must have a 'target' column.")
    if not isinstance(neg_to_pos, (int, float)) or neg_to_pos <= 0:
        raise ValueError("neg_to_pos must be a positive number.")
    if not df['target'].isin([0, 1]).all():
         print("Warning: 'target' column contains values other than 0 and 1. Assuming 0=negative, 1=positive.")

    n_pos = df['target'].eq(1).sum()
    n_neg = df['target'].eq(0).sum()

    if n_pos == 0:
        n_neg_desired = n_neg # Cannot determine target based on ratio
        current_ratio = np.inf if n_neg > 0 else np.nan
    else:
        # Use round to get the closest integer count for the target ratio.
        n_neg_desired = int(round(neg_to_pos * n_pos))
        current_ratio = n_neg / n_pos

    return n_pos, n_neg, n_neg_desired, current_ratio


def apply_neg_sampling(df, neg_to_pos, random_state=None):
    """
    Samples the negative class (target=0) in a DataFrame to achieve a
    specified ratio of negative to positive samples.

    Uses `calculate_sampling_details` to determine the target negative count.
    It will then either:
    1. Oversample the negative class (sample with replacement).
    2. Undersample the negative class (sample without replacement).
    3. Keep the negative class as is.

    The positive class (target=1) remains unchanged. The final DataFrame is shuffled.

    Args:
        df (pd.DataFrame): The input DataFrame. Must contain a 'target' column.
        neg_to_pos (float): The desired ratio of negative to positive samples.
        random_state (int, optional): Controls randomness for reproducibility. Defaults to None.

    Returns:
        pd.DataFrame: A new DataFrame with the negative class sampled to meet
                      the ratio and the rows shuffled.

    Raises:
        ValueError: Inherited from `calculate_sampling_details` or if oversampling
                    is required but the negative class is initially empty.
    """
    # --- Calculate counts and target ---
    # Input validation for df['target'] and neg_to_pos happens inside here
    n_pos, n_neg, n_neg_desired, current_ratio = calculate_sampling_details(df, neg_to_pos)

    # --- Handle edge case: No positive samples ---
    if n_pos == 0:
        print(f"Warning: No positive samples found (N={n_pos}). Cannot apply ratio {neg_to_pos}. Returning original data shuffled.")
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"Initial counts: Pos={n_pos}, Neg={n_neg}. Current Ratio={current_ratio:.2f}. Target Ratio={neg_to_pos:.2f} -> Target Neg Count={n_neg_desired}")

    # --- Separate classes (needed for sampling) ---
    pos_df = df[df['target'] == 1]
    neg_df = df[df['target'] == 0]

    # --- Perform Sampling (Over, Under, or No Change) ---
    if n_neg_desired > n_neg:
        # --- Oversample ---
        if n_neg == 0:
            raise ValueError(f"Cannot oversample negative class to {n_neg_desired} as it is initially empty (n_neg=0).")

        n_samples_to_add = n_neg_desired - n_neg
        neg_samples_added = neg_df.sample(n=n_samples_to_add, replace=True, random_state=random_state)
        neg_sampled_df = pd.concat([neg_df, neg_samples_added], ignore_index=True)
        print(f"Oversampling negative class: Adding {n_samples_to_add} samples.")

    elif n_neg_desired < n_neg:
        # --- Undersample ---
        neg_sampled_df = neg_df.sample(n=n_neg_desired, replace=False, random_state=random_state)
        print(f"Undersampling negative class: Selecting {n_neg_desired} samples from {n_neg}.")

    else:
        # --- No Change Needed ---
        print(f"Negative class already has the desired count ({n_neg_desired}). No sampling needed.")
        neg_sampled_df = neg_df # Use original negative samples

    # --- Combine positive and sampled negative classes ---
    result_df = pd.concat([pos_df, neg_sampled_df], ignore_index=True)

    # --- Shuffle the final DataFrame ---
    shuffled_df = result_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # --- Verification (optional) ---
    final_n_pos = len(shuffled_df[shuffled_df['target'] == 1])
    final_n_neg = len(shuffled_df[shuffled_df['target'] == 0])
    final_ratio = final_n_neg / final_n_pos if final_n_pos > 0 else np.inf if final_n_neg > 0 else np.nan
    # print(f"Final counts: Pos={final_n_pos}, Neg={final_n_neg}. Final Ratio: {final_ratio:.2f}")


    return shuffled_df