import time
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