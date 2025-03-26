import torch
from core.utils import PrintUtils

import ast
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
import random

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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, max_epochs):
    """
        Train the model for one epoch.
    """

    # Train model
    model.train()
    total_loss = 0
    correct = 0
    total = 0
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
        PrintUtils.start_stage(f'Training (epoch {epoch+1} / {max_epochs}): {100.0*total/len(dataloader.dataset):.2f}% (loss = {loss.item():.4f}, accuracy = {correct/total:.4f})', override_prev=True)
   
    # Return epoch loss and accuracy
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