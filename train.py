import glob
import json
import os
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np

# Import local modules
from core.classifier import *
from core.visualization import (set_plot_style, plot_training_curves, plot_roc_curve, 
                              plot_precision_recall_curve, plot_confusion_matrix, 
                              plot_score_distribution, create_model_dashboard)

# Configure logging
def setup_logging():
    """Set up logging configuration."""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/model_training_{time.strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main training function."""
    # Setup
    setup_logging()
    set_plot_style()
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Set parameters
    SEED = 42
    set_seed(SEED)
    
    BATCH_SIZE = 32
    EPOCHS = 200
    PATIENCE = 10
    LEARNING_RATE = 0.00001
    KERNEL_WIDTH = 3  # Best kernel width from cross-validation
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load the data from the specified input folder
    input_folder = './training_set/*.seq'
    input_prompts = 'prompts.json'
    logging.info(f"Loading data from {input_folder} and {input_prompts}")
    files = glob.glob('./training_set/*.seq')
    data = []
    for i, file in enumerate(files):
        if i % 1000 == 0:
            logging.info(f"Loading file {i}/{len(files)}: {file}")
        with open(file, 'r', encoding='utf8') as f:
            data.append(json.load(f))
    df = pd.DataFrame(data)

    # Join to prompts.json to add target column
    logging.info("Loading prompts.json")
    with open('prompts.json', 'r', encoding='utf8') as f:
        prompts = json.load(f)
    df['target'] = df['prompt'].apply(lambda x: 1 if x in prompts['positive']['prompts'] else 0)

    # Split into train and test sets. Hold out 20% of unique 'prompts' values for test set.
    logging.info("Splitting into train and test sets")
    unique_prompts = df.drop_duplicates(subset=['prompt'])[['prompt', 'target']]
    train_prompts, test_prompts = train_test_split(
        unique_prompts['prompt'],
        test_size=0.2,
        random_state=SEED,
        stratify=unique_prompts['target']
    )
    train_prompts = set(train_prompts)
    test_prompts = set(test_prompts)
    df_train = df[df['prompt'].isin(train_prompts)]
    df_test = df[df['prompt'].isin(test_prompts)]
    
    # Prepare data
    logging.info("Preparing data")
    df_train, max_len = prepare_data(df_train)
    df_test, _ = prepare_data(df_test)
    
    # Calculate normalization parameters
    time_norm_params, size_norm_params = calculate_norm_params(df_train, max_len)
    
    # Normalize dataframes
    df_train_normalized = normalize_dataframe(df_train, time_norm_params, size_norm_params, max_len)
    df_test_normalized = normalize_dataframe(df_test, time_norm_params, size_norm_params, max_len)
    
    # Save normalization parameters for inference
    save_normalization_params(time_norm_params, size_norm_params, max_len, 'models/normalization_params.npz')
    
    # Create datasets
    train_dataset = PreprocessedTextDataset(df_train_normalized, max_len)
    test_dataset = PreprocessedTextDataset(df_test_normalized, max_len)
    logging.info("Datasets created.")
    
    # Split training data for validation
    train_indices, val_indices = train_test_split(
        range(len(train_dataset)), test_size=0.2, random_state=SEED)
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = CNNBinaryClassifier(KERNEL_WIDTH, max_len).to(device)
    logging.info(f"Model created: {model}")
    
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=PATIENCE, 
        verbose=True, 
        path='models/checkpoint.pt'
    )
    
    # Training loop
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_epoch = 0
    
    for epoch in range(EPOCHS):
        logging.info(f"Epoch {epoch+1}/{EPOCHS}")
        
        # Train and validate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        logging.info(f'Epoch {epoch+1}/{EPOCHS}, '
                    f'Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, '
                    f'Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}')
        
        # Early stopping check
        early_stopping(val_acc, model)
        if early_stopping.early_stop:
            logging.info(f'Early stopping triggered at epoch {epoch+1}')
            best_epoch = epoch + 1 - PATIENCE
            break
        best_epoch = epoch + 1
    
    logging.info(f"Best model found at epoch {best_epoch}")
    
    # Load the best model
    model.load_state_dict(torch.load('models/checkpoint.pt'))
    
    # Save the final model
    save_model(
        model, 
        'models/cnn_binary_classifier.pth', 
        normalization_params=(time_norm_params, size_norm_params, max_len)
    )
    
    # Get predictions on validation set for evaluation
    val_scores, val_labels = get_prediction_scores(model, val_loader, device)
    val_preds = (val_scores > 0.5).astype(int)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, best_epoch, 
                        'results/training_curves.png')
    
    # Plot ROC curve
    plot_roc_curve(val_labels, val_scores, 'results/roc_curve.png')
    
    # Plot Precision-Recall curve
    plot_precision_recall_curve(val_labels, val_scores, 'results/precision_recall_curve.png')
    
    # Plot confusion matrix
    conf_matrix = plot_confusion_matrix(val_labels, val_preds, 'results/confusion_matrix.png')
    
    # Generate model dashboard
    create_model_dashboard(val_scores, val_labels, train_accs, val_accs, best_epoch, 
                          'results/model_performance_dashboard.png')
    
    # Run on test set and save predictions
    logging.info("Running inference on test dataset...")
    test_scores, _ = get_prediction_scores(model, test_loader, device)
    test_preds = (test_scores > 0.5).astype(int)
    
    # Plot prediction score distribution
    plot_score_distribution(test_scores, 'results/prediction_score_distribution.png')
    
    # Save test predictions
    df_test_normalized['prediction'] = test_preds
    df_test_normalized['score'] = test_scores
    df_test_normalized.to_csv('results/test_results.csv', index=False)
    logging.info("Results saved to results/test_results.csv")
    
    # Print confusion matrix metrics
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = 2 * precision_val * sensitivity / (precision_val + sensitivity) if (precision_val + sensitivity) > 0 else 0
    
    logging.info("\nConfusion Matrix:")
    logging.info(f"              Predicted Negative  Predicted Positive")
    logging.info(f"Actual Negative      {tn:<18} {fp}")
    logging.info(f"Actual Positive      {fn:<18} {tp}")
    logging.info("\nMetrics from Confusion Matrix:")
    logging.info(f"Sensitivity/Recall: {sensitivity:.4f}")
    logging.info(f"Specificity:        {specificity:.4f}")
    logging.info(f"Precision:          {precision_val:.4f}")
    logging.info(f"Negative Pred Value: {npv:.4f}")
    logging.info(f"Accuracy:           {accuracy:.4f}")
    logging.info(f"F1 Score:           {f1:.4f}")
    
    logging.info("Training complete!")

if __name__ == "__main__":
    main()