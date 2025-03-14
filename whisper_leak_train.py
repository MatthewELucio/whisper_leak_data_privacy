#!/usr/bin/env python3
from core.classifier import CNNBinaryClassifier
from core.classifier import LSTMBinaryClassifier
from core.classifier import prepare_data
from core.classifier import calculate_norm_params
from core.classifier import normalize_dataframe
from core.classifier import EarlyStopping
from core.classifier import set_seed
from core.classifier import train_epoch
from core.classifier import eval_epoch
from core.classifier import get_prediction_scores
from core.classifier import PreprocessedTextDataset

from core.visualization import set_plot_style
from core.visualization import plot_training_curves
from core.visualization import plot_roc_curve
from core.visualization import plot_precision_recall_curve
from core.visualization import plot_confusion_matrix 
from core.visualization import plot_score_distribution
from core.visualization import create_model_dashboard

from core.utils import ThrowingArgparse
from core.utils import PrintUtils
from core.utils import OsUtils
from core.chatbot_utils import ChatbotUtils

import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def get_self_dir():
    """
        Get the self directory.
    """
 
    # Return the self directory
    return os.path.dirname(os.path.abspath(__file__))
 
def parse_arguments():
    """
        Parse arguments.
    """

    # Load all chatbots
    PrintUtils.start_stage('Loading chatbots')
    chatbots = ChatbotUtils.load_chatbots(os.path.join(get_self_dir(), 'chatbots'))
    assert len(chatbots) > 0, Exception('Could not load any chatbots')
    chatbot_names = ', '.join([ f'*{name}*' for name in chatbots.keys() ])
    PrintUtils.print_extra(f'Loaded chatbots: {chatbot_names}')
    PrintUtils.end_stage()

    # Parsing arguments
    PrintUtils.start_stage('Parsing command-line arguments')
    parser = ThrowingArgparse()
    parser.add_argument('-c', '--chatbot', help='The chatbot.', required=True)
    parser.add_argument('-m', '--modeltype', help='The model type (CNN or LSTM).', default='CNN')
    parser.add_argument('-p', '--prompts', help='The prompts JSON file path', default='prompts.json')
    parser.add_argument('-s', '--seed', type=int, help='The random seed', default=42)
    parser.add_argument('-b', '--batchsize', type=int, help='The batch size', default=32)
    parser.add_argument('-e', '--epochs', type=int, help='The number of epochs', default=200)
    parser.add_argument('-P', '--patience', type=int, help='The patience value', default=5)
    parser.add_argument('-k', '--kernelwidth', type=int, help='The kernel width', default=3)
    parser.add_argument('-l', '--learningrate', type=float, help='The learning rate', default=0.0001)
    parser.add_argument('-t', '--testsize', type=int, help='The test size in percentage', default=20)
    args = parser.parse_args()
    assert args.seed >= 0, Exception(f'Invalid random seed: {args.seed}')
    assert args.batchsize > 0, Exception(f'Invalid batch size: {args.batchsize}')
    assert args.epochs > 0, Exception(f'Invalid number of epochs: {args.epochs}')
    assert args.patience > 0, Exception(f'Invalid patience value: {args.patience}')
    assert args.kernelwidth > 0, Exception(f'Invalid kernel width: {args.kernelwidth}')
    assert args.learningrate > 0 and args.learningrate < 1, Exception(f'Invalid learning rate: {args.learningrate}')
    assert args.testsize > 0 and args.testsize < 100, Exception(f'Invalid test size percentage: {args.testsize}')
    assert len(args.chatbot) > 0, Exception('Chatbot name cannot be empty')
    PrintUtils.end_stage()

    # Return the parsed arguments
    return args

def main():
    """
        Main routine.
    """

    # Catch-all
    is_user_cancelled = False
    last_error = None
    try:

        # Suppress STDERR
        OsUtils.suppress_stderr()

        # Print logo
        PrintUtils.print_logo()

        # Parsing arguments
        args = parse_arguments()

        # Setup
        set_plot_style()
        set_seed(args.seed)
        
        # Create directories
        PrintUtils.start_stage('Making directories')
        models_dir = os.path.join(get_self_dir(), 'models')
        assert OsUtils.mkdir(models_dir), Exception(f'Could not get or make directory "{models_dir}"')
        results_dir = os.path.join(get_self_dir(), 'results')
        assert OsUtils.mkdir(results_dir), Exception(f'Could not get or make directory "{results_dir}"')
        PrintUtils.end_stage()
   
        # Either use GPU or CPU
        PrintUtils.start_stage('Setting device')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        PrintUtils.end_stage()
        PrintUtils.print_extra(f'Using device: *{device}*')

        # Load the data from the specified input folder
        PrintUtils.start_stage('Loading sequences data')
        training_set_dir = os.path.join(get_self_dir(), 'training_set')
        files = [ os.path.join(training_set_dir, i) for i in os.listdir(training_set_dir) if i.lower().endswith(f'_{args.chatbot.lower()}.seq') ]
        assert len(files) > 0, Exception(f'Did not find training set files for chatbot {args.chatbot}')
        data = []
        file_index = 0
        for file_index in range(len(files)):
            with open(files[file_index], 'r') as fp:
                data.append(json.load(fp))
            percentage = (file_index * 100) // len(files)
            if file_index % 10 == 0:
                PrintUtils.start_stage(f'Loading sequences data ({file_index} / {len(files)} = {percentage}%)', override_prev=True)
        df = pd.DataFrame(data)
        PrintUtils.end_stage()

        # Join to prompts to add target column
        PrintUtils.start_stage('Loading prompts')
        with open(args.prompts, 'r') as f:
            prompts = json.load(f)
        df['target'] = df['prompt'].apply(lambda x: 1 if x in prompts['positive']['prompts'] else 0)
        PrintUtils.end_stage()
        total_prompts = len(prompts['positive']['prompts']) + len(prompts['negative']['prompts'])
        PrintUtils.print_extra(f'Loaded {total_prompts} prompts')

        # Split into train and test sets and hold out a percentage of unique 'prompts' values for test set
        PrintUtils.start_stage('Splitting into train and test sets')
        unique_prompts = df.drop_duplicates(subset=['prompt'])[['prompt', 'target']]
        train_prompts, test_prompts = train_test_split(
            unique_prompts['prompt'],
            test_size=args.testsize / 100,
            random_state=args.seed,
            stratify=unique_prompts['target']
        )
        train_prompts = set(train_prompts)
        test_prompts = set(test_prompts)
        df_train = df[df['prompt'].isin(train_prompts)]
        df_test = df[df['prompt'].isin(test_prompts)]
        PrintUtils.end_stage()

        # Prepare data
        PrintUtils.start_stage('Preparing data')
        df_train, max_len = prepare_data(df_train)
        df_test, _ = prepare_data(df_test)
        PrintUtils.end_stage()
        PrintUtils.print_extra(f'Max sequence length being used for model (90th percentile): *{max_len}*')
        
        # Calculate normalization parameters
        time_norm_params, size_norm_params = calculate_norm_params(df_train, max_len)
        
        # Normalize dataframes
        df_train_normalized = normalize_dataframe(df_train, time_norm_params, size_norm_params, max_len)
        df_test_normalized = normalize_dataframe(df_test, time_norm_params, size_norm_params, max_len)
        
        # Save normalization parameters for inference
        normalization_params = (time_norm_params, size_norm_params, max_len)
        CNNBinaryClassifier.save_normalization_params(
            time_norm_params, size_norm_params, max_len, os.path.join(models_dir, 'normalization_params.npz')
        )
        
        # Create datasets and loaders for training
        PrintUtils.start_stage('Creating datasets and loaders')
        train_dataset = PreprocessedTextDataset(df_train_normalized, max_len)
        test_dataset = PreprocessedTextDataset(df_test_normalized, max_len)

        # Split training data for validation
        train_indices, val_indices = train_test_split(
            range(len(train_dataset)), test_size=args.testsize / 100, random_state=args.seed)
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        
        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=args.batchsize, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=args.batchsize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)
        PrintUtils.end_stage()
    
        # Choose model architecture (CNN or LSTM)
        PrintUtils.start_stage('Instantiating model')
        model_type = args.modeltype.upper()
        if model_type == 'CNN':
            model = CNNBinaryClassifier(args.kernelwidth, max_len).to(device)
            model_path = os.path.join(models_dir, 'cnn_binary_classifier.pth')
        elif model_type == 'LSTM':
            model = LSTMBinaryClassifier(args.kernelwidth, max_len).to(device)
            model_path = os.path.join(models_dir, 'lstm_binary_classifier.pth')
        else:
            raise Exception(f'Unsupported model type: {args.modeltype}')
        PrintUtils.print_extra(f'Model created: *{model.__class__.__name__}*')
    
        # Define loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learningrate)
        
        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=args.patience, 
            verbose=True, 
            path=os.path.join(models_dir, 'checkpoint.pt')
        )
        PrintUtils.end_stage()
        
        # Training loop
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        best_epoch = 0
        PrintUtils.start_stage('Training model')
        for epoch in range(args.epochs):

            # Log
            PrintUtils.start_stage(f'Training (epoch {epoch+1} / {args.epochs})', override_prev=True)
            
            # Train and validate
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, args.epochs)
            val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            PrintUtils.start_stage(f'Training (epoch {epoch+1} / {args.epochs}): train loss: {train_loss:.4f}, train acc: {train_acc:.4f}, val loss: {val_loss:.4f}, val acc: {val_acc:.4f}', override_prev=True)
            PrintUtils.end_stage()
            
            # Early stopping check
            early_stopping(val_acc, model)
            if early_stopping.early_stop:
                PrintUtils.print_extra(f'Training (epoch {epoch+1} / {args.epochs}): early stopping triggered')
                best_epoch = epoch + 1 - args.patience
                break
            best_epoch = epoch + 1
            
        
        PrintUtils.print_extra(f'Best model found at epoch *{best_epoch}*')
        PrintUtils.start_stage('Saving model')
        
        # Load the best model
        model.load_state_dict(torch.load(os.path.join(models_dir, 'checkpoint.pt')))
        
        # Save the final model
        model.save(model_path, normalization_params=normalization_params)
        
        # Get predictions on test set for evaluation
        PrintUtils.end_stage()
        PrintUtils.start_stage('Inferencing on test dataset and generating metrics')
        test_scores, test_labels = get_prediction_scores(model, test_loader, device)
        test_preds = (test_scores > 0.5).astype(int)
        
        # Plot training curves
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, best_epoch, os.path.join(results_dir, 'training_curves.png'))
        
        # Plot ROC curve
        plot_roc_curve(test_labels, test_scores, os.path.join(results_dir, 'roc_curve.png'))
        
        # Plot Precision-Recall curve
        plot_precision_recall_curve(test_labels, test_scores, os.path.join(results_dir, 'precision_recall_curve.png'))
        
        # Plot confusion matrix
        conf_matrix = plot_confusion_matrix(test_labels, test_preds, os.path.join(results_dir, 'confusion_matrix.png'))
        
        # Generate model dashboard
        create_model_dashboard(test_scores, test_labels, train_accs, val_accs, best_epoch,  os.path.join(results_dir, 'model_performance_dashboard.png'))
        
        # Plot prediction score distribution
        plot_score_distribution(test_scores, os.path.join(results_dir, 'prediction_score_distribution.png'))
        
        # Save test predictions
        df_test_normalized['prediction'] = test_preds
        df_test_normalized['score'] = test_scores
        df_test_normalized.to_csv(os.path.join(results_dir, 'test_results.csv'), index=False)
        PrintUtils.end_stage()
        PrintUtils.print_extra('Results saved to *test_results.csv*')
        
        # Print confusion matrix metrics
        PrintUtils.start_stage('Printing confusion matrix metrics')
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * precision_val * sensitivity / (precision_val + sensitivity) if (precision_val + sensitivity) > 0 else 0
        PrintUtils.end_stage()
        PrintUtils.print_extra(f'              Predicted Negative  Predicted Positive')
        PrintUtils.print_extra(f'Actual Negative      {tn:<18} {fp}')
        PrintUtils.print_extra(f'Actual Positive      {fn:<18} {tp}')
        PrintUtils.print_extra('Metrics from Confusion Matrix:')
        PrintUtils.print_extra(f'Sensitivity/Recall: {sensitivity:.4f}')
        PrintUtils.print_extra(f'Specificity:        {specificity:.4f}')
        PrintUtils.print_extra(f'Precision:          {precision_val:.4f}')
        PrintUtils.print_extra(f'Negative Pred Value: {npv:.4f}')
        PrintUtils.print_extra(f'Accuracy:           {accuracy:.4f}')
        PrintUtils.print_extra(f'F1 Score:           {f1:.4f}')
        # Also write to output file
        with open(os.path.join(results_dir, 'confusion_matrix_metrics.txt'), 'w') as f:
            f.write(f'Confusion Matrix Metrics:\n')
            f.write(f'Sensitivity/Recall: {sensitivity:.4f}\n')
            f.write(f'Specificity:        {specificity:.4f}\n')
            f.write(f'Precision:          {precision_val:.4f}\n')
            f.write(f'Negative Pred Value: {npv:.4f}\n')
            f.write(f'Accuracy:           {accuracy:.4f}\n')
            f.write(f'F1 Score:           {f1:.4f}\n')
        PrintUtils.print_extra(f'Confusion matrix metrics saved to *confusion_matrix_metrics.txt*')
        
        # Test the inference function
        PrintUtils.start_stage('Testing inference function')
        sample_row = df_test.iloc[0]
        time_diffs, data_lengths = sample_row['time_diffs'], sample_row['data_lengths']
        
        # Test with tuple input
        prob, pred = model.inference(
            (time_diffs, data_lengths), 
            device, 
            normalization_params=normalization_params
        )
        PrintUtils.end_stage()
        PrintUtils.print_extra(f'Inference result (tuple input): prob=*{prob:.4f}*, pred=*{pred}*')
        
        # Test with DataFrame input
        sample_df = df_test.iloc[[0]]
        probs, preds = model.inference(
            sample_df,
            device,
            normalization_params=normalization_params
        )
        PrintUtils.print_extra(f'Inference result (DataFrame input): prob=*{probs[0]:.4f}*, pred=*{preds[0]}*')

    # Handle exceptions
    except Exception as ex:

        # Optionally fail stage
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message=ex, throw_on_fail=False)

        # Save error and print it as an extra
        PrintUtils.print_extra(f'Error: {ex}')
        last_error = ex

    # Handle cancel operations
    except KeyboardInterrupt:
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message='', throw_on_fail=False)
        PrintUtils.print_extra(f'Operation *cancelled* by user - please wait for cleanup code to complete')
        is_user_cancelled = True

    # Cleanups
    finally:

        # Print final status
        if last_error is not None:
            PrintUtils.print_error(last_error)
        elif is_user_cancelled:
            PrintUtils.print_extra(f'Operation *cancelled* by user')
        else:
            PrintUtils.print_extra(f'Finished successfully')

if __name__ == '__main__':
    main()
