#!/usr/bin/env python3
"""
Whisper Leak Benchmark - A tool for benchmarking ML models against multiple chatbots
to detect prompt leakage.
"""
import yaml
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    f1_score, recall_score, precision_score, 
    accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from core.classifier import (
    CNNBinaryClassifier, LSTMBinaryClassifier, prepare_data,
    calculate_norm_params, normalize_dataframe, EarlyStopping,
    set_seed, train_epoch, eval_epoch, get_prediction_scores, 
    PreprocessedTextDataset
)

from core.visualization import (
    set_plot_style, plot_training_curves, plot_roc_curve,
    plot_precision_recall_curve, plot_confusion_matrix,
    plot_score_distribution, create_model_dashboard
)

from core.utils import ThrowingArgparse, PrintUtils, OsUtils
from core.chatbot_utils import ChatbotUtils

import json
import shutil
import time
from datetime import datetime


class BenchmarkConfig:
    """
    Configuration class for the benchmark script, loaded from YAML.
    """
    def __init__(self, config_path):
        """
        Initialize the configuration from a YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        PrintUtils.start_stage(f'Loading configuration from {config_path}')
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            # Load and validate required fields
            self.chatbots = self.config.get('chatbots', [])
            if not self.chatbots:
                raise ValueError("No chatbots specified in configuration")
            
            self.benchmark_name = self.config.get('benchmark_name', 'benchmark')
            model_params = self.config.get('model', {})
            if not model_params:
                raise ValueError("No model parameters specified in configuration")

            self.model_class = model_params.get('class', 'LSTMBinaryClassifier')
            self.model_params = model_params.get('params', {})

            # Training parameters
            training_params = self.config.get('training', {})
            self.batch_size = training_params.get('batch_size', 32)
            self.max_epochs = training_params.get('max_epochs', 200)
            self.learning_rate = training_params.get('learning_rate', 0.0001)
            self.patience = training_params.get('patience', 5)
            self.test_size = training_params.get('test_size', 20)
            self.valid_size = training_params.get('valid_size', 5)
            self.seed = training_params.get('seed', 42)
            
            PrintUtils.print_extra(f'Configuration loaded for model: *{self.model_class}*')
            PrintUtils.print_extra(f'Benchmark name: *{self.benchmark_name}*')
            PrintUtils.print_extra(f'Chatbots to benchmark: *{", ".join(self.chatbots)}*')
            
        except Exception as e:
            PrintUtils.end_stage(fail_message=f"Failed to load configuration: {str(e)}")
            raise
            
        PrintUtils.end_stage()


class BenchmarkRunner:
    """
    Main class for running the benchmark suite
    """
    def __init__(self, config_path, reprocess_only=False):
        """
        Initialize the benchmark runner
        
        Args:
            config_path: Path to the YAML configuration file
            reprocess_only: If True, only reprocess statistics without retraining
        """
        self.config = BenchmarkConfig(config_path)
        self.reprocess_only = reprocess_only
        self.results = []
        
        # Setup directories
        self.base_dir = self.get_self_dir()
        self.benchmark_dir = os.path.join(self.base_dir, 'benchmark')
        self.results_file = os.path.join(self.benchmark_dir, 'benchmark.csv')
        
        # Create necessary directories
        OsUtils.mkdir(self.benchmark_dir)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        PrintUtils.print_extra(f'Using device: *{self.device}*')
        
        # Set random seed for reproducibility
        set_seed(self.config.seed)
        
        # Setup visualization style
        set_plot_style()
        
        # Check for existing results
        self.existing_results = {}
        if os.path.exists(self.results_file):
            df = pd.read_csv(self.results_file)
            for _, row in df.iterrows():
                chatbot_key = (row['CommonName'], row['ChatBot'])
                self.existing_results[chatbot_key] = True
    
    def get_self_dir(self):
        """
        Get the directory where this script is located
        """
        return os.path.dirname(os.path.abspath(__file__))
    
    def should_process_chatbot(self, chatbot, common_name):
        """
        Determine if we should process this chatbot based on existing results
        
        Args:
            chatbot: Name of the chatbot to check
            common_name: Common name for the chatbot
            
        Returns:
            bool: True if chatbot should be processed, False otherwise
        """
        return (common_name, chatbot) not in self.existing_results
    
    def run_benchmark(self):
        """
        Run the full benchmark suite across all chatbots
        """
        PrintUtils.start_stage(f'Starting benchmark suite: {self.config.benchmark_name}')
        
        # Load prompts file
        prompts_path = os.path.join(self.get_self_dir(), 'prompts.json')
        with open(prompts_path, 'r') as f:
            prompts = json.load(f)
        
        # Start with existing results if available
        if os.path.exists(self.results_file):
            try:
                existing_df = pd.read_csv(self.results_file)
                self.results = existing_df.to_dict('records')
            except Exception as e:
                PrintUtils.print_extra(f'Failed to load existing results: {str(e)}')
        
        # Process each chatbot
        for chatbot in self.config.chatbots:
            common_name = self.config.chatbot_common_names[chatbot]
            common_dir = os.path.join(self.benchmark_dir, common_name)
            chatbot_dir = os.path.join(common_dir, chatbot)
            
            # Create common directory if it doesn't exist
            OsUtils.mkdir(common_dir)
            
            if not self.should_process_chatbot(chatbot, common_name) and not self.reprocess_only:
                PrintUtils.print_extra(f'Skipping *{chatbot}* (common name: {common_name}) as it already exists in results')
                continue
                
            PrintUtils.print_extra(f'Processing chatbot: *{chatbot}* (common name: {common_name})')
            OsUtils.mkdir(chatbot_dir)
            
            # If reprocess_only, just calculate statistics without training
            if self.reprocess_only:
                self.reprocess_chatbot_statistics(chatbot, common_name, chatbot_dir, prompts)
            else:
                self.process_chatbot(chatbot, common_name, chatbot_dir, prompts)
        
        # Save all results to CSV
        self.save_results()
        
        PrintUtils.end_stage()
    
    def process_chatbot(self, chatbot, common_name, output_dir, prompts):
        """
        Process a single chatbot - train model and evaluate performance
        
        Args:
            chatbot: Name of the chatbot to process
            common_name: Common name for the chatbot
            output_dir: Directory to save results
            prompts: Loaded prompts dictionary
        """
        PrintUtils.start_stage(f'Processing {chatbot} (common name: {common_name})')
        
        try:
            # Load dataset
            df = self.load_chatbot_data(chatbot, prompts)
            
            # Split data
            df_train, df_val, df_test = self.split_data(df)
            
            # Prepare data
            df_train, max_len = prepare_data(df_train)
            df_val, _ = prepare_data(df_val)
            df_test, _ = prepare_data(df_test)
            
            # Normalize data
            time_norm_params, size_norm_params = calculate_norm_params(df_train)
            df_train_normalized = normalize_dataframe(df_train, time_norm_params, size_norm_params)
            df_val_normalized = normalize_dataframe(df_val, time_norm_params, size_norm_params)
            df_test_normalized = normalize_dataframe(df_test, time_norm_params, size_norm_params)
            
            # Save normalization parameters
            normalization_params = (time_norm_params, size_norm_params, max_len)
            norm_file = os.path.join(output_dir, 'normalization_params.npz')
            if self.config.model_type.upper() == 'CNN':
                CNNBinaryClassifier.save_normalization_params(
                    time_norm_params, size_norm_params, max_len, norm_file
                )
            else:
                LSTMBinaryClassifier.save_normalization_params(
                    time_norm_params, size_norm_params, max_len, norm_file
                )
            
            # Create datasets and data loaders
            train_dataset = PreprocessedTextDataset(df_train_normalized, max_len)
            val_dataset = PreprocessedTextDataset(df_val_normalized, max_len)
            test_dataset = PreprocessedTextDataset(df_test_normalized, max_len)
            
            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            # Initialize model
            if self.config.model_type.upper() == 'CNN':
                model = CNNBinaryClassifier(self.config.kernel_width, max_len).to(self.device)
            elif self.config.model_type.upper() == 'LSTM':
                model = LSTMBinaryClassifier(
                    self.config.kernel_width, 
                    max_len,
                    hidden_size=self.config.hidden_size,
                    num_layers=self.config.num_layers,
                    dropout_rate=self.config.dropout_rate,
                    bidirectional=self.config.bidirectional
                ).to(self.device)
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            # Setup training
            criterion = nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
            early_stopping = EarlyStopping(
                patience=self.config.patience,
                verbose=True,
                path=os.path.join(output_dir, 'best_model.pt')
            )
            
            # Train model
            model_file = os.path.join(output_dir, f'{self.config.model_type.lower()}_model.pth')
            train_losses, val_losses = [], []
            train_accs, val_accs = [], []
            best_epoch = 0
            
            PrintUtils.start_stage(f'Training model for {chatbot}')
            for epoch in range(self.config.max_epochs):
                PrintUtils.start_stage(f'Training epoch {epoch+1}/{self.config.max_epochs}', override_prev=True)
                
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, 
                    self.device, epoch, self.config.max_epochs
                )
                val_loss, val_acc = eval_epoch(model, val_loader, criterion, self.device)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                
                PrintUtils.start_stage(
                    f'Epoch {epoch+1}/{self.config.max_epochs}: '
                    f'train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, '
                    f'val_loss={val_loss:.4f}, val_acc={val_acc:.4f}',
                    override_prev=True
                )
                
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    PrintUtils.print_extra(f'Early stopping triggered at epoch {epoch+1}')
                    best_epoch = epoch + 1 - self.config.patience
                    break
                best_epoch = epoch + 1
            
            PrintUtils.end_stage()
            PrintUtils.print_extra(f'Best model found at epoch *{best_epoch}*')
            
            # Load best model and save
            model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pt')))
            model.save(model_file, normalization_params=normalization_params)
            
            # Create visualizations
            PrintUtils.start_stage(f'Creating visualizations for {chatbot}')
            test_scores, test_labels = get_prediction_scores(model, test_loader, self.device)
            test_preds = (test_scores > 0.5).astype(int)
            
            # Generate all plots
            plot_training_curves(
                train_losses, val_losses, train_accs, val_accs, 
                best_epoch, os.path.join(output_dir, 'training_curves.png')
            )
            
            plot_roc_curve(
                test_labels, test_scores, 
                os.path.join(output_dir, 'roc_curve.png')
            )
            
            plot_precision_recall_curve(
                test_labels, test_scores, 
                os.path.join(output_dir, 'precision_recall_curve.png')
            )
            
            conf_matrix = plot_confusion_matrix(
                test_labels, test_preds, 
                os.path.join(output_dir, 'confusion_matrix.png')
            )
            
            create_model_dashboard(
                test_scores, test_labels, train_accs, val_accs, 
                best_epoch, os.path.join(output_dir, 'dashboard.png')
            )
            
            plot_score_distribution(
                test_scores, test_labels, 
                os.path.join(output_dir, 'score_distribution.png')
            )
            
            # Save test results
            df_test_normalized['prediction'] = test_preds
            df_test_normalized['score'] = test_scores
            df_test_normalized.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)
            PrintUtils.end_stage()
            
            # Calculate and save metrics
            metrics = self.calculate_metrics(test_labels, test_scores, test_preds, conf_matrix, df)
            metrics['CommonName'] = common_name
            metrics['ChatBot'] = chatbot
            
            # Save metrics to results
            self.results.append(metrics)
            
            # Also save confusion matrix metrics to a text file
            self.save_confusion_metrics(test_labels, test_preds, conf_matrix, output_dir)
            
            PrintUtils.print_extra(f'Completed processing for {chatbot}')
            PrintUtils.print_extra(f'AUC: {metrics["AUC"]:.4f}, F1: {metrics["F1 Score"]:.4f}')
            
        except Exception as e:
            PrintUtils.end_stage(fail_message=f"Failed to process {chatbot}: {str(e)}")
            raise
        
        PrintUtils.end_stage()
    
    def reprocess_chatbot_statistics(self, chatbot, common_name, output_dir, prompts):
        """
        Reprocess statistics for an existing chatbot result without retraining
        
        Args:
            chatbot: Name of the chatbot to reprocess
            common_name: Common name for the chatbot
            output_dir: Directory with the chatbot's results
            prompts: Loaded prompts dictionary
        """
        PrintUtils.start_stage(f'Reprocessing statistics for {chatbot} (common name: {common_name})')
        
        try:
            # Check if test results exist
            test_results_path = os.path.join(output_dir, 'test_results.csv')
            if not os.path.exists(test_results_path):
                PrintUtils.print_extra(f'No test results found for {chatbot}, skipping')
                return
                
            # Load test results
            df_test = pd.read_csv(test_results_path)
            test_labels = df_test['target'].values
            test_scores = df_test['score'].values
            test_preds = df_test['prediction'].values
            
            # Load original data for size statistics
            df = self.load_chatbot_data(chatbot, prompts)
            
            # Calculate confusion matrix
            conf_matrix = confusion_matrix(test_labels, test_preds)
            
            # Calculate and save metrics
            metrics = self.calculate_metrics(test_labels, test_scores, test_preds, conf_matrix, df)
            metrics['CommonName'] = common_name
            metrics['ChatBot'] = chatbot
            
            # Update or add to results
            updated = False
            for i, result in enumerate(self.results):
                if result['CommonName'] == common_name and result['ChatBot'] == chatbot:
                    self.results[i] = metrics
                    updated = True
                    break
                    
            if not updated:
                self.results.append(metrics)
                
            # Also update confusion matrix metrics file
            self.save_confusion_metrics(test_labels, test_preds, conf_matrix, output_dir)
                
            PrintUtils.print_extra(f'Reprocessed statistics for {chatbot}')
            PrintUtils.print_extra(f'AUC: {metrics["AUC"]:.4f}, F1: {metrics["F1 Score"]:.4f}')
            
        except Exception as e:
            PrintUtils.end_stage(fail_message=f"Failed to reprocess {chatbot}: {str(e)}")
            
        PrintUtils.end_stage()
    
    def save_confusion_metrics(self, test_labels, test_preds, conf_matrix, output_dir):
        """
        Save detailed confusion matrix metrics to a text file
        
        Args:
            test_labels: True labels
            test_preds: Predicted labels
            conf_matrix: Confusion matrix
            output_dir: Directory to save the metrics file
        """
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        f1 = 2 * precision_val * sensitivity / (precision_val + sensitivity) if (precision_val + sensitivity) > 0 else 0
        
        with open(os.path.join(output_dir, 'confusion_matrix_metrics.txt'), 'w') as f:
            f.write(f'Confusion Matrix:\n')
            f.write(f'              Predicted Negative  Predicted Positive\n')
            f.write(f'Actual Negative      {tn:<18} {fp}\n')
            f.write(f'Actual Positive      {fn:<18} {tp}\n\n')

            f.write(f'Confusion Matrix Metrics:\n')
            f.write(f'Sensitivity/Recall: {sensitivity:.4f}\n')
            f.write(f'Specificity:        {specificity:.4f}\n')
            f.write(f'Precision:          {precision_val:.4f}\n')
            f.write(f'Negative Pred Value: {npv:.4f}\n')
            f.write(f'Accuracy:           {accuracy:.4f}\n')
            f.write(f'F1 Score:           {f1:.4f}\n')
    
    def load_chatbot_data(self, chatbot, prompts):
        """
        Load data for a specific chatbot
        
        Args:
            chatbot: Name of the chatbot to load data for
            prompts: Dictionary of prompts
            
        Returns:
            DataFrame: Loaded and processed data
        """
        PrintUtils.start_stage(f'Loading data for {chatbot}')
        
        training_set_dir = os.path.join(self.get_self_dir(), 'data')
        files = [
            os.path.join(training_set_dir, i) 
            for i in os.listdir(training_set_dir) 
            if i.lower().endswith(f'_{chatbot.lower()}.seq')
        ]
        
        if not files:
            raise ValueError(f"No training data found for chatbot {chatbot}")
            
        data = []
        for file_index, file_path in enumerate(files):
            with open(file_path, 'r') as fp:
                data.append(json.load(fp))
                
            if file_index % 10 == 0:
                percentage = (file_index * 100) // len(files)
                PrintUtils.start_stage(
                    f'Loading data for {chatbot} ({file_index}/{len(files)} = {percentage}%)', 
                    override_prev=True
                )
                
        df = pd.DataFrame(data)
        
        # Add target column
        df['target'] = df['prompt'].apply(
            lambda x: 1 if x in prompts['positive']['prompts'] else 0
        )
        
        total_positives = df['target'].sum()
        total_negatives = len(df) - total_positives
        
        PrintUtils.print_extra(f'Loaded {len(df)} samples for {chatbot}')
        PrintUtils.print_extra(f'Positive samples: {total_positives}, Negative samples: {total_negatives}')
        
        PrintUtils.end_stage()
        return df
    
    def split_data(self, df):
        """
        Split data into train, validation, and test sets
        
        Args:
            df: DataFrame to split
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        PrintUtils.start_stage('Splitting data into train, validation, and test sets')
        
        # Split into train and test sets preserving prompt distribution
        unique_prompts = df.drop_duplicates(subset=['prompt'])[['prompt', 'target']]
        train_and_val_prompts, test_prompts = train_test_split(
            unique_prompts['prompt'],
            test_size=self.config.test_size / 100,
            random_state=self.config.seed,
            stratify=unique_prompts['target']
        )
        test_prompts = set(test_prompts)
        
        # Split train into train and validation
        df_train, df_val = train_test_split(
            df[df['prompt'].isin(train_and_val_prompts)],
            test_size=self.config.valid_size / 100,
            random_state=self.config.seed,
            stratify=df[df['prompt'].isin(train_and_val_prompts)]['target']
        )
        
        df_test = df[df['prompt'].isin(test_prompts)]
        
        PrintUtils.print_extra(f'Train set size: {len(df_train)}')
        PrintUtils.print_extra(f'Validation set size: {len(df_val)}')
        PrintUtils.print_extra(f'Test set size: {len(df_test)}')
        
        PrintUtils.end_stage()
        return df_train, df_val, df_test
    
    def calculate_metrics(self, test_labels, test_scores, test_preds, conf_matrix, df):
        """
        Calculate benchmark metrics for a chatbot
        
        Args:
            test_labels: True labels
            test_scores: Prediction scores (probabilities)
            test_preds: Binary predictions
            conf_matrix: Confusion matrix
            df: Original dataframe with all data
            
        Returns:
            dict: Dictionary of calculated metrics
        """
        # Calculate basic classification metrics
        tn, fp, fn, tp = conf_matrix.ravel()
        
        # Calculate AUC
        auc_score = roc_auc_score(test_labels, test_scores)
        
        # Calculate AUPRC
        precision, recall, _ = precision_recall_curve(test_labels, test_scores)
        auprc = auc(recall, precision)
        
        # Calculate other metrics
        f1 = f1_score(test_labels, test_preds)
        recall = recall_score(test_labels, test_preds)
        precision = precision_score(test_labels, test_preds)
        accuracy = accuracy_score(test_labels, test_preds)
        
        # Calculate data statistics
        data_lengths = df['data_lengths'].apply(len)
        median_data_length = np.median(data_lengths)
        avg_data_length = np.mean(data_lengths)
        stddev_data_length = np.std(data_lengths)
        
        # Calculate size statistics (sum of data_lengths)
        data_sizes = df['data_lengths'].apply(lambda x: sum(x))
        median_data_size = np.median(data_sizes)
        avg_data_size = np.mean(data_sizes)
        stddev_data_size = np.std(data_sizes)
        
        # Calculate token statistics (assuming tokens ~ data_lengths)
        median_tokens = median_data_length
        avg_tokens = avg_data_length
        stddev_tokens = stddev_data_length
        
        # Combine all metrics
        metrics = {
            'AUC': auc_score,
            'AUPRC': auprc,
            'F1 Score': f1,
            'Recall': recall,
            'Precision': precision,
            'Accuracy': accuracy,
            'Total': len(test_labels),
            'Positives': int(test_labels.sum()),
            'Negatives': int(len(test_labels) - test_labels.sum()),
            'Median Data Length': median_data_length,
            'Avg Data Length': avg_data_length,
            'StdDev Data Length': stddev_data_length,
            'Median Data Size': median_data_size,
            'Avg Data Size': avg_data_size,
            'StdDev Data Size': stddev_data_size,
            'Median Number of Tokens': median_tokens,
            'Avg Number of Tokens': avg_tokens,
            'StdDev Number of Tokens': stddev_tokens
        }
        
        return metrics
    
    def save_results(self):
        """
        Save benchmark results to CSV file
        """
        PrintUtils.start_stage('Saving benchmark results')
        
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.results_file, index=False)
        
        PrintUtils.print_extra(f'Results saved to {self.results_file}')
        PrintUtils.end_stage()


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = ThrowingArgparse()
    parser.add_argument(
        '-c', '--config', 
        help='Path to YAML configuration file',
        default='benchmark.yaml'
    )
    parser.add_argument(
        '-r', '--reprocess', 
        help='Reprocess statistics without retraining models',
        action='store_true'
    )
    
    return parser.parse_args()


def main():
    """
    Main function to run the benchmark script
    """
    # Print logo
    PrintUtils.print_logo()
    
    # Catch-all for clean error handling
    is_user_cancelled = False
    last_error = None
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Run benchmark
        benchmark = BenchmarkRunner(args.config, reprocess_only=args.reprocess)
        benchmark.run_benchmark()
        
    except KeyboardInterrupt:
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message='', throw_on_fail=False)
        PrintUtils.print_extra('Operation *cancelled* by user - please wait for cleanup code to complete')
        is_user_cancelled = True
        
    except Exception as ex:
        # Optionally fail stage
        if PrintUtils.is_in_stage():
            PrintUtils.end_stage(fail_message=ex, throw_on_fail=False)
            
        # Save error and print it as an extra
        PrintUtils.print_extra(f'Error: {ex}')
        last_error = ex
        
    finally:
        # Print final status
        if last_error is not None:
            PrintUtils.print_error(f'{last_error}\n')
        elif is_user_cancelled:
            PrintUtils.print_extra(f'Operation *cancelled* by user\n')
        else:
            PrintUtils.print_extra(f'Benchmark completed successfully\n')


if __name__ == '__main__':
    main()
