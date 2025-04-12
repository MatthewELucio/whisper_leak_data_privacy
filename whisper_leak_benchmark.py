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
from enum import Enum
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc,
    f1_score, recall_score, precision_score, 
    accuracy_score, confusion_matrix
)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

from core.chatbot_utils import ChatbotUtils
from core.classifiers.attention_bi_lstm_classifier import AttentionBiLSTMClassifier
from core.classifiers.bert_time_series_classifier import BERTTimeSeriesClassifier
from core.classifiers.cnn_classifier import CNNClassifier
from core.classifiers.combined_lstm_bert_classifier import CombinedLSTMBERTClassifier
from core.classifiers.lstm_transformer_classifier import LSTMTransformerClassifier
from core.classifiers.loader import Loader

from core.classifiers.utils import ( EarlyStopping, apply_neg_sampling,
    set_seed, split_data, train_epoch, eval_epoch, get_prediction_scores
)

from core.classifiers.visualization import (
    calculate_metrics, set_plot_style, plot_training_curves, plot_roc_curve,
    plot_precision_recall_curve, plot_confusion_matrix,
    plot_score_distribution, create_model_dashboard
)

from core.utils import ThrowingArgparse, PrintUtils

import json


class FeatureMode(Enum):
    """
    Enum representing the feature mode used for classification
    """
    BOTH = "Both"
    DATA_SIZE_ONLY = "Data Size Only"
    TIME_ONLY = "Time Only"


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
            
            # If * in chatbots, then load the full set of chatbots
            if '*' in self.chatbots:
                self.chatbots = ChatbotUtils.load_chatbots(
                    os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        'chatbots'
                    )).keys()

                # Shuffle the chatbots to process in a predictable order
                self.chatbots = list(self.chatbots)
                np.random.seed(42)
                np.random.shuffle(self.chatbots)
                print(f"Chatbots: {self.chatbots}")
            
            self.benchmark_name = self.config.get('benchmark_name', 'benchmark')

            # Handle sampling_rate (single value or list)
            sampling_rate_config = self.config.get('sampling_rate', 1.0)
            if isinstance(sampling_rate_config, (float, int)):
                self.sampling_rates = [float(sampling_rate_config)]
            elif isinstance(sampling_rate_config, list):
                self.sampling_rates = [float(rate) for rate in sampling_rate_config]
            else:
                raise ValueError("sampling_rate must be a float or a list of floats")
            # Validate rates are between 0 and 1
            if not all(0.0 < rate <= 1.0 for rate in self.sampling_rates):
                 raise ValueError("All sampling_rates must be between 0.0 (exclusive) and 1.0 (inclusive)")
            
            self.num_trials = self.config.get('num_trials', 1)
            self.data_path = self.config.get('data_path', 'data')
            self.oversampling_train_neg_to_pos = self.config.get('oversampling_train_neg_to_pos', None)
            self.oversampling_eval_neg_to_pos = self.config.get('oversampling_eval_neg_to_pos', None)
            
            # Get model configuration
            model_config = self.config.get('model', {})
            if not model_config:
                raise ValueError("No model parameters specified in configuration")

            self.model_class = model_config.get('class', 'LSTMBinaryClassifier')
            self.model_params = model_config.get('params', {})

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
            PrintUtils.print_extra(f'Number of trials: *{self.num_trials}*')
            PrintUtils.print_extra(f'Sampling rates: *{[f"{rate*100:.1f}%" for rate in self.sampling_rates]}*')
            
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
        os.makedirs(self.benchmark_dir, exist_ok=True)
        self.results_file = os.path.join(self.benchmark_dir, f'{self.config.benchmark_name}.csv')
        self.output_dir = os.path.join(self.benchmark_dir, self.config.benchmark_name)
        
        # Create necessary directories
        os.makedirs(self.output_dir, exist_ok=True)
        
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
            try:
                df = pd.read_csv(self.results_file)
                # Check if the 'Sampling Rate' column exists for backward compatibility
                has_sampling_rate_col = 'Sampling Rate' in df.columns
                for _, row in df.iterrows():
                    # Use default 1.0 if column is missing
                    sampling_rate = row['Sampling Rate'] if has_sampling_rate_col else 1.0
                    key = (row['CommonName'], row['ChatBot'], row['Features'], row['Trial'], sampling_rate)
                    self.existing_results[key] = True
                if not has_sampling_rate_col:
                     PrintUtils.print_extra("Loaded existing results CSV without 'Sampling Rate' column. Assuming 1.0 for old entries.")
            except Exception as e:
                PrintUtils.print_extra(f"Warning: Failed to load or parse existing results file '{self.results_file}'. Will proceed without skipping based on it. Error: {e}")
                self.existing_results = {} # Reset if loading failed
    
    def get_self_dir(self):
        """
        Get the directory where this script is located
        """
        return os.path.dirname(os.path.abspath(__file__))
    
    def should_process_chatbot_trial(self, chatbot, common_name, features, trial, sampling_rate):
        """
        Determine if we should process this chatbot trial for a specific sampling rate
        based on existing results.

        Args:
            chatbot: Name of the chatbot to check
            common_name: Common name for the chatbot
            features: Feature mode (Both, Data Size Only, Time Only)
            trial: Trial number
            sampling_rate: The specific sampling rate for this run

        Returns:
            bool: True if chatbot trial should be processed, False otherwise
        """
        key = (common_name, chatbot, features, trial, sampling_rate)
        return key not in self.existing_results
    
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
        
        PrintUtils.end_stage()

        # Process each trial
        for trial in range(1, self.config.num_trials + 1):
            # Process each feature mode
            for feature_mode in FeatureMode:
                for current_sampling_rate in self.config.sampling_rates:
                    # Process each chatbot
                    for chatbot in self.config.chatbots:
                        PrintUtils.start_stage(f'Processing chatbot: *{chatbot}* (feature mode: {feature_mode.value}, trial: {trial}, sampling rate: {current_sampling_rate})', override_prev=True)
                        PrintUtils.end_stage()

                        chatbot_dir = os.path.join(self.output_dir, chatbot)
                        os.makedirs(chatbot_dir, exist_ok=True)
                        
                        trial_dir = os.path.join(chatbot_dir, f"{feature_mode.value}", f"trial_{trial}")
                        os.makedirs(trial_dir, exist_ok=True)
                        
                        if not self.should_process_chatbot_trial(chatbot, chatbot, feature_mode.value, trial, current_sampling_rate) and not self.reprocess_only:
                            PrintUtils.start_stage(f'Skipping *{chatbot}* (feature mode: {feature_mode.value}, trial: {trial}) as it already exists in results')
                            PrintUtils.end_stage()
                            continue
                        
                        PrintUtils.start_stage(f'Processing chatbot: *{chatbot}* (feature mode: {feature_mode.value}, trial: {trial})')
                        
                        # If reprocess_only, just calculate statistics without training
                        try:
                            if self.reprocess_only:
                                self.reprocess_chatbot_statistics(chatbot, chatbot, trial_dir, prompts, feature_mode, trial)
                            else:
                                self.process_chatbot(chatbot, chatbot, trial_dir, prompts, feature_mode, trial, current_sampling_rate)
                        except Exception as e:
                            PrintUtils.print_extra(f"Failed to process {chatbot} (feature mode: {feature_mode.value}, trial: {trial}): {str(e)}")
                            continue
                        
                        PrintUtils.end_stage()
        
        # Save all results to CSV
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(self.results_file, index=False)
        PrintUtils.print_extra(f'Results saved to {self.results_file}')
        
        PrintUtils.end_stage()
    
    def process_chatbot(self, chatbot, common_name, output_dir, prompts, feature_mode, trial, sampling_rate):
        """
        Process a single chatbot - train model and evaluate performance
        
        Args:
            chatbot: Name of the chatbot to process
            common_name: Common name for the chatbot
            output_dir: Directory to save results
            prompts: Loaded prompts dictionary
            feature_mode: Which features to use (Both, Data Size Only, Time Only)
            trial: Trial number
            sampling_rate: The sampling rate for this specific run
        """
        try:
            # Set trial-specific seed for reproducibility
            trial_seed = self.config.seed + trial
            set_seed(trial_seed)
            PrintUtils.print_extra(f'Using trial-specific seed: *{trial_seed}*')
            
            # Load dataset
            df = self.load_chatbot_data(chatbot, prompts, sampling_rate=1.0)
            
            # Split data with trial-specific seed
            PrintUtils.start_stage('Splitting data into train, validation, and test sets', override_prev=True)
            df_train, df_val, df_test = split_data(df, trial_seed, self.config.test_size / 100.0, self.config.valid_size / 100.0)
            PrintUtils.print_extra(f'Train set size: {len(df_train)}')
            PrintUtils.print_extra(f'Validation set size: {len(df_val)}')
            PrintUtils.print_extra(f'Test set size: {len(df_test)}')
            PrintUtils.end_stage()

            # Downsample the training set if needed
            if sampling_rate < 1.0:
                df_train = df_train.sample(frac=sampling_rate, random_state=trial_seed).reset_index(drop=True)
                PrintUtils.print_extra(f'Downsampled training set size: {len(df_train)}')

            # Adjust based on feature mode
            if feature_mode == FeatureMode.DATA_SIZE_ONLY:
                # Copy data_lengths column to time_diffs for compatibility
                df_train = df_train.copy()
                df_val = df_val.copy()
                df_test = df_test.copy()
                df_train['time_diffs'] = df_train['data_lengths']
                df_val['time_diffs'] = df_val['data_lengths']
                df_test['time_diffs'] = df_test['data_lengths']
            elif feature_mode == FeatureMode.TIME_ONLY:
                # Copy time_diffs column to data_lengths for compatibility
                df_train = df_train.copy()
                df_val = df_val.copy()
                df_test = df_test.copy()
                df_train['data_lengths'] = df_train['time_diffs']
                df_val['data_lengths'] = df_val['time_diffs']
                df_test['data_lengths'] = df_test['time_diffs']
            
            # Prepare data
            PrintUtils.start_stage('Preparing data')
            train_dataset = Loader(df_train)
            val_dataset = Loader(df_val)
            test_dataset = Loader(df_test)
            norms = train_dataset.get_normalization()
            train_dataset.apply_normalization(norms)
            val_dataset.apply_normalization(norms)
            test_dataset.apply_normalization(norms)

            # Oversample the training data and shuffle
            if self.config.oversampling_train_neg_to_pos:
                # Trim train to only required columns to reduce memory usage
                df_train = train_dataset.df

                df_train = apply_neg_sampling(
                    df_train, 
                    self.config.oversampling_train_neg_to_pos, 
                )
                train_dataset.df = df_train
                PrintUtils.print_extra(f'Oversampled training set. Positives: {df_train["target"].sum()}, Negatives: {len(df_train) - df_train["target"].sum()}')

            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
            
            # Initialize model based on configuration
            model = self.create_model(df_train, norms, feature_mode)
            model = model.to(self.device)
            
            # Setup training
            criterion = nn.BCEWithLogitsLoss()
            
            if hasattr(model, 'get_optimizer_params'):
                optimizer_params = model.get_optimizer_params(self.config.learning_rate)
                PrintUtils.print_extra(f"Using model-specific parameter groups for optimizer")
                optimizer = optim.Adam(optimizer_params)
            else:
                # Fallback to regular optimizer for other models
                optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
            early_stopping = EarlyStopping(
                patience=self.config.patience,
                verbose=True,
                path=os.path.join(output_dir, 'best_model.pt')
            )
            
            # Train model
            model_file = os.path.join(output_dir, 'model.pth')
            train_losses, val_losses, test_losses = [], [], []
            train_accs, val_accs, test_accs = [], [], []
            best_epoch = 0
            
            PrintUtils.start_stage(f'Training model for {chatbot} (feature mode: {feature_mode.value}, trial: {trial})')
            for epoch in range(self.config.max_epochs):
                PrintUtils.start_stage(f'Training epoch {epoch+1}/{self.config.max_epochs}', override_prev=True)
                
                train_loss, train_acc = train_epoch(
                    model, train_loader, criterion, optimizer, 
                    self.device, epoch, self.config.max_epochs
                )
                val_loss, val_acc = eval_epoch(model, val_loader, criterion, self.device, neg_to_pos_ratio=self.config.oversampling_train_neg_to_pos)
                test_loss, test_acc = eval_epoch(model, test_loader, criterion, self.device, neg_to_pos_ratio=self.config.oversampling_train_neg_to_pos)
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                test_losses.append(test_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                test_accs.append(test_acc)
                
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
            model.save(model_file)

            # Also save to ./models/<chatbot>_<feature_mode>_<trial>.pth
            model_path = os.path.join(self.get_self_dir(), 'models', f'{chatbot}_{feature_mode.value}_{trial}.pth')
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            model.save(model_path)

            modes = ["standard"]
            if self.config.oversampling_eval_neg_to_pos:
                modes.append("oversampling")

            for mode in modes:
                # Create visualizations
                PrintUtils.start_stage(f'Creating visualizations for {chatbot} {mode}')
                if mode == "standard":
                    neg_to_pos_ratio = None
                else:
                    neg_to_pos_ratio = self.config.oversampling_eval_neg_to_pos
                
                test_scores, test_labels, test_loss = get_prediction_scores(model, test_loader, self.device, neg_to_pos_ratio=neg_to_pos_ratio)
                test_preds = (test_scores > 0.5).astype(int)
                
                # Generate all plots
                plot_training_curves(
                    train_losses, val_losses, test_losses, train_accs, val_accs, test_accs,
                    best_epoch, os.path.join(output_dir, f'training_curves_{mode}.png')
                )
                
                plot_roc_curve(
                    test_labels, test_scores, 
                    os.path.join(output_dir, f'roc_curve_{mode}.png')
                )
                
                plot_precision_recall_curve(
                    test_labels, test_scores, 
                    os.path.join(output_dir, f'precision_recall_curve_{mode}.png')
                )
                
                conf_matrix = plot_confusion_matrix(
                    test_labels, test_preds, 
                    os.path.join(output_dir, f'confusion_matrix_{mode}.png')
                )
                
                create_model_dashboard(
                    test_scores, test_labels, train_losses, val_losses, 
                    best_epoch, os.path.join(output_dir, f'dashboard_{mode}.png')
                )
                
                plot_score_distribution(
                    test_scores, test_labels, 
                    os.path.join(output_dir, f'score_distribution_{mode}.png')
                )
                
                # Calculate and save metrics
                metrics = calculate_metrics(test_labels, test_scores, test_preds, conf_matrix, df)
                metrics['CommonName'] = common_name
                metrics['ChatBot'] = chatbot
                metrics['Features'] = feature_mode.value
                metrics['Trial'] = trial
                metrics['Mode'] = mode
                metrics['Sampling Rate'] = sampling_rate

                # Save metrics to results
                self.results.append(metrics)

                # Save metrics to CSV
                results_df = pd.DataFrame(self.results)

                # Move common columns to the front
                common_cols = ['CommonName', 'ChatBot', 'Features', 'Trial', 'Mode', 'Sampling Rate']
                all_cols = results_df.columns.tolist()
                other_cols = [col for col in all_cols if col not in common_cols]
                new_col_order = common_cols + other_cols
                results_df = results_df[new_col_order]

                results_df.to_csv(self.results_file, index=False)
                PrintUtils.print_extra(f'Results saved to {self.results_file}')
                
                # Also save confusion matrix metrics to a text file
                self.save_confusion_metrics(test_labels, test_preds, conf_matrix, 
                        os.path.join(output_dir, f'confusion_matrix_metrics_{mode}.txt')
                    )
                
                PrintUtils.print_extra(f'Completed processing for {chatbot} (feature mode: {feature_mode.value}, trial: {trial})')
                PrintUtils.print_extra(f'AUPRC: {metrics["AUPRC"]:.4f}, F1: {metrics["F1 Score"]:.4f}')
                PrintUtils.end_stage()

            # Save test results
            test_scores, test_labels, test_loss = get_prediction_scores(model, test_loader, self.device)
            test_preds = (test_scores > 0.5).astype(int)
            df_test = test_dataset.df.copy()
            df_test['prediction'] = test_preds
            df_test['score'] = test_scores
            df_test.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)
            
            
        except Exception as e:
            PrintUtils.end_stage(fail_message=f"Failed to process {chatbot} (feature mode: {feature_mode.value}, trial: {trial}): {str(e)}")
            raise
        
        PrintUtils.end_stage()
    
    
    def create_model(self, df_train, normalization_params, feature_mode):
        """
        Create model based on configuration
        
        Args:
            normalization_params: Normalization parameters
            feature_mode: Which features to use
            
        Returns:
            model: Initialized model
        """
        input_channels = 1 if feature_mode in [FeatureMode.DATA_SIZE_ONLY, FeatureMode.TIME_ONLY] else 2
        
        if self.config.model_class == 'CNNClassifier':
            model = CNNClassifier(
                normalization_params,
                **self.config.model_params,
            )
        elif self.config.model_class == 'AttentionBiLSTMClassifier':
            model = AttentionBiLSTMClassifier(
                normalization_params,
                **self.config.model_params,
            )
        elif self.config.model_class == 'LSTMTransformerClassifier':
            model = LSTMTransformerClassifier(
                normalization_params,
                **self.config.model_params,
            )
        elif self.config.model_class == 'CombinedLSTMBERTClassifier':
            model = CombinedLSTMBERTClassifier(
                normalization_params,
                **self.config.model_params
            )
        elif self.config.model_class == 'BERTTimeSeriesClassifier':
            (time_boundaries_norm, len_boundaries_norm) = BERTTimeSeriesClassifier.calculate_boundaries(
                df_train,
                num_buckets=self.config.model_params.get('num_buckets', 50),
                norm=normalization_params
            )

            interleave = True
            if feature_mode == FeatureMode.DATA_SIZE_ONLY or feature_mode == FeatureMode.TIME_ONLY:
                interleave = False

            model = BERTTimeSeriesClassifier(
                normalization_params,
                time_boundaries_norm,
                len_boundaries_norm,
                **self.config.model_params
            )
        else:
            raise ValueError(f"Unsupported model class: {self.config.model_class}")
        
        return model
    
    
    def save_confusion_metrics(self, test_labels, test_preds, conf_matrix, output_file):
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
        
        with open(output_file, 'w') as f:
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
    
    def load_chatbot_data(self, chatbot, prompts, sampling_rate):
        """
        Load data for a specific chatbot
        
        Args:
            chatbot: Name of the chatbot to load data for
            prompts: Dictionary of prompts
            sampling_rate: The fraction of data files to load (0.0 to 1.0)
            
        Returns:
            DataFrame: Loaded and processed data
        """
        PrintUtils.start_stage(f'Loading data for {chatbot}, sampling rate {sampling_rate*100:.1f}%', override_prev=True)
        
        training_set_dir = os.path.join(self.get_self_dir(), self.config.data_path)
        files = [
            os.path.join(training_set_dir, i) 
            for i in os.listdir(training_set_dir) 
            if i.lower().endswith(f'_{chatbot.lower()}.seq')
        ]
        
        if not files:
            raise ValueError(f"No training data found for chatbot {chatbot}")

        if sampling_rate < 1.0:
            # Sample files based on sampling rate
            sample_size = max(int(len(files) * sampling_rate), 1)
            files = np.random.choice(files, size=sample_size, replace=False).tolist()

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
