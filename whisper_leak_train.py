#!/usr/bin/env python3
from core.classifiers.base_classifier import BaseClassifier
from core.classifiers.cnn_classifier import CNNClassifier
from core.classifiers.attention_bi_lstm_classifier import AttentionBiLSTMClassifier
from core.classifiers.bert_time_series_classifier import BERTTimeSeriesClassifier
from core.classifiers.utils import EarlyStopping
from core.classifiers.utils import set_seed
from core.classifiers.utils import train_epoch
from core.classifiers.utils import eval_epoch
from core.classifiers.utils import get_prediction_scores
from core.classifiers.loader import Loader

from core.classifiers.visualization import calculate_metrics, set_plot_style
from core.classifiers.visualization import plot_training_curves
from core.classifiers.visualization import plot_roc_curve
from core.classifiers.visualization import plot_precision_recall_curve
from core.classifiers.visualization import plot_confusion_matrix 
from core.classifiers.visualization import plot_score_distribution
from core.classifiers.visualization import create_model_dashboard

from core.utils import ThrowingArgparse
from core.utils import PrintUtils
from core.utils import OsUtils
from core.utils import PromptUtils
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
    chatbot_names = ', '.join([ f'*{chatbot.__name__}*' for chatbot in chatbots.values() ])
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
    parser.add_argument('-v', '--validsize', type=int, help='The validation size in percentage taken from train set', default=5)
    parser.add_argument('-ds', '--downsample', type=float, help='Downsample the dataset', default=1.0)
    parser.add_argument('-i', '--input_folder', type=str, help='Input folder for the data', default='data')
    args = parser.parse_args()
    assert args.seed >= 0, Exception(f'Invalid random seed: {args.seed}')
    assert args.batchsize > 0, Exception(f'Invalid batch size: {args.batchsize}')
    assert args.epochs > 0, Exception(f'Invalid number of epochs: {args.epochs}')
    assert args.patience > 0, Exception(f'Invalid patience value: {args.patience}')
    assert args.kernelwidth > 0, Exception(f'Invalid kernel width: {args.kernelwidth}')
    assert args.learningrate > 0 and args.learningrate < 1, Exception(f'Invalid learning rate: {args.learningrate}')
    assert args.testsize > 0 and args.testsize < 100, Exception(f'Invalid test size percentage: {args.testsize}')
    assert args.validsize > 0 and args.validsize < 100, Exception(f'Invalid validation size percentage: {args.validsize}')
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
        training_set_dir = os.path.join(get_self_dir(), args.input_folder)
        files = [ os.path.join(training_set_dir, i) for i in os.listdir(training_set_dir) if i.lower().endswith(f'_{args.chatbot.lower()}.seq') ]
        assert len(files) > 0, Exception(f'Did not find training set files for chatbot {args.chatbot}')
        data = []
        file_index = 0

        if args.downsample < 1.0:
            # Downsample the files to a fraction of the original size
            files = files[:int(len(files) * args.downsample)]

        for file_index in range(len(files)):
            with open(files[file_index], 'r') as fp:
                data.append(json.load(fp))
            percentage = (file_index * 100) // len(files)
            if file_index % 10 == 0:
                PrintUtils.start_stage(f'Loading sequences data ({file_index} / {len(files)} = {percentage}%)', override_prev=True)
        df = pd.DataFrame(data)

        PrintUtils.end_stage()

        # Join to prompts to add target column
        prompts = PromptUtils.read_prompts(args.prompts)
        df['target'] = df['prompt'].apply(lambda x: 1 if x in prompts['positive']['prompts'] else 0)
        PrintUtils.end_stage()
        total_prompts = len(prompts['positive']['prompts']) + len(prompts['negative']['prompts'])
        PrintUtils.print_extra(f'Loaded {total_prompts} prompts')

        # Split into train and test sets and hold out a percentage of unique 'prompts' values for test set
        PrintUtils.start_stage('Splitting into train and test sets')
        unique_prompts = df.drop_duplicates(subset=['prompt'])[['prompt', 'target']]
        train_and_val_prompts, test_prompts = train_test_split(
            unique_prompts['prompt'],
            test_size=args.testsize / 100,
            random_state=args.seed,
            stratify=unique_prompts['target']
        )
        test_prompts = set(test_prompts)
        #train_prompts, val_prompts = train_test_split(
        #    train_and_val_prompts,
        #    test_size=args.validsize / 100,
        #    random_state=args.seed,
        #    stratify=unique_prompts[unique_prompts['prompt'].isin(train_and_val_prompts)]['target']
        #)
        #train_prompts = set(train_prompts)
        #val_prompts = set(val_prompts)
        df_train, df_val = train_test_split(
            df[df['prompt'].isin(train_and_val_prompts)],
            test_size=args.validsize / 100,
            random_state=args.seed,
            stratify=df[df['prompt'].isin(train_and_val_prompts)]['target']
        )

        #df_train = df[df['prompt'].isin(train_prompts)]
        #df_val = df[df['prompt'].isin(val_prompts)]
        df_test = df[df['prompt'].isin(test_prompts)]

        # Print the train, val, and test set sizes, and the number of unique prompts in each set by label, and count of prompts
        # in each set by label
        PrintUtils.print_extra(f'Train set size: {len(df_train)}')
        PrintUtils.print_extra(f'Validation set size: {len(df_val)}')
        PrintUtils.print_extra(f'Test set size: {len(df_test)}')
        PrintUtils.print_extra(f'Unique prompts in train set: {len(df_train["prompt"].unique())}')
        PrintUtils.print_extra(f'Unique prompts in validation set: {len(df_val["prompt"].unique())}')
        PrintUtils.print_extra(f'Unique prompts in test set: {len(df_test["prompt"].unique())}')
        PrintUtils.print_extra(f'Unique prompts in train set by label: {df_train.groupby("target")["prompt"].nunique().to_dict()}')
        PrintUtils.print_extra(f'Unique prompts in validation set by label: {df_val.groupby("target")["prompt"].nunique().to_dict()}')
        PrintUtils.print_extra(f'Unique prompts in test set by label: {df_test.groupby("target")["prompt"].nunique().to_dict()}')
        PrintUtils.print_extra(f'Count of prompts in train set by label: {df_train.groupby("target")["prompt"].count().to_dict()}')
        PrintUtils.print_extra(f'Count of prompts in validation set by label: {df_val.groupby("target")["prompt"].count().to_dict()}')
        PrintUtils.print_extra(f'Count of prompts in test set by label: {df_test.groupby("target")["prompt"].count().to_dict()}')

        # Calculate the average length of the sequences in each set
        PrintUtils.print_extra(f'Average sequence length in train set: {df_train["data_lengths"].apply(len).mean()}')
        PrintUtils.print_extra(f'Average sequence length in validation set: {df_val["data_lengths"].apply(len).mean()}')
        PrintUtils.print_extra(f'Average sequence length in test set: {df_test["data_lengths"].apply(len).mean()}')

        PrintUtils.end_stage()

        # Prepare data
        PrintUtils.start_stage('Preparing data')
        train_dataset = Loader(df_train)
        val_dataset = Loader(df_val)
        test_dataset = Loader(df_test)

        norm = train_dataset.get_normalization()
        train_dataset.apply_normalization(norm)
        val_dataset.apply_normalization(norm)
        test_dataset.apply_normalization(norm)

        train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False)

        PrintUtils.end_stage()
        PrintUtils.print_extra(f'Max sequence length being used for model (95th percentile): *{train_dataset.max_len}*')
    
        # Choose model architecture (CNN or LSTM)
        PrintUtils.start_stage('Instantiating model')
        model_type = args.modeltype.upper()
        if model_type == 'CNN':
            model = CNNClassifier(norm, args.kernelwidth).to(device)
            model_path = os.path.join(models_dir, 'cnn_binary_classifier.pth')
        elif model_type == 'LSTM':
            model = AttentionBiLSTMClassifier(
                norm
            ).to(device)
            model_path = os.path.join(models_dir, 'lstm_binary_classifier.pth')
        elif model_type == "BERT":
            # Calculate the token boundary parameters
            (time_boundaries_norm, len_boundaries_norm) = BERTTimeSeriesClassifier.calculate_boundaries(
                df_train,
                num_buckets=100,
                normalization_params=norm
            )

            model = BERTTimeSeriesClassifier(norm, time_boundaries_norm, len_boundaries_norm, num_buckets=100).to(device)
            model_path = os.path.join(models_dir, 'bert_binary_classifier.pth')
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
            early_stopping(val_loss, model)
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
        model.save(model_path)
        
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
        create_model_dashboard(test_scores, test_labels, train_losses, val_losses, best_epoch,  os.path.join(results_dir, 'model_performance_dashboard.png'))
        
        # Plot prediction score distribution
        plot_score_distribution(test_scores, test_labels, os.path.join(results_dir, 'prediction_score_distribution.png'))
        
        # Save test predictions
        df_test = test_dataset.df.copy()
        df_test['prediction'] = test_preds
        df_test['score'] = test_scores
        df_test.to_csv(os.path.join(results_dir, 'test_results.csv'), index=False)
        PrintUtils.end_stage()
        PrintUtils.print_extra('Results saved to *test_results.csv*')
        
        # Print confusion matrix metrics
        PrintUtils.start_stage('Printing confusion matrix metrics')
        
        metrics = calculate_metrics(test_labels, test_scores, test_preds, conf_matrix, df_test)

        # Iterate through printing key, value
        for key, value in metrics.items():
            PrintUtils.print_extra(f'{key}: {value}')

        # Also write to output file
        with open(os.path.join(results_dir, 'confusion_matrix_metrics.txt'), 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')
            
        PrintUtils.print_extra(f'Metrics saved to *confusion_matrix_metrics.txt*')
        PrintUtils.end_stage()
        
        # Test the inference function
        PrintUtils.start_stage('Testing inference function')

        model = BaseClassifier.load(model_path, device)

        sample_row = df_test.iloc[0]
        time_diffs, data_lengths = sample_row['time_diffs'], sample_row['data_lengths']
        
        # Test with tuple input
        prob, pred = model.inference(
            (time_diffs, data_lengths), 
            device
        )
        
        PrintUtils.print_extra(f'Inference result (tuple input): prob=*{prob:.4f}*, pred=*{pred}*')
        
        # Test with DataFrame input
        sample_df = df_test.iloc[[0]]
        probs, preds = model.inference(
            sample_df,
            device
        )
        PrintUtils.print_extra(f'Inference result (DataFrame input): prob=*{probs[0]:.4f}*, pred=*{preds[0]}*')
        PrintUtils.end_stage()

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
            PrintUtils.print_error(f'{last_error}\n')
        elif is_user_cancelled:
            PrintUtils.print_extra(f'Operation *cancelled* by user\n')
        else:
            PrintUtils.print_extra(f'Finished successfully\n')

if __name__ == '__main__':
    main()
