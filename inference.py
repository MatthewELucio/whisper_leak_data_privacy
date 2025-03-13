import argparse
import logging
import os
import torch
import pandas as pd
import ast
import numpy as np
from tqdm import tqdm

from core.classifier import *
from core.utils import set_seed
from core.visualization import plot_score_distribution

def setup_logging():
    """Set up logging configuration."""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/inference.log'),
            logging.StreamHandler()
        ]
    )

def prepare_inference_data(data_path, time_norm_params, size_norm_params, max_len):
    """Load and prepare data for inference."""
    logging.info(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Preprocess lists in the dataframe
    df['time_diffs'] = df['time_diffs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['data_lengths'] = df['data_lengths'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Normalize the dataframe
    df_normalized = normalize_dataframe(df, time_norm_params, size_norm_params, max_len)
    
    # Create dataset
    dataset = PreprocessedTextDataset(df_normalized, max_len)
    
    return df, df_normalized, dataset

def run_inference(model, dataset, device, batch_size=32):
    """Run inference on the given dataset."""
    from torch.utils.data import DataLoader
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    predictions = []
    scores = []
    
    with torch.no_grad():
        for X, _ in tqdm(dataloader, desc="Running inference"):
            X = X.to(device)
            outputs = model(X)
            predictions.extend((outputs.cpu().numpy() > 0.5).astype(int).flatten())
            scores.extend(outputs.cpu().numpy().flatten())
    
    return predictions, scores

def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Run inference with trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to input data file')
    parser.add_argument('--model', type=str, default='models/cnn_binary_classifier.pth', 
                        help='Path to trained model')
    parser.add_argument('--norm_params', type=str, default='models/normalization_params.npz',
                        help='Path to normalization parameters')
    parser.add_argument('--output', type=str, default='results/inference_results.csv',
                        help='Path to save results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--kernel_width', type=int, default=3, help='Kernel width used in the model')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda', 
                        choices=['cuda', 'cpu'], help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    set_seed(args.seed)
    
    # Determine device
    device_name = "cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu"
    device = torch.device(device_name)
    logging.info(f"Using device: {device}")
    
    # Load normalization parameters
    time_norm_params, size_norm_params, max_len = load_normalization_params(args.norm_params)
    logging.info(f"Loaded normalization parameters with max_len: {max_len}")
    
    # Load model
    model = load_model(args.model, args.kernel_width, max_len, device)
    
    # Prepare data
    original_df, normalized_df, dataset = prepare_inference_data(
        args.data, time_norm_params, size_norm_params, max_len
    )
    
    # Run inference
    logging.info("Running inference...")
    predictions, scores = run_inference(model, dataset, device, args.batch_size)
    
    # Add predictions to dataframe
    normalized_df['prediction'] = predictions
    normalized_df['score'] = scores
    
    # Visualize score distribution
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plot_score_distribution(scores, os.path.join(os.path.dirname(args.output), 'inference_score_distribution.png'))
    
    # Save results
    normalized_df.to_csv(args.output, index=False)
    logging.info(f"Results saved to {args.output}")
    
    # Print summary
    positive_count = sum(predictions)
    total_count = len(predictions)
    logging.info(f"Inference complete! {positive_count} positive predictions out of {total_count} samples "
                f"({positive_count/total_count:.2%})")

if __name__ == "__main__":
    main()
