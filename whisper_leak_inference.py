#!/usr/bin/env python3
import os
import json
import argparse
import torch
import pandas as pd
from core.utils import PrintUtils, OsUtils, ThrowingArgparse, NetworkUtils
from core.classifier import BaseClassifier, PreprocessedTextDataset, normalize_dataframe
from core.model import Datapoint
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_self_dir():
    """Get the self directory."""
    return os.path.dirname(os.path.abspath(__file__))

def parse_arguments():
    """Parse script input arguments."""
    PrintUtils.start_stage('Parsing command-line arguments')
    parser = ThrowingArgparse()
    parser.add_argument('--prompt', type=str, help='The prompt to run inference on.')
    parser.add_argument('--data', type=str, help='Path to input data file, expected to be a txt with one line per prompt to run.')
    parser.add_argument('--models', nargs='+', help='Paths to trained model files (.pth). Supports multiple models.', default=["lstm_binary_classifier.pth"])
    parser.add_argument('--norm_params', nargs='+', help='Paths to normalization param files (.npz). Should correspond exactly to the number of models.', default=["lstm_binary_classifier_norm_params.npz"])
    parser.add_argument('--capture', type=str, help='If specified, use an existing capture (.pcap) for inference instead of capturing new data.')
    parser.add_argument('--output', type=str, default='results/inference_results.json', help='Path to output inference results.')
    parser.add_argument('-t', '--tlsport', type=int, default=443, help='Remote TLS port to capture data.')
    args = parser.parse_args()

    # Validation
    assert (args.prompt or args.data), "Either --prompt or --data must be specified"
    assert len(args.models) == len(args.norm_params), "Model paths and normalization parameters count mismatch."

    PrintUtils.end_stage()
    return args

def load_input_prompts(args):
    """Read prompts from input or prompt argument directly."""
    PrintUtils.start_stage('Loading input prompts')
    if args.prompt:
        prompts = [args.prompt]
    else:
        with open(args.data, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    assert prompts, "No prompts loaded."
    PrintUtils.print_extra(f'Loaded {len(prompts)} prompts for inference.')
    PrintUtils.end_stage()
    return prompts

def perform_capture(args, chatbot_class_name='inference'):
    """Perform network capture or use existing pcap file."""
    PrintUtils.start_stage('Network Capture')
    captures_path = os.path.join(get_self_dir(), 'captures')
    assert OsUtils.mkdir(captures_path)
    pcap_file = args.capture or os.path.join(captures_path, 'inference_capture.pcap')

    if args.capture:
        PrintUtils.print_extra(f'Using existing capture file: {args.capture}')
    else:
        NetworkUtils.start_sniffing_tls(pcap_file, args.tlsport)
        PrintUtils.print_extra('Capture started. Please ensure the prompt is sent manually now...')
        input("Press Enter once the interaction is complete.")
        NetworkUtils.stop_sniffing_tls()
        PrintUtils.print_extra(f'Capture completed and saved to {pcap_file}.')
    PrintUtils.end_stage()
    return pcap_file

def generate_sequence(pcap_file, prompt, remote_port):
    """Generate the sequence from network capture."""
    PrintUtils.start_stage('Generating sequence from capture')
    base_seq_path = pcap_file.replace('.pcap', '.seq')
    datapoint = Datapoint(pcap_file, base_seq_path)
    # Assume local_port not known prior: set to detect automatically
    datapoint.generate_seq(0, remote_port, prompt, response="N/A", temperature=0.0)
    PrintUtils.print_extra(f'Sequence saved to {base_seq_path}.')
    PrintUtils.end_stage()
    return datapoint

def prepare_data_for_model(datapoint, norm_params_path):
    """Prepare and normalize data for the model."""
    PrintUtils.start_stage('Preparing data for model')
    time_norm_params, size_norm_params, max_len = BaseClassifier.load_normalization_params(norm_params_path)
    df = pd.DataFrame([datapoint.seq])

    df_normalized = normalize_dataframe(df, time_norm_params, size_norm_params, max_len)
    dataset = PreprocessedTextDataset(df_normalized, max_len)
    PrintUtils.end_stage()
    return dataset

def inference_model(model_path, norm_params_path, dataset, device):
    """Run inference on a single model."""
    model = BaseClassifier.load(model_path, 3, norm_params_path[2])
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions, scores = [], []

    with torch.no_grad():
        for X, _ in loader:
            X = X.to(device)
            outputs = model(X)
            pred_proba = outputs.item()
            predictions.append(int(pred_proba > 0.5))
            scores.append(pred_proba)

    return predictions[0], scores[0]

def main():
    try:
        OsUtils.suppress_stderr()
        PrintUtils.print_logo()

        args = parse_arguments()
        prompts = load_input_prompts(args)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        PrintUtils.print_extra(f'Using device: {device}')

        pcap_file = perform_capture(args)
        
        results = []
        for prompt in tqdm(prompts, desc="Running prompts"):
            datapoint = generate_sequence(pcap_file, prompt, args.tlsport)

            prompt_results = {'prompt': prompt, 'models': []}
            for model_path, norm_param_path in zip(args.models, args.norm_params):
                dataset = prepare_data_for_model(datapoint, norm_param_path)
                pred, score = inference_model(model_path, norm_param_path, dataset, device)
                PrintUtils.print_extra(f"Prompt: {prompt} | Model: {os.path.basename(model_path)} | Prediction: {pred} | Score: {score:.4f}")

                prompt_results['models'].append({
                    'model': os.path.basename(model_path),
                    'prediction': pred,
                    'score': score
                })
            results.append(prompt_results)

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f_out:
            json.dump(results, f_out, indent=2)

        PrintUtils.print_extra(f"Inference completed. Results saved to {args.output}")

    except KeyboardInterrupt:
        PrintUtils.print_extra('Operation cancelled by user.')

    except Exception as e:
        PrintUtils.print_error(f'Error during inference: {e}')

    finally:
        NetworkUtils.stop_sniffing_tls(best_effort=True)
        PrintUtils.print_extra('Finished inference run.')

if __name__ == '__main__':
    main()

