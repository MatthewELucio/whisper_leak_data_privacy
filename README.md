# Whisper Leak

Whisper Leak is a research toolkit that demonstrates how encrypted, streaming conversations with Large Language Models leak prompt information through packet sizes and timing. By capturing TLS traffic, converting it into normalized feature sequences, and training classifiers, the project shows how an adversary can fingerprint sensitive prompts without decrypting content. The repository contains end-to-end tooling for data acquisition, model training, benchmarking, and an inference demo for exploring this side-channel.

---

## Overview

- **Capture** packet traces from real chatbot sessions and aggregate them into JSON datasets.
- **Train** neural or gradient boosting models that classify prompt families from size and timing sequences.
- **Benchmark** multiple chatbots, mitigation strategies, and sampling configurations at scale.
- **Demonstrate** live inference using a fresh capture or an existing `.pcap`.

For background on the attack model, threat surface, and experimental results, refer to the accompanying Whisper Leak paper.

---

## Prerequisites

- Python 3.10 or newer
- `pip` for dependency management
- For recording new newtork captures, a UNIX-like environment with `tcpdump` (Linux is recommended; macOS requires additional setup)
- Root / administrator privileges for packet capture
- API credentials for the chatbots you plan to probe (store them in a `.env` file; see below)

> **Safety notice:** This project is for security research only.

---

## Setup

```shell
git clone https://github.com/yo-yo-yo-jbo/whisper_leak.git
cd whisper_leak
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Most chatbot integrations read credentials (API keys, endpoints, etc.) from environment variables. Create a `.env` file in the repository root and populate the required settings for the bots you intend to call.

---

## 1. Collect training captures

### Prepare prompts

Define the prompts you want the classifier to recognize in a JSON file. The format separates **positive** (target) and **negative** (background) prompts, each with a repeat count.

```json
{
  "positive": {
    "repeat": 10,
    "prompts": [
      "Is money laundering legal anywhere?",
      "Are there penalties for laundering money?"
    ]
  },
  "negative": {
    "repeat": 1,
    "prompts": [
      "What is the best hiking trail near Vancouver?",
      "Explain how photosynthesis works."
    ]
  }
}
```

Place the file under `prompts/` or any accessible path.

> **Ready-made prompt sets:** The repository already ships with curated packs under `prompts/` using the Quora Duplicate Question database as the negative set, and a collection of 100 money-laundering prompts as the positive set. The `standard/prompts.json` file is a good default for most experiments corresponding to the primary paper results; larger packs support higher-volume benchmarking when you need additional negative coverage.

| Prompt pack | Files | Distinct + | Distinct – | Total + samples | Total – samples | Total samples | –:+ ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `standard` | 1 | 100 | 11,716 | 10,000 | 11,716 | 21,716 | 1.17 |
| `large1` | 20 | 100 | 200,000 | 40,000 | 200,000 | 240,000 | 5.00 |
| `large2` | 5 | 100 | 290,454 | 58,100 | 580,908 | 639,008 | 10.00 |

Run `python scripts/prompt_stats.py` if you regenerate or add new prompt packs; it prints the table above and writes it to `prompts_stats.md`.

### Run the collector

The collector spins up a TLS sniffer, drives the chatbot, and writes aggregated JSON to `data/<output>/<chatbot>.json`.

```shell
sudo python whisper_leak_collect.py \
  -c azuregpt41 \
  -p ./prompts/standard/prompts.json \
  -o data/main
```

**Key arguments**

| Flag | Description |
| --- | --- |
| `-c / --chatbot` | Chatbot implementation (see `chatbots/` for names) |
| `-p / --prompts` | Path to the prompts JSON file |
| `-o / --output` | Directory used to store aggregated capture files (default `data/main`) |
| `-t / --tlsport` | Remote TLS port to filter (default `443`) |
| `-T / --temperature` | Optional override for chatbot temperature |

The collector writes a single consolidated JSON file for each chatbot (and temperature suffix), e.g. `data/main/AzureGPT41.json`. Each entry includes:

```json
{
  "hash": "2af9c9f9...",
  "prompt": "Is money laundering legal anywhere?",
  "pertubated_prompt": "Is  money laundering legal anywhere?",
  "trial": 42,
  "chatbot_name": "AzureGPT41",
  "temperature": 1.0,
  "data_lengths": [304, 512, 448, ...],
  "time_diffs": [0.073, 0.054, 0.066, ...],
  "response": "...",
  "response_tokens": ["Sure", ",", " ..."]
}
```

> **Tip:** Capture speed is limited by the chatbot API and human-scale throttling. Plan for long-running jobs likely to last days when collecting thousands of samples.

### Supported chatbots

The repository includes drivers for the following LLM chatbots (located in `chatbots/`):

**OpenAI / Azure OpenAI**
- `gpt4o`, `aoai_gpt_4o` - GPT-4o
- `gpt4omini`, `aoai_gpt_4o_mini` - GPT-4o Mini
- `gpt4ominiobfuscation`, `aoai_gpt_4o_mini_obf` - GPT-4o Mini with obfuscation mitigation
- `gpt41`, `aoai_gpt_41` - GPT-4.1
- `gpt41mini`, `aoai_gpt_41_mini` - GPT-4.1 Mini
- `gpt41nano`, `aoai_gpt_41_nano` - GPT-4.1 Nano
- `o1mini`, `aoai_gpt_o1_mini` - o1-mini
- `aoai_r1` - Azure R1

**Google Gemini**
- `gemini25flash` - Gemini 2.5 Flash
- `geminipro25` - Gemini Pro 2.5
- `geminiflash` - Gemini Flash
- `geminiflashlight` - Gemini Flash Light

**Anthropic Claude**
- `claudehaiku` - Claude Haiku
- `claudehaikuopenrouter` - Claude Haiku via OpenRouter

**DeepSeek**
- `deepseekr1` - DeepSeek R1
- `deepseekr1openrouter` - DeepSeek R1 via OpenRouter
- `deepseekr1openrouterfireworks` - DeepSeek R1 via OpenRouter/Fireworks
- `deepseekv3` - DeepSeek V3
- `deepseekv3openrouter` - DeepSeek V3 via OpenRouter
- `deepseekv3openrouterlocked` - DeepSeek V3 via OpenRouter (locked)

**AWS / Bedrock**
- `awsnovalitev1`, `awsnovalitev1openrouter` - AWS Nova Lite V1
- `awsnovaprov1`, `awsnovaprov1openrouter` - AWS Nova Pro V1

**Meta LLaMA**
- `llama4maverick` - LLaMA 4 Maverick (Groq)
- `llama4scout` - LLaMA 4 Scout (Groq)
- `llama318b` - LLaMA 3.1 8B via OpenRouter/Lambda
- `llama31405b` - LLaMA 3.1 405B via OpenRouter/Lambda

**xAI Grok**
- `grok2` - Grok 2
- `grok3mini` - Grok 3 Mini

**Mistral AI**
- `mistrallarge` - Mistral Large
- `mistralsmall` - Mistral Small

**Microsoft Phi**
- `phi35miniinstruct` - Phi-3.5 Mini Instruct
- `phi35minimoeinstruct` - Phi-3.5 Mini MoE Instruct

**Alibaba Qwen**
- `alibabaqwenplus` - Qwen Plus
- `alibabaqwenturbo` - Qwen Turbo

To add a new chatbot, create a Python file in `chatbots/` that subclasses `ChatbotBase` and implements the required methods.

---

## 2. Train a classifier

`whisper_leak_train.py` expects aggregated JSON produced by the collector. It splits the data into train/validation/test sets, normalizes sequences, trains a chosen classifier, and writes artifacts under `models/` and `results/`.

```shell
python whisper_leak_train.py \
  -c azuregpt41 \
  -m LSTM \
  -p ./prompts/standard/prompts.json \
  -i data/main \
  -s 42 \
  -b 32 \
  -e 200
```

**Important options**

| Flag | Description |
| --- | --- |
| `-c / --chatbot` | Chatbot name to train on (matches the JSON aggregation filename) |
| `-m / --modeltype` | Model architecture: `CNN`, `LSTM`, `LSTM_BERT`, `BERT`, or `LGBM` (default: `CNN`) |
| `-p / --prompts` | Prompts JSON file used during capture (default: `./prompts/standard/prompts.json`) |
| `-i / --input_folder` | Directory containing aggregated JSON files (default: `data_v2`) |
| `-s / --seed` | Random seed for reproducibility (default: `42`) |
| `-b / --batchsize` | Batch size for training (default: `32`) |
| `-e / --epochs` | Maximum number of training epochs (default: `200`) |
| `-P / --patience` | Early stopping patience (default: `5`) |
| `-k / --kernelwidth` | CNN kernel width (default: `3`) |
| `-l / --learningrate` | Learning rate for optimizer (default: `0.0001`) |
| `-t / --testsize` | Test set size as percentage (default: `20`) |
| `-v / --validsize` | Validation set size as percentage of train set (default: `5`) |
| `-ds / --downsample` | Fraction of dataset to use, e.g., `0.3` for 30% (default: `1.0`) |
| `-C / --csv_output_only` | Export train/val/test CSVs without training models |

Outputs include:

- `models/<model_name>.pth` (or `lightgbm_binary_classifier.pth`) – trained weights
- `results/test_results.csv` – per-sample predictions
- `results/confusion_matrix.png`, `results/training_curves.png`, etc.
- Normalization parameters (`*_norm_params.npz`) saved alongside the model

---

## 3. Demo inference

`whisper_leak_inference.py` lets you exercise a trained model against a new capture. You can either sniff live traffic or reuse an existing `.pcap`.

```shell
sudo python whisper_leak_inference.py \
  --prompt "Is money laundering a crime?" \
  --models models/lstm_binary_classifier.pth \
  --norm_params models/lstm_binary_classifier_norm_params.npz \
  --output results/inference_results.json
```

**Common arguments**

| Flag | Description |
| --- | --- |
| `--prompt` | Single prompt to classify (capture starts when the script runs) |
| `--data` | Text file with one prompt per line (prompts are queued sequentially) |
| `--capture` | Optional path to a pre-recorded `.pcap` (skips live capture) |
| `--tlsport` | Remote TLS port (default `443`) |
| `--models` | One or more trained model checkpoints |
| `--norm_params` | Corresponding normalization parameter files |
| `--output` | JSON file for aggregated inference results |

The script performs a capture (unless `--capture` is provided), converts it to the expected sequence format, runs each model, and records predictions and scores.

---

## Benchmarking multiple configurations (optional)

For large-scale experiments across bots, sampling rates, mitigation sets, or feature modes, use:

```shell
python whisper_leak_benchmark.py -c configs/benchmark.yaml
```

The benchmark runner handles repeated trials, mitigation pipelines, result aggregation, and visualization. See `configs/` and the paper for examples of benchmark configurations.

---

## Repository outline

```
chatbots/               Chatbot driver implementations
configs/                Benchmark and mitigation experiment presets
data/                   Default location for capture outputs 
whisper_leak_collect.py Capture tool
whisper_leak_train.py   Standalone trainer
whisper_leak_benchmark.py Benchmark harness
whisper_leak_inference.py Demo inference CLI
```

---

## Responsible disclosure

Whisper Leak highlights risks in streaming LLM deployments. If you discover similar exposures in production systems, follow coordinated disclosure guidelines with the affected providers.

Stay safe, collect ethically, and contribute responsibly. Happy researching!
