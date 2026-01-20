# Bash Scripts for Speech Model Evaluation

This directory contains executable bash scripts for downloading datasets and running evaluations.

## Quick Start

```bash
# 1. Download all datasets
./download_all.sh

# 2. Run evaluation on a specific dataset
./run_eval.sh --dataset librispeech_clean --model Qwen/Qwen3-Omni-30B-A3B-Instruct

# 3. Run full evaluation suite
./run_full_eval.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct
```

## Download Scripts

| Script | Description |
|--------|-------------|
| `download_all.sh` | Download all evaluation datasets |
| `download_asr.sh` | Download ASR datasets (LibriSpeech, AISHELL, etc.) |
| `download_audio_understanding.sh` | Download MMAU, MMSU, AIR-Bench |
| `download_spoken_qa.sh` | Download VoiceBench, OpenAudioBench |
| `download_translation.sh` | Download CoVoST2, FLEURS |
| `download_emotion.sh` | Download IEMOCAP, MELD |

### Download Usage

```bash
# Download all datasets to default directory (./data)
./download_all.sh

# Download to custom directory
./download_all.sh --output-dir /path/to/data

# Download with sample limit (for testing)
./download_all.sh --limit 100

# Force re-download
./download_all.sh --force

# Download specific category
./download_asr.sh
./download_audio_understanding.sh
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OUTPUT_DIR` | Data output directory | `./data` |

## Evaluation Scripts

| Script | Description |
|--------|-------------|
| `run_eval.sh` | Run evaluation on a single dataset |
| `run_asr_eval.sh` | Run all ASR evaluations |
| `run_audio_understanding_eval.sh` | Run audio understanding evaluations |
| `run_full_eval.sh` | Run complete evaluation suite |

### Evaluation Usage

```bash
# Single dataset evaluation
./run_eval.sh --dataset librispeech_clean --model Qwen/Qwen3-Omni-30B-A3B-Instruct

# With options
./run_eval.sh \
    --dataset mmau \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --data-dir ./data \
    --output-dir ./outputs \
    --batch-size 8 \
    --limit 100

# Run all ASR evaluations
./run_asr_eval.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct

# Run full evaluation suite
./run_full_eval.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATA_DIR` | Local data directory | `./data` |
| `OUTPUT_DIR` | Results output directory | `./outputs` |

## Script Arguments

### Common Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--model` | `-m` | Model path or HuggingFace ID (required for eval) |
| `--dataset` | `-d` | Dataset name (for single eval) |
| `--data-dir` | | Local data directory |
| `--output-dir` | `-o` | Output directory |
| `--limit` | `-l` | Limit number of samples |
| `--batch-size` | `-b` | Batch size for inference |
| `--force` | `-f` | Force re-download (for download scripts) |

## Available Datasets

### ASR (Automatic Speech Recognition)
- `librispeech_clean` - LibriSpeech test-clean
- `librispeech_other` - LibriSpeech test-other
- `common_voice_en` - Common Voice English [AUTH]
- `common_voice_zh` - Common Voice Chinese [AUTH]
- `aishell1` - AISHELL-1 Mandarin
- `gigaspeech` - GigaSpeech [AUTH]
- `wenetspeech` - WenetSpeech

### Audio Understanding
- `mmau` - MMAU benchmark
- `mmau_pro` - MMAU-Pro advanced
- `mmsu` - MMSU speech understanding
- `airbench` - AIR-Bench

### Spoken QA
- `voicebench` - VoiceBench
- `openaudiobench` - OpenAudioBench

### Speech Translation
- `covost2_en_zh` - CoVoST2 En→Zh [AUTH]
- `covost2_zh_en` - CoVoST2 Zh→En [AUTH]
- `fleurs_en` - FLEURS English
- `fleurs_zh` - FLEURS Chinese

### Emotion Recognition
- `iemocap` - IEMOCAP
- `meld` - MELD

**[AUTH]** = Requires HuggingFace authentication. Run `huggingface-cli login` first.

## Examples

### Example 1: Quick Test Run

```bash
# Download a small subset for testing
./download_asr.sh --limit 10

# Run quick evaluation
./run_eval.sh \
    --dataset librispeech_clean \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --limit 10
```

### Example 2: Full ASR Benchmark

```bash
# Download all ASR datasets
./download_asr.sh

# Run all ASR evaluations
./run_asr_eval.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct
```

### Example 3: Complete Evaluation

```bash
# Download everything
./download_all.sh

# Run full evaluation suite
./run_full_eval.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct

# Results will be in ./outputs/summary.txt
```

## Troubleshooting

### Permission Denied
```bash
chmod +x *.sh
```

### HuggingFace Authentication
```bash
pip install huggingface_hub
huggingface-cli login
```

### Missing Dependencies
```bash
pip install -r ../requirements.txt
```

### Dataset Download Fails
Some datasets require authentication or have usage restrictions:
- Common Voice: Agree to terms on HuggingFace
- CoVoST2: Agree to terms on HuggingFace
- IEMOCAP: May require manual download
