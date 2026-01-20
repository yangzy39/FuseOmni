# Speech Model Evaluation Framework

A comprehensive evaluation framework for speech/audio language models, supporting multiple benchmarks with [vLLM-Omni](https://github.com/vllm-project/vllm-omni) as the inference engine.

## Overview

This framework provides:
- **Unified CLI** for evaluating speech models across diverse benchmarks
- **20+ datasets** covering ASR, Audio Understanding, Spoken QA, Translation, Emotion
- **Standard metrics** (WER, CER, BLEU, Accuracy, F1)
- **vLLM-Omni integration** for efficient Qwen-Omni inference
- **Local dataset support** with unified download scripts
- **Bash scripts** for automated download and evaluation

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets (to ./data)
cd bash_scripts
./download_all.sh

# 3. Run evaluation
./run_eval.sh --dataset librispeech_clean --model Qwen/Qwen3-Omni-30B-A3B-Instruct
```

## Supported Models

| Model | Architecture | Recommended Usage |
|-------|--------------|--------------------|
| **Qwen3-Omni** | Thinker-Talker MoE + AuT encoder | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |
| **Qwen2.5-Omni** | Qwen2.5 backbone + audio encoder | `Qwen/Qwen2.5-Omni-7B-Instruct` |
| **Qwen2-Audio** | Whisper-large-v3 + Qwen-7B | `Qwen/Qwen2-Audio-7B-Instruct` |

## Dataset Download

### Using Bash Scripts (Recommended)

```bash
cd eval/bash_scripts

# Download all datasets
./download_all.sh

# Download by category
./download_asr.sh                    # ASR datasets
./download_audio_understanding.sh    # MMAU, MMSU, AIR-Bench
./download_spoken_qa.sh              # VoiceBench, OpenAudioBench
./download_translation.sh            # CoVoST2, FLEURS
./download_emotion.sh                # IEMOCAP, MELD

# Download with options
./download_all.sh --output-dir /path/to/data --limit 100
```

### Using Python Script

```bash
# Download all datasets
python -m eval.download_datasets --output-dir ./data

# Download specific datasets
python -m eval.download_datasets --output-dir ./data --datasets librispeech_clean mmau

# List available datasets
python -m eval.download_datasets --list

# Download with sample limit (for testing)
python -m eval.download_datasets --output-dir ./data --limit 100
```

### HuggingFace Authentication

Some datasets require authentication:
```bash
pip install huggingface_hub
huggingface-cli login
```

Required for: Common Voice, CoVoST2, GigaSpeech

## Running Evaluations

### Using Bash Scripts (Recommended)

```bash
cd eval/bash_scripts

# Single dataset evaluation
./run_eval.sh --dataset librispeech_clean --model Qwen/Qwen3-Omni-30B-A3B-Instruct

# With options
./run_eval.sh \
    --dataset mmau \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --data-dir ./data \
    --output-dir ./outputs \
    --batch-size 8

# Run all ASR evaluations
./run_asr_eval.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct

# Run audio understanding evaluations
./run_audio_understanding_eval.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct

# Run full evaluation suite
./run_full_eval.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct
```

### Using Python CLI

```bash
# Basic evaluation
python -m eval.eval \
    --dataset librispeech_clean \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct

# With local data directory
python -m eval.eval \
    --dataset librispeech_clean \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --data-dir ./data

# List available datasets
python -m eval.eval --list-datasets

# Quick test with limited samples
python -m eval.eval \
    --dataset mmau \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --limit 10
```

## Benchmark Description Table

### ASR (Automatic Speech Recognition)

| Dataset | Language | Metrics | Description |
|---------|----------|---------|-------------|
| `librispeech_clean` | EN | WER, CER | Clean read English speech (~2.6k samples) |
| `librispeech_other` | EN | WER, CER | Challenging read speech (~2.9k samples) |
| `common_voice_en` | EN | WER, CER | Crowdsourced English [AUTH] |
| `common_voice_zh` | ZH | CER | Crowdsourced Chinese [AUTH] |
| `aishell1` | ZH | CER | Mandarin read speech (178 hours) |
| `gigaspeech` | EN | WER, CER | Large-scale diverse English [AUTH] |
| `wenetspeech` | ZH | CER | Multi-domain Mandarin (10k+ hours) |

### Audio Understanding

| Dataset | Task | Metrics | Description |
|---------|------|---------|-------------|
| `mmau` | MCQ | Accuracy | Speech, Music, Sound understanding |
| `mmau_pro` | MCQ | Accuracy | Advanced scenarios (long-form, spatial) |
| `mmsu` | MCQ | Accuracy | Linguistic nuances (intonation, prosody) |
| `airbench` | QA | Accuracy | Audio instruction recognition |

### Spoken Question Answering

| Dataset | Task | Metrics | Description |
|---------|------|---------|-------------|
| `voicebench` | QA | Accuracy | Comprehensive spoken QA |
| `openaudiobench` | QA | Accuracy | AlpacaEval, TriviaQA audio |

### Speech Translation

| Dataset | Direction | Metrics | Description |
|---------|-----------|---------|-------------|
| `covost2_en_zh` | EN→ZH | BLEU, chrF | English to Chinese [AUTH] |
| `covost2_zh_en` | ZH→EN | BLEU, chrF | Chinese to English [AUTH] |
| `fleurs_en` | - | - | FLEURS English source |
| `fleurs_zh` | - | - | FLEURS Chinese target |

### Emotion Recognition

| Dataset | Classes | Metrics | Description |
|---------|---------|---------|-------------|
| `iemocap` | 9 emotions | Accuracy, Macro-F1 | Interactive emotional speech |
| `meld` | 7 emotions | Accuracy, Weighted-F1 | Friends TV series |

**[AUTH]** = Requires HuggingFace authentication

## Project Structure

```
eval/
├── __init__.py
├── schema.py              # Core data structures
├── registry.py            # Dataset/metric registration
├── eval.py                # Main CLI entrypoint
├── download_datasets.py   # Dataset download script
├── requirements.txt
├── README.md
│
├── bash_scripts/          # Executable scripts
│   ├── README.md
│   ├── download_all.sh
│   ├── download_asr.sh
│   ├── download_audio_understanding.sh
│   ├── download_spoken_qa.sh
│   ├── download_translation.sh
│   ├── download_emotion.sh
│   ├── run_eval.sh
│   ├── run_asr_eval.sh
│   ├── run_audio_understanding_eval.sh
│   └── run_full_eval.sh
│
├── engine/
│   └── vllm_omni.py       # vLLM-Omni inference wrapper
│
├── datasets/
│   ├── base.py            # BaseDataset class
│   ├── common_types.py    # Normalization utilities
│   ├── asr/               # ASR datasets
│   ├── audio_understanding/
│   ├── spoken_qa/
│   ├── speech_translation/
│   └── emotion/
│
├── metrics/
│   ├── base.py            # BaseMetric class
│   ├── wer.py             # WER, CER
│   ├── accuracy.py        # Accuracy, ExactMatch
│   ├── bleu.py            # BLEU, chrF
│   ├── f1.py              # QA F1, EM
│   └── classification.py  # Macro-F1, Weighted-F1
│
├── io/
│   ├── jsonl.py           # JSONL read/write
│   └── cache.py           # Prediction caching
│
└── prompts/
    ├── templates.py       # Prompt templates
    └── chat.py            # Chat formatting
```

## CLI Reference

### Dataset Options

| Argument | Description |
|----------|-------------|
| `--dataset, -d` | Dataset name (required) |
| `--list-datasets` | List all available datasets |
| `--split` | Dataset split (default: test) |
| `--limit, -n` | Max samples to evaluate |
| `--data-dir` | Local data directory |

### Model Options

| Argument | Description |
|----------|-------------|
| `--model-path, -m` | Model path or HuggingFace ID (required) |
| `--tensor-parallel-size, -tp` | Tensor parallelism (default: 1) |
| `--stage-configs-path` | vLLM-Omni stage configs YAML |
| `--dtype` | Model dtype (auto/float16/bfloat16) |

### Sampling Parameters

| Argument | Description |
|----------|-------------|
| `--temperature, -t` | Sampling temperature (default: 0.0) |
| `--top-p` | Top-p sampling (default: 1.0) |
| `--max-tokens` | Max generation tokens (default: 512) |
| `--seed` | Random seed (default: 42) |

### Runtime Options

| Argument | Description |
|----------|-------------|
| `--batch-size, -b` | Inference batch size (default: 1) |
| `--output-dir, -o` | Output directory (default: outputs) |
| `--resume` | Resume from checkpoint |

## Output Structure

```
outputs/
└── librispeech_clean_20250120_123456/
    ├── config.json        # Run configuration
    ├── predictions.jsonl  # Per-sample predictions
    └── metrics.json       # Aggregated metrics
```

## vLLM-Omni Integration

This framework uses [vLLM-Omni](https://github.com/vllm-project/vllm-omni) for inference:

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(
    model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
    stage_configs_path=None,
)

# Audio placeholder format
prompt = """<|im_start|>system
You are Qwen...<|im_end|>
<|im_start|>user
<|audio_start|><|audio_pad|><|audio_end|>Transcribe this audio.<|im_end|>
<|im_start|>assistant
"""
```

## Adding New Datasets

1. Create a new file in `eval/datasets/<category>/`:

```python
from ..base import BaseDataset
from ...schema import EvalSample
from ...registry import register_dataset

@register_dataset("my_dataset")
class MyDataset(BaseDataset):
    name = "my_dataset"
    task_type = "asr"  # asr, mcq, qa, translation, emotion
    metrics = ["wer"]
    
    def load_from_hf(self):
        # Load from HuggingFace
        from datasets import load_dataset
        dataset = load_dataset("my/dataset", split=self.split)
        
        for idx, sample in enumerate(dataset):
            yield EvalSample(
                id=f"my_dataset_{idx}",
                audio_path=sample["audio"]["path"],
                text_prompt="Transcribe the audio.",
                reference=sample["text"],
                meta={"audio_array": sample["audio"]["array"]},
            )
    
    def get_reference(self, sample):
        return sample["text"]
```

2. Import in `eval/datasets/<category>/__init__.py`

3. Add to download registry in `eval/download_datasets.py`

## License

This project is for research and evaluation purposes.

## Citation

```bibtex
@misc{speecheval2025,
  title={Speech Model Evaluation Framework},
  year={2025},
  url={https://github.com/your-org/eval}
}
```
