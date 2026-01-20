#!/bin/bash
# ==============================================================================
# Download ASR Datasets
# ==============================================================================
# Downloads ASR (Automatic Speech Recognition) datasets:
# - LibriSpeech (clean & other)
# - Common Voice (English & Chinese) 
# - AISHELL-1
# - GigaSpeech
# - WenetSpeech
#
# Usage:
#   ./download_asr.sh                     # Download all ASR datasets
#   ./download_asr.sh --limit 100         # Download with sample limit
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$EVAL_DIR")"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/data}"

echo "=============================================="
echo "Downloading ASR Datasets"
echo "=============================================="

cd "$PROJECT_DIR"
python -m eval.download_datasets \
    --output-dir "$OUTPUT_DIR" \
    --datasets \
        librispeech_clean \
        librispeech_other \
        common_voice_en \
        common_voice_zh \
        aishell1 \
        gigaspeech \
        wenetspeech \
    "$@"

echo ""
echo "ASR datasets downloaded to: $OUTPUT_DIR"
