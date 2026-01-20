#!/bin/bash
# ==============================================================================
# Download Spoken QA Datasets
# ==============================================================================
# Downloads spoken question answering benchmarks:
# - VoiceBench
# - OpenAudioBench
#
# Usage:
#   ./download_spoken_qa.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$EVAL_DIR")"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/data}"

echo "=============================================="
echo "Downloading Spoken QA Datasets"
echo "=============================================="

cd "$PROJECT_DIR"
python -m eval.download_datasets \
    --output-dir "$OUTPUT_DIR" \
    --datasets \
        voicebench \
        openaudiobench \
    "$@"

echo ""
echo "Spoken QA datasets downloaded to: $OUTPUT_DIR"
