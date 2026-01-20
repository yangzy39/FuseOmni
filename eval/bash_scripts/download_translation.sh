#!/bin/bash
# ==============================================================================
# Download Speech Translation Datasets
# ==============================================================================
# Downloads speech translation benchmarks:
# - CoVoST2 (en-zh, zh-en)
# - FLEURS (en, zh)
#
# Note: CoVoST2 requires HuggingFace authentication
#
# Usage:
#   ./download_translation.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$EVAL_DIR")"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/data}"

echo "=============================================="
echo "Downloading Speech Translation Datasets"
echo "=============================================="
echo "Note: CoVoST2 requires HuggingFace authentication."
echo "Run 'huggingface-cli login' if you haven't already."
echo ""

cd "$PROJECT_DIR"
python -m eval.download_datasets \
    --output-dir "$OUTPUT_DIR" \
    --datasets \
        covost2_en_zh \
        covost2_zh_en \
        fleurs_en \
        fleurs_zh \
    "$@"

echo ""
echo "Speech translation datasets downloaded to: $OUTPUT_DIR"
