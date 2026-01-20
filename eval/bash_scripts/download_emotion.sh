#!/bin/bash
# ==============================================================================
# Download Emotion Recognition Datasets
# ==============================================================================
# Downloads emotion recognition benchmarks:
# - IEMOCAP (may require manual download)
# - MELD
#
# Usage:
#   ./download_emotion.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$EVAL_DIR")"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/data}"

echo "=============================================="
echo "Downloading Emotion Recognition Datasets"
echo "=============================================="
echo "Note: IEMOCAP may require manual download due to licensing."
echo ""

cd "$PROJECT_DIR"
python -m eval.download_datasets \
    --output-dir "$OUTPUT_DIR" \
    --datasets \
        iemocap \
        meld \
    "$@"

echo ""
echo "Emotion datasets downloaded to: $OUTPUT_DIR"
