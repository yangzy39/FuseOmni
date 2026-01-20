#!/bin/bash
# ==============================================================================
# Download Audio Understanding Datasets
# ==============================================================================
# Downloads audio understanding benchmarks:
# - MMAU (Massive Multi-task Audio Understanding)
# - MMAU-Pro
# - MMSU (Speech Understanding)
# - AIR-Bench
#
# Usage:
#   ./download_audio_understanding.sh
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$EVAL_DIR")"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/data}"

echo "=============================================="
echo "Downloading Audio Understanding Datasets"
echo "=============================================="

cd "$PROJECT_DIR"
python -m eval.download_datasets \
    --output-dir "$OUTPUT_DIR" \
    --datasets \
        mmau \
        mmau_pro \
        mmsu \
        airbench \
    "$@"

echo ""
echo "Audio understanding datasets downloaded to: $OUTPUT_DIR"
