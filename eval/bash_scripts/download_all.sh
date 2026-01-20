#!/bin/bash
# ==============================================================================
# Download All Datasets
# ==============================================================================
# Downloads all evaluation datasets to local storage.
# 
# Usage:
#   ./download_all.sh                    # Download all datasets
#   ./download_all.sh --limit 100        # Download with sample limit (for testing)
#   ./download_all.sh --force            # Force re-download
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$EVAL_DIR")"

# Default output directory
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/data}"

# Parse arguments
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

echo "=============================================="
echo "Downloading All Evaluation Datasets"
echo "=============================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run download script
cd "$PROJECT_DIR"
python -m eval.download_datasets \
    --output-dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

echo ""
echo "=============================================="
echo "Download complete!"
echo "Datasets saved to: $OUTPUT_DIR"
echo "=============================================="
