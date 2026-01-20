#!/bin/bash
# ==============================================================================
# Run Single Dataset Evaluation
# ==============================================================================
# Runs evaluation on a single dataset with specified model.
#
# Usage:
#   ./run_eval.sh --dataset librispeech_clean --model Qwen/Qwen3-Omni-30B-A3B-Instruct
#   ./run_eval.sh --dataset mmau --model Qwen/Qwen3-Omni-30B-A3B-Instruct --limit 100
#
# Arguments:
#   --dataset      Dataset name (required)
#   --model        Model path or HuggingFace ID (required)
#   --data-dir     Local data directory (optional)
#   --output-dir   Output directory for results (default: ./outputs)
#   --limit        Limit number of samples (optional)
#   --batch-size   Batch size (default: 8)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$EVAL_DIR")"

# Default values
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"
BATCH_SIZE=8

# Parse arguments
DATASET=""
MODEL=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset|-d)
            DATASET="$2"
            shift 2
            ;;
        --model|-m)
            MODEL="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size|-b)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$DATASET" ]]; then
    echo "Error: --dataset is required"
    echo "Usage: ./run_eval.sh --dataset <name> --model <path>"
    exit 1
fi

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required"
    echo "Usage: ./run_eval.sh --dataset <name> --model <path>"
    exit 1
fi

echo "=============================================="
echo "Running Evaluation"
echo "=============================================="
echo "Dataset:    $DATASET"
echo "Model:      $MODEL"
echo "Data dir:   $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "=============================================="
echo ""

cd "$PROJECT_DIR"
python -m eval.eval \
    --dataset "$DATASET" \
    --model-path "$MODEL" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    $EXTRA_ARGS

echo ""
echo "Evaluation complete! Results saved to: $OUTPUT_DIR"
