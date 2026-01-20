#!/bin/bash
# ==============================================================================
# Run ASR Evaluation Suite
# ==============================================================================
# Runs evaluation on all ASR datasets.
#
# Usage:
#   ./run_asr_eval.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct
#   ./run_asr_eval.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct --limit 100
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_DIR="$(dirname "$EVAL_DIR")"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/data}"

MODEL=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m)
            MODEL="$2"
            shift 2
            ;;
        --output-dir|-o)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "Error: --model is required"
    exit 1
fi

ASR_DATASETS=(
    "librispeech_clean"
    "librispeech_other"
    "aishell1"
)

echo "=============================================="
echo "Running ASR Evaluation Suite"
echo "=============================================="
echo "Model: $MODEL"
echo "Datasets: ${ASR_DATASETS[*]}"
echo "=============================================="

for dataset in "${ASR_DATASETS[@]}"; do
    echo ""
    echo ">>> Evaluating: $dataset"
    echo "-------------------------------------------"
    
    python -m eval.eval \
        --dataset "$dataset" \
        --model-path "$MODEL" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR/$dataset" \
        $EXTRA_ARGS || echo "Warning: $dataset evaluation failed"
done

echo ""
echo "=============================================="
echo "ASR Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
