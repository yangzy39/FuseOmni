#!/bin/bash
# ==============================================================================
# Run Audio Understanding Evaluation Suite
# ==============================================================================
# Runs evaluation on audio understanding benchmarks (MMAU, MMSU, AIR-Bench).
#
# Usage:
#   ./run_audio_understanding_eval.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct
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

DATASETS=(
    "mmau"
    "mmau_pro"
    "mmsu"
    "airbench"
)

echo "=============================================="
echo "Running Audio Understanding Evaluation"
echo "=============================================="
echo "Model: $MODEL"
echo "Datasets: ${DATASETS[*]}"
echo "=============================================="

for dataset in "${DATASETS[@]}"; do
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
echo "Audio Understanding Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=============================================="
