#!/bin/bash
# ==============================================================================
# Run Full Evaluation Suite
# ==============================================================================
# Runs evaluation on ALL datasets with specified model.
#
# Usage:
#   ./run_full_eval.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct
#   ./run_full_eval.sh --model Qwen/Qwen3-Omni-30B-A3B-Instruct --limit 50
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
    echo "Usage: ./run_full_eval.sh --model <path>"
    exit 1
fi

# All datasets grouped by category
declare -A CATEGORIES
CATEGORIES["ASR"]="librispeech_clean librispeech_other aishell1"
CATEGORIES["Audio_Understanding"]="mmau mmau_pro mmsu airbench"
CATEGORIES["Spoken_QA"]="voicebench openaudiobench"
CATEGORIES["Translation"]="covost2_en_zh covost2_zh_en"
CATEGORIES["Emotion"]="meld"

echo "=============================================="
echo "Running Full Evaluation Suite"
echo "=============================================="
echo "Model: $MODEL"
echo "Data dir: $DATA_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "=============================================="

# Create results summary file
SUMMARY_FILE="$OUTPUT_DIR/summary.txt"
mkdir -p "$OUTPUT_DIR"
echo "Evaluation Summary" > "$SUMMARY_FILE"
echo "Model: $MODEL" >> "$SUMMARY_FILE"
echo "Date: $(date)" >> "$SUMMARY_FILE"
echo "==========================================" >> "$SUMMARY_FILE"

for category in "${!CATEGORIES[@]}"; do
    echo ""
    echo "=============================================="
    echo "Category: $category"
    echo "=============================================="
    echo "" >> "$SUMMARY_FILE"
    echo "[$category]" >> "$SUMMARY_FILE"
    
    for dataset in ${CATEGORIES[$category]}; do
        echo ""
        echo ">>> Evaluating: $dataset"
        echo "-------------------------------------------"
        
        if python -m eval.eval \
            --dataset "$dataset" \
            --model-path "$MODEL" \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUTPUT_DIR/$dataset" \
            $EXTRA_ARGS; then
            echo "  $dataset: SUCCESS" >> "$SUMMARY_FILE"
        else
            echo "  $dataset: FAILED" >> "$SUMMARY_FILE"
            echo "Warning: $dataset evaluation failed"
        fi
    done
done

echo ""
echo "=============================================="
echo "Full Evaluation Complete!"
echo "=============================================="
echo "Results saved to: $OUTPUT_DIR"
echo "Summary: $SUMMARY_FILE"
echo "=============================================="
