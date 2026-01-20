#!/bin/bash

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Determine Project Root (Assuming script is in datasets/ or datasets/scripts/)
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
    PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
elif [[ "$(basename "$SCRIPT_DIR")" == "datasets" ]]; then
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi

# Define paths
DATASETS_SCRIPT="$PROJECT_ROOT/datasets/scripts/download_datasets.py"
CONVERT_SCRIPT="$PROJECT_ROOT/datasets/scripts/convert_utils.py"
OUTPUT_BASE="$PROJECT_ROOT/datasets/data"
CALIBRATION_DIR="$OUTPUT_BASE/calibration"
MSSWIFT_DIR="$OUTPUT_BASE/ms_swift"

# Configuration
HF_TOKEN="xx"  # Your HuggingFace Token
SAMPLES=100  # Adjust sample count as needed

# 1. Environment Setup
export HF_ENDPOINT=https://hf-mirror.com
# export HF_TOKEN="$HF_TOKEN" # Optional: Export as env var as well
echo ">> [Env] Set HF_ENDPOINT=$HF_ENDPOINT"

echo ">> [Env] Installing dependencies..."
pip install datasets soundfile tqdm pandas huggingface_hub -q

# 2. Download Datasets
echo ""
echo ">> [1/2] Downloading all datasets (Samples: $SAMPLES)..."
if [ ! -f "$DATASETS_SCRIPT" ]; then
    echo "Error: Could not find download_datasets.py at $DATASETS_SCRIPT"
    exit 1
fi

python "$DATASETS_SCRIPT" \
    --dataset all \
    --output "$OUTPUT_BASE" \
    --samples $SAMPLES \
    --token "$HF_TOKEN"

# 3. Convert to MS-SWIFT Format
echo ""
echo ">> [2/2] Converting to MS-SWIFT format..."
mkdir -p "$MSSWIFT_DIR"

for modality in "audio" "video" "mixed"; do
    INPUT_FILE="$CALIBRATION_DIR/$modality.jsonl"
    OUTPUT_FILE="$MSSWIFT_DIR/${modality}_sft.jsonl"
    
    if [ -f "$INPUT_FILE" ]; then
        echo "  Processing $modality data..."
        python "$CONVERT_SCRIPT" msswift \
            "$INPUT_FILE" \
            "$OUTPUT_FILE" \
            --task sft \
            --system "You are a helpful multimodal assistant."
            
        echo "  -> Saved to $OUTPUT_FILE"
    else
        echo "  ! Skipping $modality (File not found: $INPUT_FILE)"
    fi
done

echo ""
echo ">> All tasks completed."
echo ">> MS-SWIFT data available in: $MSSWIFT_DIR"
