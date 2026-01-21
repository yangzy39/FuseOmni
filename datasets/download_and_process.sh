#!/bin/bash
# =============================================================================
# Audio Dataset Download and Processing Script for Speech SFT
# =============================================================================
#
# This script automates the workflow of:
# 1. Downloading audio datasets from HuggingFace
# 2. Converting them to MS-SWIFT SFT format
# 3. Organizing the output for training
#
# Environment Variables (set before running):
#   HF_ENDPOINT   - HuggingFace mirror (default: https://hf-mirror.com)
#   HF_TOKEN      - Your HuggingFace authentication token
#
# Usage:
#   # Basic usage (downloads audio datasets with 100 samples each)
#   ./download_and_process.sh
#
#   # Custom samples and output directory
#   ./download_and_process.sh --samples 500 --output ./my_data
#
#   # Download specific modality
#   ./download_and_process.sh --modality video --samples 200
#
#   # Skip download, only convert existing data
#   ./download_and_process.sh --convert-only
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Get the absolute path of the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Determine Project Root
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
    PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
elif [[ "$(basename "$SCRIPT_DIR")" == "datasets" ]]; then
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi

# Define paths
SCRIPTS_DIR="$PROJECT_ROOT/datasets/scripts"
OUTPUT_BASE="$PROJECT_ROOT/datasets/data"
CALIBRATION_DIR="$OUTPUT_BASE/calibration"
MSSWIFT_DIR="$OUTPUT_BASE/ms_swift"

# Default configuration
HF_TOKEN="${HF_TOKEN:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
SAMPLES=100
MODALITY="audio"
CONVERT_ONLY=false
SYSTEM_PROMPT="You are a helpful multimodal assistant."

# =============================================================================
# Argument Parsing
# =============================================================================

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --samples N        Number of samples per dataset (default: 100)"
    echo "  --output DIR       Output directory (default: ./datasets/data)"
    echo "  --modality TYPE    Modality to download: audio|video|mixed|all (default: audio)"
    echo "  --token TOKEN      HuggingFace token (or set HF_TOKEN env var)"
    echo "  --convert-only     Skip download, only convert existing data"
    echo "  --system PROMPT    System prompt for MS-SWIFT format"
    echo "  --help             Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  HF_ENDPOINT        HuggingFace mirror URL (default: https://hf-mirror.com)"
    echo "  HF_TOKEN           HuggingFace authentication token"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --samples)
            SAMPLES="$2"
            shift 2
            ;;
        --output)
            OUTPUT_BASE="$2"
            CALIBRATION_DIR="$OUTPUT_BASE/calibration"
            MSSWIFT_DIR="$OUTPUT_BASE/ms_swift"
            shift 2
            ;;
        --modality)
            MODALITY="$2"
            shift 2
            ;;
        --token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --convert-only)
            CONVERT_ONLY=true
            shift
            ;;
        --system)
            SYSTEM_PROMPT="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# =============================================================================
# Environment Setup
# =============================================================================

echo ""
echo "============================================================"
echo "Audio Dataset Download and Processing"
echo "============================================================"
echo ""

# Set HuggingFace endpoint
export HF_ENDPOINT="$HF_ENDPOINT"
echo ">> [Env] HF_ENDPOINT=$HF_ENDPOINT"

if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
    echo ">> [Env] HF_TOKEN=****** (set)"
else
    echo ">> [Env] HF_TOKEN not set (some datasets may require authentication)"
fi

# Check Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.9+."
    exit 1
fi

echo ">> [Env] Python: $(python --version)"
echo ""

# Install dependencies
echo ">> [Env] Installing dependencies..."
pip install datasets soundfile tqdm pandas huggingface_hub -q
echo ">> [Env] Dependencies installed"
echo ""

# =============================================================================
# Download Datasets
# =============================================================================

if [ "$CONVERT_ONLY" = false ]; then
    echo "============================================================"
    echo "[1/2] Downloading Datasets"
    echo "============================================================"
    echo ""
    echo "  Modality: $MODALITY"
    echo "  Samples:  $SAMPLES"
    echo "  Output:   $OUTPUT_BASE"
    echo ""
    
    DOWNLOAD_SCRIPT="$SCRIPTS_DIR/download_datasets.py"
    
    if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
        echo "Error: Download script not found at $DOWNLOAD_SCRIPT"
        exit 1
    fi
    
    python "$DOWNLOAD_SCRIPT" \
        --output "$OUTPUT_BASE" \
        --modality "$MODALITY" \
        --samples "$SAMPLES" \
        ${HF_TOKEN:+--token "$HF_TOKEN"}
    
    echo ""
    echo ">> Download completed"
else
    echo ">> Skipping download (--convert-only specified)"
fi

# =============================================================================
# Convert to MS-SWIFT Format
# =============================================================================

echo ""
echo "============================================================"
echo "[2/2] Converting to MS-SWIFT Format"
echo "============================================================"
echo ""

CONVERT_SCRIPT="$SCRIPTS_DIR/convert_utils.py"

if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "Error: Convert script not found at $CONVERT_SCRIPT"
    exit 1
fi

mkdir -p "$MSSWIFT_DIR"

# Process each dataset directory
for dataset_dir in "$OUTPUT_BASE"/*/; do
    if [ -d "$dataset_dir" ]; then
        dataset_name=$(basename "$dataset_dir")
        
        # Skip special directories
        if [[ "$dataset_name" == "ms_swift" ]] || [[ "$dataset_name" == "calibration" ]]; then
            continue
        fi
        
        manifest_file="$dataset_dir/manifest.json"
        
        if [ -f "$manifest_file" ]; then
            output_file="$MSSWIFT_DIR/${dataset_name}_sft.jsonl"
            
            echo "  Processing $dataset_name..."
            
            python "$CONVERT_SCRIPT" manifest \
                "$manifest_file" \
                "$output_file" \
                --task sft \
                --system "$SYSTEM_PROMPT"
            
            echo "    -> Saved to $output_file"
        fi
    fi
done

# Also process any existing JSONL files in calibration directory
if [ -d "$CALIBRATION_DIR" ]; then
    for jsonl_file in "$CALIBRATION_DIR"/*.jsonl; do
        if [ -f "$jsonl_file" ]; then
            filename=$(basename "$jsonl_file" .jsonl)
            output_file="$MSSWIFT_DIR/${filename}_sft.jsonl"
            
            echo "  Processing $filename..."
            
            python "$CONVERT_SCRIPT" msswift \
                "$jsonl_file" \
                "$output_file" \
                --task sft \
                --system "$SYSTEM_PROMPT"
            
            echo "    -> Saved to $output_file"
        fi
    done
fi

# =============================================================================
# Merge All Data (Optional)
# =============================================================================

echo ""
echo ">> Merging all SFT data..."

# Count files
file_count=$(find "$MSSWIFT_DIR" -name "*_sft.jsonl" 2>/dev/null | wc -l)

if [ "$file_count" -gt 0 ]; then
    python "$CONVERT_SCRIPT" merge \
        "$MSSWIFT_DIR"/*_sft.jsonl \
        -o "$MSSWIFT_DIR/all_sft.jsonl" \
        --shuffle
    
    echo ">> Merged data saved to: $MSSWIFT_DIR/all_sft.jsonl"
else
    echo ">> No SFT files to merge"
fi

# =============================================================================
# Validation
# =============================================================================

echo ""
echo ">> Validating MS-SWIFT format..."

if [ -f "$MSSWIFT_DIR/all_sft.jsonl" ]; then
    python "$CONVERT_SCRIPT" validate "$MSSWIFT_DIR/all_sft.jsonl"
fi

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "============================================================"
echo "Processing Complete"
echo "============================================================"
echo ""
echo "Output Directories:"
echo "  Raw Data:     $OUTPUT_BASE"
echo "  MS-SWIFT:     $MSSWIFT_DIR"
echo ""
echo "Files Created:"
find "$MSSWIFT_DIR" -name "*.jsonl" -exec basename {} \; 2>/dev/null | sort | while read f; do
    filepath="$MSSWIFT_DIR/$f"
    if [ -f "$filepath" ]; then
        lines=$(wc -l < "$filepath")
        echo "  $f: $lines samples"
    fi
done
echo ""
echo "============================================================"
echo "Next Steps:"
echo "  1. Use the generated data for MS-SWIFT training:"
echo "     swift sft --dataset $MSSWIFT_DIR/all_sft.jsonl ..."
echo ""
echo "  2. Or use individual dataset files for specific tasks"
echo "============================================================"
echo ""
