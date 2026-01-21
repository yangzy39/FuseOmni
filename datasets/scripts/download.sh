#!/bin/bash
# =============================================================================
# Dataset Download Script using huggingface-cli
# =============================================================================
#
# Downloads datasets one by one using huggingface-cli download.
# This preserves the original dataset structure for specialized processing.
#
# Usage:
#   ./download.sh                          # Download all configured datasets
#   ./download.sh --datasets librispeech   # Download specific dataset
#   ./download.sh --list                   # List available datasets
#
# Environment Variables:
#   HF_ENDPOINT   - HuggingFace mirror (default: https://hf-mirror.com)
#   HF_TOKEN      - Your HuggingFace authentication token
#
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_OUTPUT_DIR="/mnt/afs/00036/yzy/FuseOmni/datasets/data"

# HuggingFace settings
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_TOKEN="${HF_TOKEN:-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}"

# Dataset registry: name|hf_id|requires_auth|description
declare -A DATASETS=(
    # Audio-only (ASR)
    ["librispeech"]="openslr/librispeech_asr|false|English audiobook ASR"
    ["common_voice"]="mozilla-foundation/common_voice_17_0|true|Multilingual crowdsourced ASR"
    ["gigaspeech"]="speechcolab/gigaspeech|true|Large-scale English ASR"
    ["aishell1"]="AISHELL/AISHELL-1|false|Chinese Mandarin ASR"
    ["voxpopuli"]="facebook/voxpopuli|false|European Parliament multilingual"
    ["wenetspeech"]="wenet-e2e/wenetspeech|false|Large-scale Chinese ASR"
    ["libritts"]="openslr/libritts|false|English TTS dataset"
    
    # Audio captioning
    ["wavcaps"]="cvssp/WavCaps|false|Audio captioning dataset"
    
    # Video datasets
    ["youcook2"]="merve/YouCook2|false|Cooking instruction videos"
    ["longvideobench"]="longvideobench/LongVideoBench|false|Long video understanding"
    
    # Mixed (audio+video)
    ["ugc_videocap"]="openinterx/UGC-VideoCap|false|Short video multimodal captioning"
)

# =============================================================================
# Functions
# =============================================================================

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Downloads datasets using huggingface-cli for specialized processing."
    echo ""
    echo "Options:"
    echo "  --output DIR         Output directory (default: $DEFAULT_OUTPUT_DIR)"
    echo "  --datasets NAME...   Specific datasets to download"
    echo "  --list               List available datasets"
    echo "  --token TOKEN        HuggingFace token (or set HF_TOKEN env var)"
    echo "  --help               Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  HF_ENDPOINT          HuggingFace mirror URL"
    echo "  HF_TOKEN             HuggingFace authentication token"
    echo ""
    echo "Examples:"
    echo "  $0 --list"
    echo "  $0 --datasets librispeech aishell1"
    echo "  $0 --output ./my_data --datasets librispeech"
}

list_datasets() {
    echo ""
    echo "============================================================"
    echo "Available Datasets"
    echo "============================================================"
    echo ""
    printf "%-20s %-40s %-6s %s\n" "NAME" "HF_ID" "AUTH" "DESCRIPTION"
    echo "------------------------------------------------------------"
    
    for name in "${!DATASETS[@]}"; do
        IFS='|' read -r hf_id requires_auth description <<< "${DATASETS[$name]}"
        auth_marker=""
        if [ "$requires_auth" = "true" ]; then
            auth_marker="[AUTH]"
        fi
        printf "%-20s %-40s %-6s %s\n" "$name" "$hf_id" "$auth_marker" "$description"
    done | sort
    
    echo ""
    echo "[AUTH] = Requires HuggingFace authentication (set HF_TOKEN)"
    echo "============================================================"
    echo ""
}

download_dataset() {
    local name="$1"
    local output_dir="$2"
    
    if [ -z "${DATASETS[$name]}" ]; then
        echo "[ERROR] Unknown dataset: $name"
        return 1
    fi
    
    IFS='|' read -r hf_id requires_auth description <<< "${DATASETS[$name]}"
    
    echo ""
    echo "============================================================"
    echo "Downloading: $name"
    echo "  HF ID: $hf_id"
    echo "  Output: $output_dir/$name"
    echo "============================================================"
    
    # Check authentication
    if [ "$requires_auth" = "true" ] && [ -z "$HF_TOKEN" ]; then
        echo "[WARNING] Dataset requires authentication but HF_TOKEN not set"
        echo "[WARNING] Skipping $name"
        return 0
    fi
    
    # Create output directory
    local dataset_dir="$output_dir/$name"
    mkdir -p "$dataset_dir"
    
    # Build download command
    local cmd="huggingface-cli download --repo-type dataset \"$hf_id\" --local-dir \"$dataset_dir\""
    
    if [ -n "$HF_TOKEN" ]; then
        cmd="$cmd --token \"$HF_TOKEN\""
    fi
    
    # Execute download
    echo ""
    echo "[CMD] $cmd"
    echo ""
    
    if eval "$cmd"; then
        echo ""
        echo "[OK] Successfully downloaded $name"
        
        # List downloaded contents
        echo ""
        echo "Downloaded structure:"
        ls -la "$dataset_dir" | head -20
    else
        echo ""
        echo "[ERROR] Failed to download $name"
        return 1
    fi
}

# =============================================================================
# Argument Parsing
# =============================================================================

OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
DATASETS_TO_DOWNLOAD=()
LIST_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --datasets)
            shift
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                DATASETS_TO_DOWNLOAD+=("$1")
                shift
            done
            ;;
        --token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --list)
            LIST_ONLY=true
            shift
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
# Main
# =============================================================================

# List datasets if requested
if [ "$LIST_ONLY" = true ]; then
    list_datasets
    exit 0
fi

# Set environment
export HF_ENDPOINT="$HF_ENDPOINT"
echo ""
echo "============================================================"
echo "Dataset Download Script"
echo "============================================================"
echo ""
echo "HF_ENDPOINT: $HF_ENDPOINT"
echo "HF_TOKEN: ${HF_TOKEN:+[SET]}"
echo "Output: $OUTPUT_DIR"

# Check huggingface-cli
if ! command -v huggingface-cli &> /dev/null; then
    echo ""
    echo "[ERROR] huggingface-cli not found. Install with:"
    echo "  pip install huggingface_hub"
    exit 1
fi

# Determine which datasets to download
if [ ${#DATASETS_TO_DOWNLOAD[@]} -eq 0 ]; then
    # Download all datasets
    echo ""
    echo "No datasets specified, downloading ALL datasets..."
    DATASETS_TO_DOWNLOAD=("${!DATASETS[@]}")
fi

echo ""
echo "Datasets to download: ${DATASETS_TO_DOWNLOAD[*]}"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Download each dataset
SUCCESS=()
FAILED=()
SKIPPED=()

for name in "${DATASETS_TO_DOWNLOAD[@]}"; do
    if download_dataset "$name" "$OUTPUT_DIR"; then
        SUCCESS+=("$name")
    else
        FAILED+=("$name")
    fi
done

# Summary
echo ""
echo "============================================================"
echo "Download Summary"
echo "============================================================"
echo ""
echo "Successful: ${#SUCCESS[@]}"
for name in "${SUCCESS[@]}"; do
    echo "  [OK] $name"
done

if [ ${#SKIPPED[@]} -gt 0 ]; then
    echo ""
    echo "Skipped: ${#SKIPPED[@]}"
    for name in "${SKIPPED[@]}"; do
        echo "  [-] $name"
    done
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed: ${#FAILED[@]}"
    for name in "${FAILED[@]}"; do
        echo "  [X] $name"
    done
fi

echo ""
echo "============================================================"
echo "Next Steps:"
echo "  1. Run processor to convert to MS-SWIFT format:"
echo "     python process.py --input $OUTPUT_DIR --output ./output"
echo ""
echo "  2. Or process individual datasets:"
echo "     python process.py --input $OUTPUT_DIR/librispeech --dataset librispeech"
echo "============================================================"
echo ""
