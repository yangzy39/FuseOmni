#!/bin/bash

# REAP-OMNI Dataset Preparation Script
# This script installs dependencies and downloads the recommended calibration datasets.

set -e  # Exit on error

# Configuration
OUTPUT_DIR="/mnt/afs/00036/yzy/FuseOmni/datasets/data"
SAMPLES=100

echo "========================================================"
echo "   REAP-OMNI Dataset Auto-Preparation Pipeline"
echo "========================================================"

# 1. Check Python environment
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 could not be found."
    exit 1
fi

echo "✅ Python environment check passed."

# 2. Install Dependencies
echo ""
echo "📦 Step 1: Installing dependencies..."
python3 -m pip install datasets soundfile tqdm pandas requests

# 3. Run Download & Conversion
echo ""
echo "⬇️  Step 2: Downloading and converting datasets..."
echo "   Target Directory: $OUTPUT_DIR"
echo "   Samples per dataset: $SAMPLES"

# Using quickstart.py which orchestrates the download of recommended datasets
python3 quickstart.py --output "$OUTPUT_DIR" --samples "$SAMPLES"

# 4. Verification
echo ""
echo "🔍 Step 3: Verifying output..."

if [ -d "$OUTPUT_DIR/calibration" ]; then
    echo "✅ Calibration data created successfully:"
    ls -lh "$OUTPUT_DIR/calibration/"
    
    echo ""
    echo "🎉 Done! You can now use these files for REAP pruning:"
    echo ""
    echo "  python ../reap_expert_pruning.py \\"
    echo "      --audio-data $OUTPUT_DIR/calibration/audio.jsonl \\"
    echo "      --video-data $OUTPUT_DIR/calibration/video.jsonl \\"
    echo "      --mixed-data $OUTPUT_DIR/calibration/mixed.jsonl"
else
    echo "❌ Error: Calibration directory not found!"
    exit 1
fi
