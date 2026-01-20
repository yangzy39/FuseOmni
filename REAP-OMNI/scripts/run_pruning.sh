#!/bin/bash
#
# REAP-OMNI Pruning Pipeline for Qwen3-Omni-30B-A3B
#
# This script orchestrates three pruning operations:
# 1. Vision Modality Stripping - Remove vision encoder and projector
# 2. REAP Expert Pruning - Prune MoE experts based on audio affinity
# 3. Layer Similarity Pruning - Remove redundant transformer layers
#
# Usage:
#   ./run_pruning.sh [OPTIONS]
#
# Options:
#   --model-path PATH       Path to original model (default: ../models/Qwen3-Omni-30B-A3B-Instruct)
#   --output-dir PATH       Base output directory (default: ../models)
#   --vision-only           Only run vision stripping
#   --reap-only             Only run REAP expert pruning
#   --layer-only            Only run layer pruning
#   --retention-rate RATE   Expert retention rate for REAP (default: 0.5)
#   --max-layers NUM        Maximum layers to prune (default: 8)
#   --dry-run               Analyze without making changes
#   --help                  Show this help message
#
# Author: REAP-OMNI Implementation
#

set -e  # Exit on error

# ============================================================================
# Default Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${SCRIPT_DIR}/../models/Qwen3-Omni-30B-A3B-Instruct"
OUTPUT_DIR="${SCRIPT_DIR}/../models"

# Pruning parameters
RETENTION_RATE=0.5
MAX_LAYERS=8
SIMILARITY_THRESHOLD=0.9
COMPONENT="thinker"

# Flags
RUN_VISION=true
RUN_REAP=true
RUN_LAYER=true
DRY_RUN=false
VERBOSE=true

# ============================================================================
# Helper Functions
# ============================================================================

print_banner() {
    echo ""
    echo "============================================================"
    echo "  REAP-OMNI Pruning Pipeline"
    echo "  Qwen3-Omni-30B-A3B Model Compression"
    echo "============================================================"
    echo ""
}

print_step() {
    echo ""
    echo "------------------------------------------------------------"
    echo "  $1"
    echo "------------------------------------------------------------"
}

print_success() {
    echo "[SUCCESS] $1"
}

print_error() {
    echo "[ERROR] $1" >&2
}

print_info() {
    echo "[INFO] $1"
}

show_help() {
    echo "REAP-OMNI Pruning Pipeline for Qwen3-Omni-30B-A3B"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --model-path PATH       Path to original model"
    echo "                          Default: $MODEL_PATH"
    echo "  --output-dir PATH       Base output directory"
    echo "                          Default: $OUTPUT_DIR"
    echo "  --vision-only           Only run vision stripping"
    echo "  --reap-only             Only run REAP expert pruning"
    echo "  --layer-only            Only run layer pruning"
    echo "  --retention-rate RATE   Expert retention rate for REAP (0.0-1.0)"
    echo "                          Default: $RETENTION_RATE"
    echo "  --max-layers NUM        Maximum layers to prune"
    echo "                          Default: $MAX_LAYERS"
    echo "  --similarity-threshold  Layer similarity threshold"
    echo "                          Default: $SIMILARITY_THRESHOLD"
    echo "  --component NAME        Component to prune (thinker|talker)"
    echo "                          Default: $COMPONENT"
    echo "  --dry-run               Analyze without making changes"
    echo "  --quiet                 Reduce output verbosity"
    echo "  --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo ""
    echo "  # Run all pruning steps"
    echo "  $0 --model-path ./models/Qwen3-Omni-30B-A3B-Instruct"
    echo ""
    echo "  # Only strip vision modality"
    echo "  $0 --vision-only"
    echo ""
    echo "  # REAP pruning with 60% expert retention"
    echo "  $0 --reap-only --retention-rate 0.6"
    echo ""
    echo "  # Dry run to preview changes"
    echo "  $0 --dry-run"
    echo ""
}

check_python() {
    if ! command -v python &> /dev/null; then
        if command -v python3 &> /dev/null; then
            PYTHON_CMD="python3"
        else
            print_error "Python not found. Please install Python 3.8+."
            exit 1
        fi
    else
        PYTHON_CMD="python"
    fi
    print_info "Using Python: $($PYTHON_CMD --version)"
}

check_dependencies() {
    print_info "Checking dependencies..."
    
    $PYTHON_CMD -c "import torch" 2>/dev/null || {
        print_error "PyTorch not found. Install with: pip install torch"
        exit 1
    }
    
    $PYTHON_CMD -c "import safetensors" 2>/dev/null || {
        print_error "safetensors not found. Install with: pip install safetensors"
        exit 1
    }
    
    $PYTHON_CMD -c "from tqdm import tqdm" 2>/dev/null || {
        print_error "tqdm not found. Install with: pip install tqdm"
        exit 1
    }
    
    print_success "All dependencies satisfied"
}

check_model_path() {
    if [ ! -d "$MODEL_PATH" ]; then
        print_error "Model path not found: $MODEL_PATH"
        exit 1
    fi
    
    if [ ! -f "$MODEL_PATH/config.json" ]; then
        print_error "config.json not found in model path"
        exit 1
    fi
    
    if [ ! -f "$MODEL_PATH/model.safetensors.index.json" ]; then
        print_error "model.safetensors.index.json not found in model path"
        exit 1
    fi
    
    print_success "Model path validated: $MODEL_PATH"
}

# ============================================================================
# Parse Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --vision-only)
            RUN_VISION=true
            RUN_REAP=false
            RUN_LAYER=false
            shift
            ;;
        --reap-only)
            RUN_VISION=false
            RUN_REAP=true
            RUN_LAYER=false
            shift
            ;;
        --layer-only)
            RUN_VISION=false
            RUN_REAP=false
            RUN_LAYER=true
            shift
            ;;
        --retention-rate)
            RETENTION_RATE="$2"
            shift 2
            ;;
        --max-layers)
            MAX_LAYERS="$2"
            shift 2
            ;;
        --similarity-threshold)
            SIMILARITY_THRESHOLD="$2"
            shift 2
            ;;
        --component)
            COMPONENT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --quiet)
            VERBOSE=false
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Main Execution
# ============================================================================

print_banner

# Pre-flight checks
check_python
check_dependencies
check_model_path

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build common arguments
COMMON_ARGS=""
if [ "$DRY_RUN" = true ]; then
    COMMON_ARGS="$COMMON_ARGS --dry-run"
fi
if [ "$VERBOSE" = true ]; then
    COMMON_ARGS="$COMMON_ARGS --verbose"
fi

# ============================================================================
# Step 1: Vision Modality Stripping
# ============================================================================

if [ "$RUN_VISION" = true ]; then
    print_step "Step 1: Vision Modality Stripping"
    
    VISION_OUTPUT="$OUTPUT_DIR/Qwen3-Omni-30B-A3B-Vision-Stripped"
    
    print_info "Input:  $MODEL_PATH"
    print_info "Output: $VISION_OUTPUT"
    
    $PYTHON_CMD "$SCRIPT_DIR/vision_strip.py" \
        --model-path "$MODEL_PATH" \
        --output-path "$VISION_OUTPUT" \
        $COMMON_ARGS
    
    if [ "$DRY_RUN" = false ]; then
        print_success "Vision stripping completed"
        # Use vision-stripped model as input for next steps
        MODEL_PATH="$VISION_OUTPUT"
    fi
fi

# ============================================================================
# Step 2: REAP Expert Pruning
# ============================================================================

if [ "$RUN_REAP" = true ]; then
    print_step "Step 2: REAP Expert Pruning"
    
    REAP_OUTPUT="$OUTPUT_DIR/Qwen3-Omni-30B-A3B-REAP-Pruned"
    
    print_info "Input:  $MODEL_PATH"
    print_info "Output: $REAP_OUTPUT"
    print_info "Retention Rate: $RETENTION_RATE"
    print_info "Component: $COMPONENT"
    
    $PYTHON_CMD "$SCRIPT_DIR/reap_expert_pruning.py" \
        --model-path "$MODEL_PATH" \
        --output-path "$REAP_OUTPUT" \
        --component "$COMPONENT" \
        --retention-rate "$RETENTION_RATE" \
        $COMMON_ARGS
    
    if [ "$DRY_RUN" = false ]; then
        print_success "REAP expert pruning completed"
        # Use REAP-pruned model as input for next step
        MODEL_PATH="$REAP_OUTPUT"
    fi
fi

# ============================================================================
# Step 3: Layer Similarity Pruning
# ============================================================================

if [ "$RUN_LAYER" = true ]; then
    print_step "Step 3: Layer Similarity Pruning"
    
    LAYER_OUTPUT="$OUTPUT_DIR/Qwen3-Omni-30B-A3B-Layer-Pruned"
    
    print_info "Input:  $MODEL_PATH"
    print_info "Output: $LAYER_OUTPUT"
    print_info "Max Layers to Prune: $MAX_LAYERS"
    print_info "Similarity Threshold: $SIMILARITY_THRESHOLD"
    print_info "Component: $COMPONENT"
    
    $PYTHON_CMD "$SCRIPT_DIR/layer_similarity_pruning.py" \
        --model-path "$MODEL_PATH" \
        --output-path "$LAYER_OUTPUT" \
        --component "$COMPONENT" \
        --max-layers "$MAX_LAYERS" \
        --similarity-threshold "$SIMILARITY_THRESHOLD" \
        $COMMON_ARGS
    
    if [ "$DRY_RUN" = false ]; then
        print_success "Layer similarity pruning completed"
    fi
fi

# ============================================================================
# Summary
# ============================================================================

print_step "Pipeline Complete"

if [ "$DRY_RUN" = true ]; then
    print_info "This was a DRY RUN. No files were modified."
else
    print_info "Pruned models saved to: $OUTPUT_DIR"
    echo ""
    echo "Output models:"
    if [ "$RUN_VISION" = true ]; then
        echo "  - Vision-stripped: $OUTPUT_DIR/Qwen3-Omni-30B-A3B-Vision-Stripped"
    fi
    if [ "$RUN_REAP" = true ]; then
        echo "  - REAP-pruned:     $OUTPUT_DIR/Qwen3-Omni-30B-A3B-REAP-Pruned"
    fi
    if [ "$RUN_LAYER" = true ]; then
        echo "  - Layer-pruned:    $OUTPUT_DIR/Qwen3-Omni-30B-A3B-Layer-Pruned"
    fi
fi

echo ""
echo "============================================================"
echo "  REAP-OMNI Pruning Pipeline Finished"
echo "============================================================"
echo ""
