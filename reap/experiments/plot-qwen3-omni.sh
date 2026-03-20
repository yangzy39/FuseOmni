#!/bin/bash

# Configuration
OBSERVATIONS_DIR="artifacts/Qwen3-Omni-30B-A3B-Instruct/aishell1_test_asr.jsonl/all"
OUTPUT_DIR="fig/qwen3-omni-clusters"
SEED=42
NUM_SAMPLES=1024

# Derived path (matching the output of prune-qwen3-omni.sh)
OBS_FILE="${OBSERVATIONS_DIR}/observations_${NUM_SAMPLES}_cosine-seed_${SEED}.pt"

CONDA_PATH="/mnt/afs/00036/software/conda/bin/activate"
CONDA_ENV="reap_omni"
source ${CONDA_PATH} ${CONDA_ENV}

# artifacts/Qwen3-Omni-30B-A3B-Instruct/aishell1_test_asr.jsonl/all/observations_2048_cosine-seed_42.pt
# Ensure output directory exists
mkdir -p ${OUTPUT_DIR}

echo "Running cluster analysis for Qwen3-Omni..."
echo "Observations: ${OBS_FILE}"
echo "Output: ${OUTPUT_DIR}"

cd /mnt/afs/00036/project_fuseomni/FuseOmni/reap

# Run the clustering and plotting
python src/reap/cluster.py \
    --observations_path "${OBS_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --compression_ratio 0.5 \
    --expert_sim "router_logits" \
    --cluster_method "agglomerative"

echo "Plotting completed. Check ${OUTPUT_DIR} for results."
