#!/bin/bash
# SCO ACP Job Submission Script Template
# Usage: ./submit_job.sh

# ========== USER CONFIGURATION (modify these) ==========
GPU_NUMS=4

# Training configuration
YAML_FILE="/mnt/afs/00036/ljh/lmf_lhr/lmf/exp/llama3.2_1b/fusechat.yaml"
LOG_FILE_PATH="/mnt/afs/00036/ljh/lmf_lhr/lmf/exp/llama3.2_1b/logs"
output_dir="/mnt/afs/00036/ljh/code_files/key_token/save_models/Llama-3.2-1B-Instruct/my_experiment"

# Conda environment name
CONDA_ENV="ljh_lmf"

# Working directory (where training code is located)
WORK_DIR="/mnt/afs/00036/ljh/lmf_lhr/lmf"

# Optional: Setup script to run before training
SETUP_SCRIPT="/mnt/afs/00036/ljh/lmf_lhr/lmf/exp/llama3.2_1b/setup.sh"

# Optional: SwanLab experiment tracking
use_swanlab=true
swanlab_run_name="my_experiment"

# Optional: TopK Loss configuration
use_topk_loss=false
topk_loss_mode="max"
topk_loss_ratio=0.9

# ========== DO NOT MODIFY BELOW (unless you know what you're doing) ==========

# Create log directory
mkdir -p "${LOG_FILE_PATH}"
LOG_FILE="${LOG_FILE_PATH}/terminal_log.txt"

# Build training command
TRAIN_CMD="llamafactory-cli train ${YAML_FILE}"
TRAIN_CMD="${TRAIN_CMD} output_dir=\"${output_dir}\""
TRAIN_CMD="${TRAIN_CMD} per_device_train_batch_size=8"
TRAIN_CMD="${TRAIN_CMD} gradient_accumulation_steps=4"

if [[ "${use_topk_loss}" == "true" ]]; then
  TRAIN_CMD="${TRAIN_CMD} use_topk_loss=${use_topk_loss}"
  TRAIN_CMD="${TRAIN_CMD} topk_loss_ratio=${topk_loss_ratio}"
  TRAIN_CMD="${TRAIN_CMD} topk_loss_mode=${topk_loss_mode}"
fi

if [[ "${use_swanlab}" == "true" ]]; then
  TRAIN_CMD="${TRAIN_CMD} use_swanlab=${use_swanlab}"
  TRAIN_CMD="${TRAIN_CMD} swanlab_run_name=\"${swanlab_run_name}\""
fi

# Submit job
sco acp jobs create \
  --workspace-name=share-space \
  --aec2-name=share-cluster \
  --job-name=training \
  --container-image-url="registry.cn-sh-01.sensecore.cn/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image" \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=n6ls.iu.i40.${GPU_NUMS} \
  --priority=normal \
  --storage-mount=01995892-d478-76d8-aec7-13fd8284477e:/mnt/afs \
  --command="source /mnt/afs/00036/software/conda/bin/activate ${CONDA_ENV}; \
    source ${SETUP_SCRIPT} &> ${LOG_FILE}; \
    cd ${WORK_DIR}; \
    pwd &>> ${LOG_FILE}; \
    ${TRAIN_CMD} &>> ${LOG_FILE}"
