#!/bin/bash
# ========== SCO ACP - Submit Perpetual Motion Machine ==========
# Submits the GPU perpetual motion machine to SCO ACP cluster
# Once running, you can submit/stop tasks from anywhere via shared storage
# ================================================================

set -e

# ========== VARIABLES (modify as needed) ==========
GPU_NUMS=${1:-2}      # Default 2 GPUs, can override: ./submit.sh 4
WORKER_NODES=${2:-1}  # Default 1 node, can override: ./submit.sh 4 2
JOB_NAME="gpu-perpetual-motion"

# Paths (modify these to match your environment)
CODE_DIR="/mnt/afs/00036/yzy/FuseOmni/ydj"
QUEUE_BASE="/mnt/afs/00036/yzy/gpu_queue"

# Conda environment
CONDA_PATH="/mnt/afs/00036/software/conda/bin/activate"
CONDA_ENV="swift"

# ========== JOB SUBMISSION ==========
echo "=========================================="
echo "Submitting GPU Perpetual Motion Machine"
echo "=========================================="
echo "GPUs:  ${GPU_NUMS}"
echo "Nodes: ${WORKER_NODES}"
echo "Queue: ${QUEUE_BASE}"
echo "Code:  ${CODE_DIR}"
echo ""

sco acp jobs create \
  --workspace-name=share-space \
  --aec2-name=share-cluster \
  --job-name=${JOB_NAME} \
  --container-image-url="registry.cn-sh-01.sensecore.cn/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image" \
  --training-framework=pytorch \
  --worker-nodes=${WORKER_NODES} \
  --worker-spec=n6ls.iu.i40.${GPU_NUMS} \
  --priority=normal \
  --storage-mount=01995892-d478-76d8-aec7-13fd8284477e:/mnt/afs \
  --command="source ${CONDA_PATH} ${CONDA_ENV}; \
    export QUEUE_BASE=${QUEUE_BASE}; \
    cd ${CODE_DIR}; \
    python perpetual_motion.py"

echo ""
echo "=========================================="
echo "Perpetual Motion Machine Submitted!"
echo "=========================================="
echo ""
echo "Submit tasks from ANYWHERE (no need to login to node):"
echo ""
echo "  # 1. Load client tools"
echo "  source ${CODE_DIR}/client.sh"
echo ""
echo "  # 2. Submit tasks"
echo "  gpu_submit /path/to/train.sh"
echo "  gpu_run 'python train.py --epochs 10'"
echo ""
echo "  # 3. Monitor"
echo "  gpu_status"
echo "  gpu_tail"
echo ""
echo "  # 4. Control"
echo "  gpu_stop <task_name>"
echo "  gpu_shutdown"
echo ""
echo "=========================================="
