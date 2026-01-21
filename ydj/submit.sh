#!/bin/bash
# ========== SCO ACP - Submit Perpetual Motion Machine ==========
# Submits the GPU perpetual motion machine to SCO ACP cluster
# Once running, you can submit/stop tasks from anywhere via shared storage
# 
# Usage: ./submit.sh <JOB_ID> [GPU_NUMS] [WORKER_NODES]
#   JOB_ID:       Required. Unique name for this perpetual motion instance
#   GPU_NUMS:     Optional. Number of GPUs per node (default: 2)
#   WORKER_NODES: Optional. Number of nodes (default: 1)
#
# Examples:
#   ./submit.sh train-exp1           # 2 GPUs, 1 node, queue at gpu_queue/train-exp1/
#   ./submit.sh train-exp2 4         # 4 GPUs, 1 node
#   ./submit.sh train-exp3 8 2       # 8 GPUs, 2 nodes
# ================================================================

set -e

# ========== ARGUMENTS ==========
if [ -z "$1" ]; then
    echo "Usage: $0 <JOB_ID> [GPU_NUMS] [WORKER_NODES]"
    echo ""
    echo "Arguments:"
    echo "  JOB_ID       Required. Unique name for this instance (e.g., 'train-exp1')"
    echo "  GPU_NUMS     Optional. GPUs per node (default: 2)"
    echo "  WORKER_NODES Optional. Number of nodes (default: 1)"
    echo ""
    echo "Examples:"
    echo "  $0 train-exp1           # Create queue at gpu_queue/train-exp1/"
    echo "  $0 train-exp2 4         # 4 GPUs"
    echo "  $0 train-exp3 8 2       # 8 GPUs, 2 nodes"
    exit 1
fi

JOB_ID="$1"
GPU_NUMS=${2:-2}
WORKER_NODES=${3:-1}

# ========== PATHS (modify these to match your environment) ==========
CODE_DIR="/mnt/afs/00036/yzy/FuseOmni/ydj"
QUEUE_ROOT="/mnt/afs/00036/yzy/gpu_queue"

# Conda environment
CONDA_PATH="/mnt/afs/00036/software/conda/bin/activate"
CONDA_ENV="swift"

# ========== JOB SUBMISSION ==========
echo "=========================================="
echo "Submitting GPU Perpetual Motion Machine"
echo "=========================================="
echo "JOB_ID: ${JOB_ID}"
echo "GPUs:   ${GPU_NUMS} per node"
echo "Nodes:  ${WORKER_NODES}"
echo "Queue:  ${QUEUE_ROOT}/${JOB_ID}/"
echo "Code:   ${CODE_DIR}"
echo ""

sco acp jobs create \
  --workspace-name=share-space \
  --aec2-name=share-cluster \
  --job-name=ydj-${JOB_ID} \
  --container-image-url="registry.cn-sh-01.sensecore.cn/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image" \
  --training-framework=pytorch \
  --worker-nodes=${WORKER_NODES} \
  --worker-spec=n6ls.iu.i40.${GPU_NUMS} \
  --priority=normal \
  --storage-mount=01995892-d478-76d8-aec7-13fd8284477e:/mnt/afs \
  --command="source ${CONDA_PATH} ${CONDA_ENV}; \
    export JOB_ID=${JOB_ID}; \
    export QUEUE_ROOT=${QUEUE_ROOT}; \
    cd ${CODE_DIR}; \
    python perpetual_motion.py --job-id ${JOB_ID} --queue-root ${QUEUE_ROOT}"

echo ""
echo "=========================================="
echo "Perpetual Motion Machine Submitted!"
echo "=========================================="
echo ""
echo "This instance: ${JOB_ID}"
echo "Queue path:    ${QUEUE_ROOT}/${JOB_ID}/"
echo ""
echo "Submit tasks from ANYWHERE:"
echo ""
echo "  # 1. Load client tools (specify JOB_ID)"
echo "  export JOB_ID=${JOB_ID}"
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
