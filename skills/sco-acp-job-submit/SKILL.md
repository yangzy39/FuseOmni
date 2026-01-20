---
name: sco-acp-job-submit
description: |
  Generate and manage SCO ACP (高性能AI算力池) job submission scripts for GPU training tasks.
  Use when: (1) Creating training job scripts for SCO ACP cluster, (2) Configuring GPU resources (n6ls.iu.i40.X),
  (3) Setting up LLaMA-Factory or other ML training commands, (4) Managing ACP jobs (delete/list/exec),
  (5) User mentions "提交任务", "SCO ACP", "算力池", "GPU训练", "sco acp jobs create"
---

# SCO ACP Job Submit

Generate shell scripts for submitting GPU training jobs to SCO ACP (高性能AI算力池).

## Quick Reference

### Submit Job
```bash
path/to/submit_job.sh
# Returns: pt-xxxxxxxx (Job ID)
```

### Delete Job
```bash
sco acp jobs delete --workspace-name=share-space pt-xxxxxxxx
```

### List Jobs
```bash
sco acp jobs list --workspace-name=share-space
# Real-time watch (filter by username):
watch -n 0.5 "sco acp jobs list --workspace-name=share-space --page-size 500 | awk 'NR<=3 || /USERNAME/'"
```

### Login to Running Node
```bash
Jobid=pt-xxxxxxxx; sco acp jobs exec --workspace-name=share-space --worker-name=$(sco acp jobs get-workers --workspace-name=share-space $Jobid | grep "worker-0" | awk -F'|' '{print $2}' | xargs) $Jobid
```

## Workflow: Generate Job Script

### Step 1: Gather Required Parameters

Ask user for these parameters (provide defaults where shown):

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `GPU_NUMS` | Yes | 4 | Number of GPUs (1-8) |
| `YAML_FILE` | Yes | - | Path to training config YAML |
| `LOG_FILE_PATH` | Yes | - | Directory for logs |
| `output_dir` | Yes | - | Model checkpoint output path |
| Additional training args | No | - | e.g., batch_size, learning_rate |

### Step 2: Generate Script Using Template

Use `assets/job_template.sh` as base. Fill in:

1. **Variables section**: GPU count, paths, training flags
2. **Command section**: Training command with all arguments

### Step 3: Output Complete Script

Generate a `.sh` file that user can execute directly.

## Script Template Structure

```bash
# ========== VARIABLES (modify as needed) ==========
GPU_NUMS=4

# Training config
YAML_FILE="/mnt/afs/path/to/config.yaml"
LOG_FILE_PATH="/mnt/afs/path/to/logs"
mkdir -p "${LOG_FILE_PATH}"
LOG_FILE="${LOG_FILE_PATH}/terminal_log.txt"
output_dir="/mnt/afs/path/to/save_models"

# ========== JOB SUBMISSION (usually unchanged) ==========
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
  --command="source /mnt/afs/00036/software/conda/bin/activate ENV_NAME; \
    cd /mnt/afs/path/to/code; \
    YOUR_TRAINING_COMMAND &>> ${LOG_FILE}"
```

## Key Parameters Reference

### Cluster & Resources
- `--workspace-name=share-space` - Workspace (usually fixed)
- `--aec2-name=share-cluster` - Cluster name (usually fixed)
- `--worker-nodes=1` - Node count (typically 1)
- `--worker-spec=n6ls.iu.i40.${GPU_NUMS}` - Machine spec with GPU count

### Container (DO NOT change unless necessary)
- Image includes: CUDA 12.4, PyTorch 2.3, TransformerEngine

### Storage
- `--storage-mount=01995892-d478-76d8-aec7-13fd8284477e:/mnt/afs` - AFS mount point
- All code, models, data accessed via `/mnt/afs`

## Common Training Commands

### LLaMA-Factory
```bash
llamafactory-cli train ${YAML_FILE} \
  output_dir="${output_dir}" \
  per_device_train_batch_size=8 \
  gradient_accumulation_steps=4 \
  &>> ${LOG_FILE}
```

### Custom Training Script
```bash
python train.py \
  --config ${YAML_FILE} \
  --output_dir ${output_dir} \
  &>> ${LOG_FILE}
```

## Warnings

- Deleting a job immediately releases GPU and terminates training
- Unsaved checkpoints will be lost on job deletion
- Long idle sessions will disconnect (re-run exec command to reconnect)
