#!/bin/bash
# ========== SCO ACP Job Submit Script ==========
# Training: Qwen3-Omni-30B-A3B-Instruct SFT with DeepSpeed ZeRO3
# GPU: 2x L40S (2 * 60GiB)
# ================================================

set -e

# ========== VARIABLES (modify as needed) ==========
GPU_NUMS=2

# Paths
MODEL_PATH="/mnt/afs/00036/yzy/FuseOmni/models"
MODEL_ID="Qwen3-Omni-30B-A3B-Instruct"
OUTPUT_DIR="/mnt/afs/00036/yzy/checkpoints/qwen3_omni_sft_full"
CODE_DIR="/mnt/afs/00036/yzy/FuseOmni/ms-swift"

# Log configuration
LOG_FILE_PATH="/mnt/afs/00036/yzy/logs/qwen3_omni_sft_zero3"
LOG_FILE="${LOG_FILE_PATH}/terminal_log_$(date +%Y%m%d_%H%M%S).txt"

# Conda environment
CONDA_PATH="/mnt/afs/00036/software/conda/bin/activate"
CONDA_ENV="swift"

# ========== JOB SUBMISSION ==========
sco acp jobs create \
  --workspace-name=share-space \
  --aec2-name=share-cluster \
  --job-name=qwen3-omni-sft-zero3 \
  --container-image-url="registry.cn-sh-01.sensecore.cn/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image" \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=n6ls.iu.i40.${GPU_NUMS} \
  --priority=normal \
  --storage-mount=01995892-d478-76d8-aec7-13fd8284477e:/mnt/afs \
  --command="source ${CONDA_PATH} ${CONDA_ENV}; \
    mkdir -p ${LOG_FILE_PATH}; \
    cd ${CODE_DIR}; \
    MAX_PIXELS=1003520 \
    NPROC_PER_NODE=${GPU_NUMS} \
    VIDEO_MAX_PIXELS=50176 \
    FPS_MAX_FRAMES=12 \
    CUDA_VISIBLE_DEVICES=0,1 \
    swift sft \
        --model ${MODEL_PATH}/${MODEL_ID} \
        --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#10000' \
                  'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
                  'speech_asr/speech_asr_aishell1_trainsets:train#5000' \
                  'modelscope/speech_asr_commonvoice_en_trainsets:train#5000' \
        --split_dataset_ratio 0.01 \
        --load_from_cache_file true \
        --tuner_type lora \
        --torch_dtype bfloat16 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --attn_impl flash_attn \
        --learning_rate 1e-4 \
        --lora_rank 8 \
        --lora_alpha 32 \
        --target_modules all-linear \
        --freeze_vit true \
        --freeze_aligner true \
        --padding_free true \
        --gradient_accumulation_steps 1 \
        --gradient_checkpointing true \
        --eval_steps 50 \
        --save_steps 50 \
        --save_total_limit 2 \
        --logging_steps 5 \
        --max_length 4096 \
        --output_dir ${OUTPUT_DIR} \
        --warmup_ratio 0.05 \
        --dataset_num_proc 4 \
        --deepspeed zero3 \
        --dataloader_num_workers 4 \
    &>> ${LOG_FILE}"

echo "Job submitted! Use 'sco acp jobs list --workspace-name=share-space' to check status."
echo "Log file: ${LOG_FILE}"
