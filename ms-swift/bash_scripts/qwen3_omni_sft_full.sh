#!/bin/bash
# =============================================================================
# Qwen3-Omni-30B-A3B-Instruct SFT Training Script (Full Fine-tuning)
# =============================================================================
# 
# Requirements:
#   - 8x A100 80GB GPUs (or 16x A100 40GB)
#   - pip install transformers>=4.57 soundfile decord qwen_omni_utils
#
# WARNING: Full fine-tuning requires significantly more GPU memory!
#          Use LoRA (qwen3_omni_sft_lora.sh) for most use cases.
#
# Dataset Format (JSONL):
#   {"messages": [{"role": "user", "content": "<audio>What did the audio say?"}, {"role": "assistant", "content": "..."}], "audios": ["/path/to/audio.wav"]}
#   {"messages": [{"role": "user", "content": "<video>Describe this video"}, {"role": "assistant", "content": "..."}], "videos": ["/path/to/video.mp4"]}
#
# Usage:
#   bash qwen3_omni_sft_full.sh
# =============================================================================

set -e

# ========================= Configuration =========================
MODEL_PATH="/mnt/afs/00036/yzy/FuseOmni/models"
MODEL_ID="Qwen3-Omni-30B-A3B-Instruct"
DATASET="/path/to/your/sft_data.jsonl"
OUTPUT_DIR="./output/qwen3_omni_sft_full"

# GPU Configuration
CUDA_DEVICES="0,1,2,3,4,5,6,7"
NPROC_PER_NODE=8

# ========================= Environment Variables =========================
export MAX_PIXELS=1003520
export VIDEO_MAX_PIXELS=50176
export FPS_MAX_FRAMES=12
export ENABLE_AUDIO_OUTPUT=0
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ========================= Training Command =========================
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
swift sft \
    --model ${MODEL_PATH}/${MODEL_ID} \
    --dataset "${DATASET}" \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --output_dir ${OUTPUT_DIR} \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --load_from_cache_file true \
    --freeze_vit true \
    --freeze_aligner true \
    --deepspeed zero3 \
    --gradient_checkpointing true

echo "Full fine-tuning completed! Model saved to ${OUTPUT_DIR}"
