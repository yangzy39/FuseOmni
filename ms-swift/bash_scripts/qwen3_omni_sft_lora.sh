#!/bin/bash
# =============================================================================
# Qwen3-Omni-30B-A3B-Instruct SFT Training Script (LoRA)
# =============================================================================
# 
# Requirements:
#   - 4x A100 80GB GPUs (or equivalent)
#   - pip install transformers>=4.57 soundfile decord qwen_omni_utils
#
# Dataset Format (JSONL):
#   {"messages": [{"role": "user", "content": "<audio>What did the audio say?"}, {"role": "assistant", "content": "..."}], "audios": ["/path/to/audio.wav"]}
#   {"messages": [{"role": "user", "content": "<video>Describe this video"}, {"role": "assistant", "content": "..."}], "videos": ["/path/to/video.mp4"]}
#   {"messages": [{"role": "user", "content": "<image>What is in this image?"}, {"role": "assistant", "content": "..."}], "images": ["/path/to/image.jpg"]}
#
# Usage:
#   bash qwen3_omni_sft_lora.sh
# =============================================================================

set -e

# ========================= Configuration =========================
# Model
MODEL_ID="Qwen/Qwen3-Omni-30B-A3B-Instruct"

# Dataset - Replace with your custom dataset path
# Format: /path/to/dataset.jsonl or dataset_id#num_samples
DATASET="/path/to/your/sft_data.jsonl"

# Output
OUTPUT_DIR="./output/qwen3_omni_sft_lora"

# GPU Configuration
CUDA_DEVICES="0,1,2,3"
NPROC_PER_NODE=4

# ========================= Environment Variables =========================
# Multimodal processing settings
export MAX_PIXELS=1003520           # Max pixels for images
export VIDEO_MAX_PIXELS=50176       # Max pixels for video frames
export FPS_MAX_FRAMES=12            # Max frames to extract from video
export ENABLE_AUDIO_OUTPUT=0        # Set to 1 if training audio generation

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ========================= Training Command =========================
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
swift sft \
    --model ${MODEL_ID} \
    --dataset "${DATASET}" \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --max_length 4096 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --output_dir ${OUTPUT_DIR} \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --load_from_cache_file true \
    --freeze_vit true \
    --freeze_aligner true \
    --deepspeed zero2 \
    --gradient_checkpointing true

echo "Training completed! Model saved to ${OUTPUT_DIR}"
