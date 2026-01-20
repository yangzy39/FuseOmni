#!/bin/bash
# =============================================================================
# Qwen3-Omni-30B-A3B-Instruct GRPO Training Script
# =============================================================================
# 
# GRPO (Group Relative Policy Optimization) for reinforcement learning
# with multimodal Qwen3-Omni model.
#
# Requirements:
#   - 4x A100 80GB GPUs
#   - pip install transformers>=4.57 math_verify trl soundfile decord qwen_omni_utils
#
# Dataset Format (JSONL) - Only prompts, no responses needed:
#   {"messages": [{"role": "user", "content": "<image>Solve this math problem"}], "images": ["/path/to/image.jpg"]}
#   {"messages": [{"role": "user", "content": "<audio>Transcribe and answer the question"}], "audios": ["/path/to/audio.wav"]}
#   {"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Question here"}]}
#
# Note: For GRPO, the model generates responses and learns from reward signals.
#       You can pass additional fields (e.g., "solution") for custom reward functions.
#
# Usage:
#   bash qwen3_omni_grpo.sh
# =============================================================================

set -e

# ========================= Configuration =========================
MODEL_ID="Qwen/Qwen3-Omni-30B-A3B-Instruct"

# Dataset for GRPO - prompts only
DATASET="/path/to/your/grpo_prompts.jsonl"

# Output
OUTPUT_DIR="./output/qwen3_omni_grpo"

# GPU Configuration
CUDA_DEVICES="0,1,2,3"
NPROC_PER_NODE=4

# ========================= Environment Variables =========================
export MAX_PIXELS=1003520
export VIDEO_MAX_PIXELS=50176
export FPS_MAX_FRAMES=12
export ENABLE_AUDIO_OUTPUT=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ========================= GRPO Training Command =========================
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} \
NPROC_PER_NODE=${NPROC_PER_NODE} \
swift rlhf \
    --rlhf_type grpo \
    --model ${MODEL_ID} \
    --dataset "${DATASET}" \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --reward_funcs format \
    --reward_weights 1.0 \
    --num_generations 4 \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --max_length 4096 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir ${OUTPUT_DIR} \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --load_from_cache_file true \
    --temperature 1.0 \
    --top_p 0.99 \
    --top_k 50 \
    --deepspeed zero2 \
    --log_completions true

echo "GRPO training completed! Model saved to ${OUTPUT_DIR}"
