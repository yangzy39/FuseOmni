#!/bin/bash
# =============================================================================
# Qwen3-Omni-30B-A3B-Instruct GRPO Training Script (with vLLM acceleration)
# =============================================================================
# 
# This script uses vLLM for faster generation during GRPO training.
# The "colocate" mode runs vLLM within the same process for efficiency.
#
# Requirements:
#   - 4x A100 80GB GPUs
#   - pip install transformers>=4.57 vllm>=0.5.1 math_verify trl
#
# Usage:
#   bash qwen3_omni_grpo_vllm.sh
# =============================================================================

set -e

# ========================= Configuration =========================
MODEL_ID="Qwen/Qwen3-Omni-30B-A3B-Instruct"
DATASET="/path/to/your/grpo_prompts.jsonl"
OUTPUT_DIR="./output/qwen3_omni_grpo_vllm"

# GPU Configuration
CUDA_DEVICES="0,1,2,3"
NPROC_PER_NODE=4

# ========================= Environment Variables =========================
export MAX_PIXELS=1003520
export VIDEO_MAX_PIXELS=50176
export FPS_MAX_FRAMES=12
export ENABLE_AUDIO_OUTPUT=1
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'

# ========================= GRPO + vLLM Training Command =========================
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
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --reward_funcs format \
    --reward_weights 1.0 \
    --num_generations 8 \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --max_length 8192 \
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

echo "GRPO training with vLLM completed! Model saved to ${OUTPUT_DIR}"
