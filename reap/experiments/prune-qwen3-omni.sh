#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1:-"0"}
FIRST_DEVICE=$(echo "$1" | cut -d',' -f1)
port=$((8000 + FIRST_DEVICE))
model_name="D:/PycharmProjects/FuseOmni/models/Qwen3-Omni-30B-A3B-Instruct"
dataset_name="D:/PycharmProjects/FuseOmni/REAP-OMNI/train.jsonl"
pruning_method=${2:-"reap"}
seed=${3:-42}
compression_ratio=${4:-0.25}

num_samples=1024
output_file_name="observations_${num_samples}_cosine-seed_${seed}.pt"

server_log_file_name="pruning-cli-${FIRST_DEVICE}.log"

echo "Running Qwen3-Omni thinker pruning with dataset: $dataset_name"

python src/reap/prune.py \
    --model_name "$model_name" \
    --dataset_name "$dataset_name" \
    --compression_ratio $compression_ratio \
    --prune_method $pruning_method \
    --profile false \
    --vllm_port $port \
    --server_log_file_name $server_log_file_name \
    --do_eval false \
    --distance_measure cosine \
    --seed $seed \
    --output_file_name ${output_file_name} \
    --singleton_super_experts false \
    --singleton_outlier_experts false \
    --samples_per_category ${num_samples} \
    --record_pruning_metrics_only true
