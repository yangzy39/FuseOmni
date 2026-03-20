#!/bin/bash

export CUDA_VISIBLE_DEVICES=${1:-"0,1"}
FIRST_DEVICE=$(echo "$1" | cut -d',' -f1)
port=$((8000 + FIRST_DEVICE))
model_name="/mnt/afs/share/Qwen3-Omni-30B-A3B-Instruct"
dataset_name="/mnt/afs/00036/project_fuseomni/FuseOmni/data/airshell1/aishell1_test_asr.jsonl"
pruning_method=${2:-"reap"}
seed=${3:-42}
compression_ratio=${4:-0.5}

CONDA_PATH="/mnt/afs/00036/software/conda/bin/activate"
CONDA_ENV="reap_omni"
source ${CONDA_PATH} ${CONDA_ENV}

#num_samples=2048
num_samples=1024
output_file_name="observations_${num_samples}_cosine-seed_${seed}.pt"

server_log_file_name="pruning-cli-${FIRST_DEVICE}.log"

echo "Running Qwen3-Omni thinker pruning with dataset: $dataset_name"
#    --results_dir /mnt/afs/00036/project_fuseomni/FuseOmni/reap/outputs \

cd /mnt/afs/00036/project_fuseomni/FuseOmni/reap

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
    --record_pruning_metrics_only false
