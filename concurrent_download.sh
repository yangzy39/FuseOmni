#!/bin/bash

# ModelScope 并发下载脚本
# 用法: ./concurrent_download.sh [并发数]
# 示例: ./concurrent_download.sh 4

# 设置并发数 (默认为 4)
CONCURRENCY=${1:-4}
SAVE_DIR="./weights"

# 在此处定义要下载的模型列表
MODELS=(
    "Qwen/Qwen2.5-0.5B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct" 
    # 添加更多模型...
)

# 检查 modelscope 是否安装
if ! command -v modelscope &> /dev/null; then
    echo "Error: modelscope CLI 未找到。请先运行: pip install modelscope"
    exit 1
fi

echo "Start downloading ${#MODELS[@]} models with concurrency $CONCURRENCY..."
echo "Models will be saved to: $SAVE_DIR"

# 创建下载函数
download_one() {
    local model_id=$1
    local save_dir=$2
    
    # 获取模型名称用于目录名 (移除组织名前缀)
    local model_name=$(basename "$model_id")
    local target_dir="$save_dir/$model_name"
    
    echo "[Started] $model_id -> $target_dir"
    
    # 执行下载
    # modelscope download --model <model_id> --local_dir <dir>
    modelscope download --model "$model_id" --local_dir "$target_dir" > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "[Success] $model_id"
    else
        echo "[Failed]  $model_id"
    fi
}

export -f download_one
export SAVE_DIR

# 创建保存目录
mkdir -p "$SAVE_DIR"

# 使用 xargs 进行并发下载
printf "%s\n" "${MODELS[@]}" | xargs -P "$CONCURRENCY" -I {} bash -c 'download_one "{}" "$SAVE_DIR"'

echo "All tasks finished."
