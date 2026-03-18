# Qwen3-Omni 训练脚本

本目录包含用于训练 **Qwen3-Omni-30B-A3B-Instruct** 模型的 bash 脚本，支持 SFT（监督微调）和 GRPO（强化学习）训练。

## 📁 脚本列表

| 脚本 | 训练类型 | 微调方式 | GPU 需求 | 说明 |
|------|----------|----------|----------|------|
| `qwen3_omni_sft_lora.sh` | SFT | LoRA | 4x A100 80GB | **推荐**，内存效率高 |
| `qwen3_omni_sft_full.sh` | SFT | Full | 8x A100 80GB | 全参数微调 |
| `qwen3_omni_grpo.sh` | GRPO | LoRA | 4x A100 80GB | 强化学习训练 |
| `qwen3_omni_grpo_vllm.sh` | GRPO | LoRA + vLLM | 4x A100 80GB | 加速版 GRPO |

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install transformers>=4.57 soundfile decord qwen_omni_utils
pip install ms-swift -U

# GRPO 需要额外安装
pip install math_verify trl

# vLLM 加速版需要
pip install vllm>=0.5.1
```

### 2. 准备数据

数据格式请参考 [datasets/README.md](../../datasets/README.md) 中的 MS-SWIFT 格式说明。

**SFT 数据示例** (`sft_data.jsonl`):
```jsonl
{"messages": [{"role": "user", "content": "<audio>What did the audio say?"}, {"role": "assistant", "content": "The speaker said hello."}], "audios": ["/absolute/path/to/audio.wav"]}
{"messages": [{"role": "user", "content": "<video>Describe this video"}, {"role": "assistant", "content": "A cat is playing with a ball."}], "videos": ["/absolute/path/to/video.mp4"]}
{"messages": [{"role": "user", "content": "<image>What is in this image?"}, {"role": "assistant", "content": "This is a cute dog."}], "images": ["/absolute/path/to/image.jpg"]}
```

**GRPO 数据示例** (`grpo_prompts.jsonl`):
```jsonl
{"messages": [{"role": "user", "content": "<image>Solve this math problem step by step."}], "images": ["/path/to/math_problem.jpg"]}
{"messages": [{"role": "user", "content": "<audio>Transcribe and summarize this audio."}], "audios": ["/path/to/audio.wav"]}
{"messages": [{"role": "user", "content": "What is the capital of France?"}]}
```

### 3. 修改脚本配置

编辑脚本中的配置部分：

```bash
# 修改数据集路径
DATASET="/path/to/your/sft_data.jsonl"

# 修改输出目录
OUTPUT_DIR="./output/my_experiment"

# 根据 GPU 数量修改
CUDA_DEVICES="0,1,2,3"
NPROC_PER_NODE=4
```

### 4. 运行训练

```bash
# SFT LoRA 训练（推荐）
bash qwen3_omni_sft_lora.sh

# SFT 全参数训练
bash qwen3_omni_sft_full.sh

# GRPO 训练
bash qwen3_omni_grpo.sh

# GRPO + vLLM 加速
bash qwen3_omni_grpo_vllm.sh
```

## 📋 数据格式详解

### 多模态标记

在 `content` 字段中使用以下标记指定媒体插入位置：

| 标记 | 对应字段 | 示例 |
|------|----------|------|
| `<image>` | `images` | `"content": "<image>Describe this image"` |
| `<video>` | `videos` | `"content": "<video>What happens in this video?"` |
| `<audio>` | `audios` | `"content": "<audio>Transcribe this audio"` |

### 多媒体支持

```jsonl
# 单图片
{"messages": [...], "images": ["/path/to/image.jpg"]}

# 多图片
{"messages": [{"role": "user", "content": "<image><image>Compare these two images"}], "images": ["/path/img1.jpg", "/path/img2.jpg"]}

# 混合模态
{"messages": [{"role": "user", "content": "<image><audio>Describe the image and transcribe the audio"}], "images": ["/path/img.jpg"], "audios": ["/path/audio.wav"]}
```

### 消息角色

| 角色 | 说明 | 是否必需 |
|------|------|----------|
| `system` | 系统提示，设定模型行为 | 可选 |
| `user` | 用户输入/问题 | **必需** |
| `assistant` | 模型回复（SFT必需，GRPO不需要） | SFT 必需 |

## ⚙️ 关键参数说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `MAX_PIXELS` | 图片最大像素数 | 1003520 |
| `VIDEO_MAX_PIXELS` | 视频帧最大像素数 | 50176 |
| `FPS_MAX_FRAMES` | 视频最大帧数 | 12 |
| `ENABLE_AUDIO_OUTPUT` | 是否启用音频输出 | 0 (SFT) / 1 (GRPO) |

### 训练参数

| 参数 | SFT LoRA | SFT Full | GRPO |
|------|----------|----------|------|
| `--tuner_type` | lora | full | lora |
| `--lora_rank` | 8 | - | 8 |
| `--learning_rate` | 1e-4 | 1e-5 | 1e-5 |
| `--deepspeed` | zero2 | zero3 | zero2 |
| `--freeze_vit` | true | true | - |
| `--freeze_aligner` | true | true | - |

### GRPO 特有参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--num_generations` | 每个 prompt 生成的回复数 | 4-8 |
| `--reward_funcs` | 奖励函数 | format, external_r1v_acc |
| `--temperature` | 采样温度 | 1.0 |
| `--use_vllm` | 是否使用 vLLM 加速 | true |
| `--vllm_mode` | vLLM 部署模式 | colocate |

## 🔧 自定义奖励函数（GRPO）

创建 `reward_plugin.py`:

```python
def custom_reward(completions, **kwargs):
    """
    自定义奖励函数
    
    Args:
        completions: 模型生成的回复列表
        **kwargs: 数据集中的额外字段
        
    Returns:
        奖励分数列表
    """
    rewards = []
    for completion in completions:
        # 你的奖励逻辑
        score = 1.0 if "correct" in completion.lower() else 0.0
        rewards.append(score)
    return rewards
```

在训练脚本中添加：

```bash
--external_plugins reward_plugin.py \
--reward_funcs custom_reward
```

## 📊 训练后使用

### 推理

```bash
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters ./output/qwen3_omni_sft_lora/checkpoint-xxx \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

### 合并 LoRA

```bash
swift export \
    --adapters ./output/qwen3_omni_sft_lora/checkpoint-xxx \
    --merge_lora true \
    --output_dir ./merged_model
```

### 部署

```bash
CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --adapters ./output/qwen3_omni_sft_lora/checkpoint-xxx \
    --infer_backend vllm
```

## ❓ 常见问题

### Q: 显存不足怎么办？

1. 减小 `--per_device_train_batch_size`
2. 增大 `--gradient_accumulation_steps`
3. 减小 `MAX_PIXELS` 和 `VIDEO_MAX_PIXELS`
4. 使用 LoRA 而非全参数微调
5. 使用 DeepSpeed ZeRO3

### Q: 训练速度太慢？

1. 增加 `--dataloader_num_workers`
2. 启用 `--load_from_cache_file true`
3. GRPO 使用 vLLM 加速版脚本
4. 考虑使用 Megatron 并行训练

### Q: 如何使用多机训练？

参考 `ms-swift/examples/train/grpo/multi_node/` 中的多机训练脚本。

## 📚 参考资料

- [MS-SWIFT 官方文档](https://swift.readthedocs.io/)
- [GRPO 训练指南](https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html)
- [Qwen3-Omni 模型](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
- [数据格式转换工具](../../datasets/README.md)
