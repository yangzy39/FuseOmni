# REAP-OMNI: 多模态大模型剪枝工具包

<p align="center">
  <img src="https://img.shields.io/badge/Model-Qwen3--Omni--30B-blue" alt="Model">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="Framework">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
</p>

基于 REAP-OMNI 实现的 **Qwen3-Omni-30B-A3B** 多模态 MoE 模型剪枝工具包，支持三种剪枝策略：

- 🎯 **视觉模态剥离** - 完全移除视觉编码器和投影层
- 🔧 **REAP 专家剪枝** - 基于音频亲和度的 MoE 专家剪枝
- 📊 **层间相似度剪枝** - 移除冗余的 Transformer 层

## 📋 目录

- [概述](#概述)
- [算法原理](#算法原理)
- [安装](#安装)
- [快速开始](#快速开始)
- [详细使用](#详细使用)
- [文件结构](#文件结构)
- [配置参数](#配置参数)
- [示例](#示例)
- [引用](#引用)

## 概述

### 背景

Qwen3-Omni-30B-A3B 是一个支持文本、音频、视频多模态输入的大型 MoE（Mixture of Experts）模型：

| 组件 | 规格 |
|------|------|
| **视觉编码器** | 27 层, hidden_size=1152, patch_size=16 |
| **音频编码器** | 32 层, d_model=1280 |
| **Thinker (主LLM)** | 48 层, 128 专家, 每token激活8个专家 |
| **Talker (语音合成)** | 20 层, 128 专家, 每token激活6个专家 |
| **总参数量** | ~35B |

### 目标

本工具包旨在将 Qwen3-Omni 压缩为**纯音频模型**，通过：

1. **移除视觉模态** → 减少视觉编码器和投影层参数
2. **剪枝视觉相关专家** → 保留音频相关的 MoE 专家
3. **移除冗余层** → 基于层间相似度剪枝

## 算法原理

### 1. 视觉模态剥离

直接从模型权重中移除所有视觉相关组件：

```
移除的权重模式：
├── thinker.visual.patch_embed.*      # 视觉 Patch 嵌入
├── thinker.visual.blocks.*           # 视觉 Transformer 块
├── thinker.visual.merger.*           # 视觉投影层
└── thinker.visual.deepstack_*        # 深度堆叠视觉特征
```

### 2. REAP 专家剪枝

**REAP (Router-weighted Expert Activation Pruning)** 通过计算专家的音频亲和度来识别和保留音频相关专家。

#### 专家显著性公式

$$S(e, D) = \frac{1}{|D|} \sum_{x \in D} (g_e(x) \cdot \|h_e(x)\|_2)$$

- $g_e(x)$: 路由器分配给专家 $e$ 的门控权重
- $\|h_e(x)\|_2$: 专家 $e$ 输出的 L2 范数

#### 音频亲和度分数

$$\mathcal{A}_{audio}(e) = S_1(e) + \lambda \cdot \text{ReLU}(S_3(e) - \beta \cdot S_2(e))$$

| 符号 | 含义 | 默认值 |
|------|------|--------|
| $S_1$ | 纯音频数据上的显著性 | - |
| $S_2$ | 纯视频数据上的显著性 | - |
| $S_3$ | 混合数据上的显著性 | - |
| $\lambda$ | 混合数据权重 | 1.0 |
| $\beta$ | 视频去噪系数 | 1.0 |

#### 剪枝流程

```
1. 对三种数据类型运行校准推理
2. 计算每个专家的 S1, S2, S3
3. 计算音频亲和度 A_audio
4. 按 A_audio 降序排列专家
5. 保留 Top-K 专家 (如 50%)
6. 从权重文件中移除被剪枝的专家
```

### 3. 层间相似度剪枝

基于相邻层隐藏状态的相似度识别冗余层。**使用与 REAP 第二步相同的音频校准数据进行 forward pass，收集真实的 hidden states 用于计算层间相似度。**

$$\text{similarity}(H_l, H_{l+1}) = \frac{H_l \cdot H_{l+1}}{\|H_l\| \cdot \|H_{l+1}\|}$$

#### 层剪枝流程

```
1. 加载模型到 GPU
2. 加载音频校准数据（与 REAP step 2 相同格式）
3. 注册 forward hooks 在每个 decoder layer
4. 对校准数据进行 forward pass，收集每层的 hidden states
5. 计算相邻层之间的 cosine similarity
6. 选择相似度超过阈值的层作为剪枝候选
7. 移除冗余层并重新编号剩余层
```

相似度超过阈值（如 0.9）的层被视为冗余层候选。

## 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (推荐)

### 安装依赖

```bash
pip install torch safetensors tqdm transformers
```

### 克隆仓库

```bash
git clone https://github.com/your-repo/REAP-OMNI.git
cd REAP-OMNI
```

## 快速开始

### 一键运行全部剪枝

**Windows:**
```batch
run_pruning.bat --model-path ..\models\Qwen3-Omni-30B-A3B-Instruct
```

**Linux/Mac/WSL:**
```bash
chmod +x run_pruning.sh
./run_pruning.sh --model-path ../models/Qwen3-Omni-30B-A3B-Instruct
```

### Dry Run 预览

```bash
# 预览将要执行的操作，不实际修改文件
python vision_strip.py --dry-run
python reap_expert_pruning.py --dry-run
python layer_similarity_pruning.py --dry-run
```

## 详细使用

### 1. 视觉模态剥离

```bash
python vision_strip.py \
    --model-path ../models/Qwen3-Omni-30B-A3B-Instruct \
    --output-path ../models/Qwen3-Omni-Audio-Only \
    --verbose
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-path`, `-m` | 原始模型路径 | - |
| `--output-path`, `-o` | 输出模型路径 | - |
| `--dry-run` | 仅分析不修改 | False |
| `--device` | 计算设备 | cuda |
| `--no-copy-unaffected` | 不复制未修改的分片 | False |

### 2. REAP 专家剪枝

```bash
python reap_expert_pruning.py \
    --model-path ../models/Qwen3-Omni-30B-A3B-Instruct \
    --output-path ../models/Qwen3-Omni-REAP-50 \
    --component thinker \
    --retention-rate 0.5 \
    --verbose
```

**使用校准数据：**

```bash
python reap_expert_pruning.py \
    --model-path ../models/Qwen3-Omni-30B-A3B-Instruct \
    --output-path ../models/Qwen3-Omni-REAP-50 \
    --audio-data ./calibration/audio.jsonl \
    --video-data ./calibration/video.jsonl \
    --mixed-data ./calibration/mixed.jsonl \
    --retention-rate 0.5
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--component` | 剪枝组件 (thinker/talker) | thinker |
| `--retention-rate`, `-r` | 专家保留比例 (0.0-1.0) | 0.5 |
| `--lambda-weight` | 混合数据权重 λ | 1.0 |
| `--beta-weight` | 视频去噪系数 β | 1.0 |
| `--audio-data` | 音频校准数据路径 (JSONL) | None |
| `--video-data` | 视频校准数据路径 (JSONL) | None |
| `--mixed-data` | 混合校准数据路径 (JSONL) | None |
| `--calibration-samples` | 每种模态的校准样本数 | 100 |

### 3. 层间相似度剪枝

**推荐方式：使用音频校准数据（与 REAP step 2 相同）**

```bash
python layer_similarity_pruning.py \
    --model-path ../models/Qwen3-Omni-30B-A3B-Instruct \
    --output-path ../models/Qwen3-Omni-Layer-Pruned \
    --audio-data ./calibration/audio.jsonl \
    --component thinker \
    --similarity-threshold 0.9 \
    --max-layers 8 \
    --verbose
```

**Dry Run 模式：仅查看层间相似度，不执行剪枝**

```bash
python layer_similarity_pruning.py \
    --audio-data ./calibration/audio.jsonl \
    --dry-run
```

**静态模式：手动指定要剪枝的层（无需加载模型）**

```bash
python layer_similarity_pruning.py \
    --static \
    --prune-layers 12 16 20 24 28 32
```

**参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--component` | 剪枝组件 (thinker/talker) | thinker |
| `--audio-data` | 音频校准数据路径 (JSONL，与 REAP step 2 相同格式) | None |
| `--calibration-samples` | 使用的校准样本数 | 32 |
| `--similarity-threshold`, `-t` | 相似度阈值 | 0.9 |
| `--max-layers` | 最大剪枝层数 | 8 |
| `--prune-layers` | 手动指定要剪枝的层索引 | None |
| `--protect-first` | 保护前 N 层不剪枝 | 4 |
| `--protect-last` | 保护后 N 层不剪枝 | 4 |
| `--layers-to-skip` | 比较间隔 (1=相邻层) | 1 |
| `--static` | 静态模式：不加载模型，需配合 --prune-layers | False |
| `--device` | 模型加载设备 | cuda |
| `--dtype` | 模型精度 (bfloat16/float16/float32) | bfloat16 |

## 文件结构

```
REAP-OMNI/
├── README.md                      # 本文档
├── reap-omni.pdf                  # 参考论文
├── vision_strip.py                # 视觉模态剥离
├── reap_expert_pruning.py         # REAP 专家剪枝
├── layer_similarity_pruning.py    # 层间相似度剪枝
├── run_pruning.sh                 # Linux/Mac 执行脚本
└── run_pruning.bat                # Windows 执行脚本
```

## 配置参数

### 校准数据格式

校准数据使用 JSONL 格式，每行一个样本：

```json
{"id": "sample_001", "text": "音频转录文本或描述", "modality": "audio"}
{"id": "sample_002", "text": "视频描述文本", "modality": "video"}
{"id": "sample_003", "text": "音视频混合描述", "modality": "mixed"}
```

### 推荐配置

| 压缩目标 | 视觉剥离 | 专家保留率 | 层剪枝数 | 预计压缩比 |
|----------|----------|------------|----------|------------|
| 轻度压缩 | ✓ | 75% | 4 | ~30% |
| 中度压缩 | ✓ | 50% | 8 | ~50% |
| 激进压缩 | ✓ | 25% | 12 | ~70% |

## 示例

### 完整流水线示例

```bash
#!/bin/bash
# 完整的 REAP-OMNI 压缩流水线

MODEL_PATH="./models/Qwen3-Omni-30B-A3B-Instruct"
OUTPUT_BASE="./models"

# Step 1: 视觉模态剥离
echo "Step 1: Stripping vision modality..."
python vision_strip.py \
    --model-path $MODEL_PATH \
    --output-path $OUTPUT_BASE/step1-vision-stripped

# Step 2: REAP 专家剪枝 (保留50%专家)
echo "Step 2: REAP expert pruning..."
python reap_expert_pruning.py \
    --model-path $OUTPUT_BASE/step1-vision-stripped \
    --output-path $OUTPUT_BASE/step2-reap-pruned \
    --retention-rate 0.5

# Step 3: 层剪枝（使用音频校准数据计算真实相似度）
echo "Step 3: Layer similarity pruning..."
python layer_similarity_pruning.py \
    --model-path $OUTPUT_BASE/step2-reap-pruned \
    --output-path $OUTPUT_BASE/final-compressed \
    --audio-data ./calibration/audio.jsonl \
    --max-layers 8 \
    --similarity-threshold 0.9

echo "Done! Compressed model saved to: $OUTPUT_BASE/final-compressed"
```

### Python API 使用

```python
from vision_strip import VisionWeightStripper
from reap_expert_pruning import REAPExpertPruner, REAPConfig
from layer_similarity_pruning import LayerSimilarityPruner, LayerPruningConfig

# 1. 视觉剥离
stripper = VisionWeightStripper(
    model_path="./models/Qwen3-Omni-30B-A3B-Instruct",
    output_path="./models/vision-stripped"
)
stats = stripper.strip_vision_weights()
print(f"Removed {stats['vision_weights']} vision weights")

# 2. REAP 专家剪枝
config = REAPConfig(
    model_path="./models/vision-stripped",
    output_path="./models/reap-pruned",
    retention_rate=0.5,
    component="thinker"
)
pruner = REAPExpertPruner(config)
stats = pruner.run_static_pruning()
print(f"Kept {stats['weights_to_keep']} weights")

# 3. 层剪枝（使用音频校准数据）
config = LayerPruningConfig(
    model_path="./models/reap-pruned",
    output_path="./models/layer-pruned",
    audio_data_path="./calibration/audio.jsonl",
    similarity_threshold=0.9,
    max_layers_to_prune=8
)
pruner = LayerSimilarityPruner(config)
stats = pruner.run_with_calibration()  # 加载模型，forward pass，计算真实相似度
print(f"Pruned {stats['layers_pruned']} layers")
```

## 注意事项

1. **磁盘空间**: 每个剪枝步骤会创建新的模型副本，确保有足够的磁盘空间
2. **内存需求**: 处理大型 safetensor 分片时需要足够的 RAM
3. **GPU 显存**: 层剪枝使用校准模式时需要加载完整模型，建议使用 80GB+ 显存的 GPU
4. **备份原模型**: 建议在剪枝前备份原始模型
5. **验证输出**: 剪枝后建议运行推理测试验证模型功能
6. **校准数据复用**: 层剪枝使用与 REAP step 2 相同的音频校准数据格式，可复用

## 引用

如果您使用了本工具包，请引用：

```bibtex
@article{reap-omni,
  title={REAP-OMNI: Multimodal Model Pruning for Audio-focused Applications},
  author={...},
  year={2025}
}
```

## 参考资料

- [REAP: Cerebras Research](https://github.com/CerebrasResearch/reap)
- [PruneMe: Layer Similarity Pruning](https://github.com/arcee-ai/PruneMe)
- [FlowCut: Vision Token Pruning](https://github.com/TungChintao/FlowCut)
- [Qwen3-Omni Official](https://github.com/QwenLM/Qwen)

## License

Apache License 2.0
