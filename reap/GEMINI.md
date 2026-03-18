# REAP (Router-weighted Expert Activation Pruning) 仓库分析指南

## 1. 项目概述

**REAP** 是一个用于大语言模型中**稀疏激活混合专家模型 (SMoE) 压缩**的开源代码库。该仓库实现了论文 [*REAP the Experts: Why Pruning Prevails for One-Shot MoE compression*](https://arxiv.org/abs/2510.13999) 中提出的专家剪枝（Expert Pruning）和专家合并（Expert Merging）方法。

通过 REAP 算法，可以有效减少 SMoE 模型的内存开销。与传统的专家合并方法相比，REAP 综合考虑了**路由门控值（Router gate-values）**和**专家激活范数（Expert activation norms）**作为剪枝准则，在各种规模的 SMoE 模型（20B 到 1T，如 Qwen3, GLM4.5, Kimi, DeepSeek等）上，尤其是在 50% 压缩率下，展现出了近乎无损的性能保留，特别是在代码生成和工具调用等复杂任务上。

## 2. 核心原理与贡献

1. **揭示合并局限性**：从理论上证明了“专家合并”会引入不可约误差（不可逆的合并导致路由器对专家输出的独立调节能力丧失，引发功能子空间坍缩）。而专家剪枝保留了路由器对剩余专家的完全独立控制。
2. **REAP 剪枝准则**：提出了一种创新的专家显著性评价准则。它选择那些对层最终输出贡献最小的专家进行剪枝，计算时结合了路由器的门控权重以及激活值的平均范数。
3. **广泛验证**：支持多种架构及超大规模模型，并兼容如 LM-Eval, EvalPlus, LiveCodeBench, WildBench 等多样化的生成评估基准。

## 3. 核心目录结构分析

```text
reap/
├── config/             # 配置文件，包含用于 vLLM 部署的 WildBench 等评测环境配置
├── experiments/        # 实验执行脚本（合并与剪枝），包含核心的 CLI 入口
│   ├── merging-cli.sh
│   └── pruning-cli.sh
├── scripts/            # 辅助脚本（如 build.sh 用于快速构建和环境安装）
├── src/
│   └── reap/           # 核心源代码目录
│       ├── args.py             # 数据类配置，用于定义和解析各种超参数
│       ├── cluster.py          # 专家聚类算法（层次聚类、动态频率惩罚聚类等）
│       ├── data.py             # 数据集加载和处理逻辑
│       ├── eval.py             # 模型评估与基准测试工具封装
│       ├── main.py             # 实验运行的通用流水线与主程序入口
│       ├── merge.py            # 专家合并（Expert Merging）的具体实现逻辑
│       ├── metrics.py          # 距离和相似度计算（用于分析模型专家冗余度）
│       ├── model_util.py       # 模型工具：各种 SMoE 模型的结构适配器（MODEL_ATTRS）和辅助函数
│       ├── observer.py         # 探针/钩子函数：前向传播截获并记录专家激活频率和路由概率等统计数据
│       ├── permute.py          # 专家权重的排列与对齐逻辑
│       ├── prune.py            # [核心] 专家剪枝的实现入口及逻辑（含裁剪模型参数和保存逻辑）
│       └── models/             # 针对部分特定架构（未通过标准 forward 返回 router_logits 的模型）的魔改和兼容代码（如 GLM, ERNIE）
├── third-party/        # 第三方依赖库库源码（LiveCodeBench, evalplus, evalscope 等）
├── pyproject.toml      # 项目包与依赖声明 (Python >= 3.12, transformers=4.55等)
└── Dockerfile & docker-compose.yaml # 容器化运行环境配置
```

## 4. 关键代码解析 (`src/reap/`)

- **数据收集阶段 (`observer.py` & `main.py`)**：通过 `record_activations` 函数向模型内部诸如 Router 和 Expert 层注入 Hook，收集输入数据在各层各专家上的激活频率（Frequency）和激活表现（Activation Norm等）。
- **剪枝判断与裁剪维度 (`prune.py`)**：负责读入统计数据。基于所选的策略进行重要度排序（比如按频率或我们提出的 `reap` 评分排序）。然后通过保留前 K 大最重要的专家索引，**直接对 PyTorch 模型内的 `torch.nn.ModuleList` （对应专家集合）进行切片截断**，同时缩减 `router.weight` （只保留对应保留专家的权重部分），从而达到物理压缩的目的。并支持排除或保护极端专家（Super/Outlier Experts）。
- **多模型适配 (`model_util.py`)**：维护了一个重要的 `MODEL_ATTRS` 字典。由于各厂商的开源 MoE 模型字段命名各不相同，字典通过模型的 `__class__.__name__` 映射专家的模块名、路由器的模块名、投影矩阵等元数据。使得复用一套裁剪逻辑适应如 `Qwen2_5_MoE` 或 `Ernie` 或 `Llama-4` 变得可行。
- **专家合并 (`merge.py` & `cluster.py`)**：实现了作为对比的合并策略（如 `hc_smoe` 等）。利用聚类算法对专家知识进行合并计算重组。

## 5. 环境配置与使用指南

### 环境依赖
项目基于 Python >= 3.12 并使用 `uv` / `hatchling` 进行包管理。你可以通过以下脚本快速初始化 `venv`：
```bash
bash scripts/build.sh
```
或者使用 Docker 挂载 HuggingFace 模型缓存进行一键启动：
```bash
docker compose up --build -d
docker compose exec app bash
```

### 如何添加新的 MoE 模型支持
1. 在 `src/reap/model_util.py` 的 `MODEL_ATTRS` 添加模型类名。
2. 配置好 `moe_block`, `experts`, `router` 等在 HuggingFace `modeling_*.py` 中的特定属性名。
3. （必要时）在 `src/reap/models` 内补充打 Patch 的建模代码以支持读取 router_logits。

### 核心实验执行

**执行专家合并 (Expert Merging):**
```bash
bash experiments/merging-cli.sh <CUDA_DEVICES> [MODEL_NAME] [MERGE_METHOD] [SEED] [COMPRESSION_RATIO] [DATASET_NAME] [RUN_LM_EVAL=true/false] ...
```

**执行专家剪枝 (Expert Pruning) - 即本文的核心方法 REAP:**
```bash
bash experiments/pruning-cli.sh <CUDA_DEVICES> [MODEL_NAME] [PRUNING_METHOD] [SEED] [COMPRESSION_RATIO] [DATASET_NAME] [RUN_LM_EVAL=true/false] ...
```
示例：对 `Qwen3-30B-A3B` 模型进行频率为基础的剪枝：
```bash
bash experiments/pruning-cli.sh 0 Qwen/Qwen3-30B-A3B reap 42 0.25 theblackcat102/evol-codealpaca-v1 true true true false false
```
*(注意 `PRUNING_METHOD` 设置为 `reap` 即可调用官方提出的综合考虑特征的裁剪算法)*
