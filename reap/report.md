# REAP 仓库 Qwen3-Omni 剪枝支持实验记录

## 1. 仓库核心用法解析

`REAP` (Router-weighted Expert Activation Pruning) 旨在通过分析专家激活分布，对 Sparsely-activated Mixture-of-Experts (SMoE) 模型进行压缩。

*   **激活收集 (Observation)**：通过在模型 MoE 层注册 Forward Hook，统计专家在特定数据集上的激活频率、激活范数 (EAN) 以及专家间的相似度。
*   **专家剪枝 (Pruning)**：基于显著性指标（如激活频率或激活幅度）移除不重要的专家。本项目主要使用 `reap` 算法进行显著性评估。
*   **专家合并 (Merging)**：基于聚类算法（如 MC-SMoE）将相似的专家参数进行合并，从而减少模型大小。
*   **评估 (Evaluation)**：提供对剪枝后模型的下游任务能力验证。

## 2. 仓库代码功能分解

*   `src/reap/main.py`: 核心调度脚本。负责初始化模型与分词器、加载数据集、启动观察者收集数据、执行聚类合并决策。
*   `src/reap/observer.py`: 定义了 `MoETransformerObserver` 以及针对不同架构的 `HookConfig`。它是收集统计数据（如 `ean_sum`, `expert_frequency`）的底层实现。
*   `src/reap/model_util.py`: 模型适配层。提供 `MODEL_ATTRS` 字典，记录不同模型 MoE 模块的内部属性路径（如 `mlp`, `gate`, `experts`）。
*   `src/reap/data.py`: 数据预处理模块。包含各类 `DatasetProcessor`，负责将原始数据转换为符合模型 Chat Template 的 Token 序列。
*   `src/reap/prune.py`: 剪枝逻辑入口，支持直接通过显著性排序移除专家并更新模型权重。

## 3. Omni 模型剪枝实现路径

针对 `Qwen3-Omni-30B-A3B-Instruct` 模型的特殊性，本次修改重点在于支持其 `thinker` 模块的文本专家剪枝。

### 3.1 模型架构适配
*   **模型类支持**：`Qwen3-Omni` 属于 `Qwen3OmniMoeForConditionalGeneration` 类，标准 `AutoModelForCausalLM` 无法加载。
*   **路径补丁**：在 `model_util.py` 中新增 `MODEL_ATTRS` 条目，并将 `get_moe` 函数修改为支持 `model.thinker.model.layers` 的多级路径，以精准定位 `thinker` 模块。
*   **Hook 注入**：在 `observer.py` 中新增 `Qwen3OmniMoEObserverHookConfig`，将 Hook 绑定至 `Qwen3OmniMoeThinkerTextSparseMoeBlock`。

### 3.2 自定义数据集加载
*   **多模态解析**：针对 `train.jsonl`，在 `data.py` 中新增 `FuseOmniChatDataset` 类。其生成的 `_map_fn` 可自动将原始 JSON 里的 `audio_path` 和 `text` 字段转化为符合 QwenTokenizer 预期的内容列表。
*   **本地文件支持**：修改 `main.py` 的加载逻辑，当检测到以 `.jsonl` 结尾的 `--dataset_name` 时，自动采用 `load_dataset('json')` 进行本地加载。

### 3.3 执行脚本
*   创建 `experiments/prune-qwen3-omni.sh`。该脚本封装了复杂的参数传递，允许用户通过简单的单一命令执行针对 `thinker` 模块的剪枝流程。

## 4. 遇到的 Bug 与解决方案

| 遇到的问题 | 根本原因 | 解决方案 |
| :--- | :--- | :--- |
| **AutoModel 加载失败** | `transformers` 的 `AutoModel` 映射中未包含该特定类。 | 显式从 `transformers` 导入 `Qwen3OmniMoeForConditionalGeneration` 并使用 `.from_pretrained`。 |
| **MoE 模块获取为空** | 默认代码检索 `model.model.layers`，而 Omni 模型封装了 `thinker` 层。 | 修改 `model_util.py` 中的 `get_moe` 逻辑，增加对 `thinker` 成员的判断。 |
| **模块类名匹配失效** | 模型层实际类名与通用 Qwen 架构略有不同。 | 使用 Python 探针脚本确认各层 Class Name，将 `HookConfig` 中的正则匹配更新为 `Qwen3OmniMoeThinkerTextSparseMoeBlock`。 |
| **JSONL 解析异常** | 原始 JSONL 里的 `messages` 结构是多模态子列表形式。 | 在 `data.py` 中实现定制化的 `_map_fn` 逻辑，显式转换 `audio_path` 路径映射。 |
| **环境依赖缺失** | 运行环境缺少 `datasets` 或 `accelerate`。 | 记录并提示用户补全必要的 Python 库依赖。 |

---
**实验结论**：通过上述路径，REAP 现在能够正确识别多模态模型的思维 (Thinker) 核心并进行专家显著性分析，成功将传统文本 MoE 剪枝能力扩展至多模态混合模型。
