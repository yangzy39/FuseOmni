# REAP-OMNI 数据集工具包

本目录包含用于 REAP-OMNI 多模态模型剪枝的数据集下载和转换工具，同时支持转换为 **MS-SWIFT** 训练框架所需的格式。

## 📁 文件结构

```
datasets/
├── README.md                     # 本文档
├── DATASETS_CATALOG.md           # 完整数据集目录表格
└── scripts/
    ├── download.sh               # huggingface-cli 批量下载脚本
    ├── process.py                # 数据集处理主入口
    ├── convert_utils.py          # 格式转换工具
    ├── search_hf_datasets.py     # HuggingFace 数据集搜索工具
    └── processors/               # 数据集处理器
        ├── __init__.py           # 处理器注册表
        ├── base.py               # 基类定义
        ├── librispeech.py        # LibriSpeech 处理器
        ├── common_voice.py       # Common Voice 处理器
        ├── gigaspeech.py         # GigaSpeech 处理器
        ├── wavcaps.py            # WavCaps 处理器
        └── aishell.py            # AISHELL-1 处理器
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 设置 HuggingFace 镜像（中国用户推荐）
export HF_ENDPOINT=https://hf-mirror.com

# 设置 HuggingFace Token（部分数据集需要认证）
export HF_TOKEN="your_token_here"

# 安装依赖
pip install datasets soundfile tqdm pandas huggingface_hub pyarrow
```

### 2. 新工作流：下载 → 处理

**推荐使用新的两步工作流：**

```bash
# 步骤 1: 使用 huggingface-cli 下载数据集（保留原始结构）
./scripts/download.sh --datasets librispeech aishell1

# 步骤 2: 使用专用处理器转换为 MS-SWIFT 格式
python scripts/process.py --input ./data --output ./output --merge
```

### 3. 查看支持的数据集

```bash
# 查看可下载的数据集
./scripts/download.sh --list

# 查看可处理的数据集
python scripts/process.py --list
```

### 4. 处理单个数据集

```bash
# 下载单个数据集
./scripts/download.sh --datasets librispeech --output ./data

# 处理单个数据集
python scripts/process.py \
    --input ./data/librispeech \
    --dataset librispeech \
    --output ./output \
    --max-samples 1000
```

### 5. 处理所有已下载数据集

```bash
# 自动检测并处理所有数据集
python scripts/process.py --input ./data --output ./output --merge

# 处理为 GRPO 格式（仅 prompt）
python scripts/process.py --input ./data --output ./output --task grpo
```

## 📊 支持的数据集处理器

| 数据集 | HF ID | 语言 | 任务 | 需认证 |
|--------|-------|------|------|--------|
| librispeech | openslr/librispeech_asr | en | ASR | ❌ |
| common_voice | mozilla-foundation/common_voice_17_0 | en | ASR | ✅ |
| common_voice_zh | mozilla-foundation/common_voice_17_0 | zh | ASR | ✅ |
| gigaspeech | speechcolab/gigaspeech | en | ASR | ✅ |
| aishell1 | AISHELL/AISHELL-1 | zh | ASR | ❌ |
| wavcaps | cvssp/WavCaps | en | Audio Captioning | ❌ |

## 🔄 数据格式

### 处理后输出格式 (MS-SWIFT SFT)

每个处理器将数据转换为 MS-SWIFT 兼容的 JSONL 格式：

```jsonl
{"messages": [{"role": "user", "content": "<audio>Transcribe the following audio exactly as spoken."}, {"role": "assistant", "content": "Hello world."}], "audios": ["/absolute/path/to/audio.wav"]}
```

### 目录结构

处理后的输出结构：

```
output/
├── librispeech/
│   ├── sft.jsonl           # MS-SWIFT 格式数据
│   ├── audio/              # 提取的音频文件
│   └── metadata.json       # 处理元数据
├── aishell1/
│   └── ...
└── all_sft.jsonl           # 合并后的数据（使用 --merge）
```

## 🔧 处理器架构

每个数据集有独立的处理器类，了解其特有的数据格式：

```python
from processors import create_processor, ProcessorConfig

# 创建处理器
processor = create_processor(
    name="librispeech",
    data_dir=Path("./data/librispeech"),
    output_dir=Path("./output"),
    max_samples=1000,
    task_type="sft",
    system_prompt="You are a helpful assistant.",
)

# 执行处理
stats = processor.process()
print(f"Processed {stats['processed']} samples")
```

### 添加新数据集处理器

1. 在 `processors/` 目录创建新文件（如 `my_dataset.py`）
2. 继承 `BaseProcessor` 或 `ParquetProcessor`
3. 实现必需方法：
   - `get_dataset_info()` - 返回数据集元数据
   - `iter_samples()` - 迭代原始样本
   - `process_sample()` - 转换单个样本
4. 在 `__init__.py` 中注册处理器

```python
from .base import ParquetProcessor, Sample

class MyDatasetProcessor(ParquetProcessor):
    def get_dataset_info(self):
        return {"name": "my_dataset", "modality": "audio", ...}
    
    def iter_samples(self):
        for pq_file in self.find_parquet_files():
            yield from self.iter_parquet_rows(pq_file)
    
    def process_sample(self, raw_sample, idx):
        # 转换逻辑
        return Sample(id=..., text=..., audio_path=...)
```

## 🤖 MS-SWIFT 集成

处理后的数据可直接用于 MS-SWIFT 训练：

```bash
# 使用处理后的数据进行 SFT 训练
swift sft \
    --dataset ./output/all_sft.jsonl \
    --model Qwen/Qwen3-Omni-7B-Instruct \
    --output_dir ./sft_output
```

## ⚠️ 注意事项

1. **Common Voice 17.0**: 现在通过 Mozilla Data Collective (MDC) 分发，需要手动下载

2. **音频格式**: 
   - LibriSpeech: FLAC → 自动转换为 WAV
   - Common Voice: MP3（保持原格式或转换）

3. **存储空间**: 完整数据集可能需要数 TB 空间

4. **路径格式**: 输出使用绝对路径，确保跨环境兼容

5. **依赖安装**:
   ```bash
   pip install datasets soundfile pyarrow pandas huggingface_hub
   ```

## 📚 参考资料

- [DATASETS_CATALOG.md](./DATASETS_CATALOG.md) - 完整数据集目录
- [MS-SWIFT 文档](https://swift.readthedocs.io/) - 训练框架指南
- [HuggingFace Hub CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) - 下载工具
