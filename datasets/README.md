# REAP-OMNI 数据集工具包

本目录包含用于 REAP-OMNI 多模态模型剪枝的数据集下载和转换工具，同时支持转换为 **MS-SWIFT** 训练框架所需的格式。

## 📁 文件结构

```
datasets/
├── README.md                 # 本文档
├── DATASETS_CATALOG.md       # 完整数据集目录表格
├── download_datasets.py      # 主数据集下载脚本
├── extended_datasets.py      # 扩展数据集支持
├── convert_utils.py          # 格式转换工具（支持 MS-SWIFT 格式）
└── quickstart.py             # 快速开始脚本
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install datasets soundfile tqdm pandas
```

### 2. 查看支持的数据集

```bash
python download_datasets.py --list
```

### 3. 下载推荐数据集

```bash
python quickstart.py --output ./calibration_data --samples 100
```

### 4. 下载特定数据集

```bash
# 下载单个数据集
python download_datasets.py --dataset librispeech --output ./data --samples 100

# 下载多个数据集
python download_datasets.py --dataset librispeech gigaspeech common_voice --output ./data

# 按模态下载
python download_datasets.py --modality audio --output ./data
python download_datasets.py --modality video --output ./data
python download_datasets.py --modality mixed --output ./data
```

## 📊 支持的数据集

### Audio-only (S1) - 用于计算纯音频专家亲和度

| 数据集 | 规模 | 说明 |
|--------|------|------|
| LibriSpeech | 960h | 英语有声读物 |
| Common Voice | 19K+h | 多语言众包 |
| GigaSpeech | 10K+h | 多领域英语 |
| VoxPopuli | 400K+h | 欧洲议会多语言 |
| WenetSpeech | 10K+h | 中文多领域 |
| WavCaps | 400K clips | 音频描述 |
| AISHELL-1 | 170h | 中文普通话 |
| CoVoST2 | 2.9K h | 语音翻译 |

### Video-only (S2) - 用于计算纯视频专家亲和度

| 数据集 | 规模 | 说明 |
|--------|------|------|
| Kinetics-400/700 | 306K/650K clips | 动作识别 |
| MSR-VTT | 10K clips | 视频描述 |
| VATEX | 41K clips | 多语言视频描述 |
| YouCook2 | 2K videos | 烹饪教学 |
| LongVideoBench | 3.7K videos | 长视频理解 |
| ActivityNet-QA | 58K QA | 视频问答 |

### Mixed (S3) - 用于音视频联合校准

| 数据集 | 规模 | 说明 |
|--------|------|------|
| VoxCeleb | 1M+ utterances | 说话人识别 |
| LRS2/LRS3 | 数千句子 | 音视频语音识别 |
| How2 | 80K clips | 教学视频 |
| AudioSet | 2M+ clips | 音频事件 |
| VGGSound | 210K videos | 音视频对应 |
| MELD | TV episodes | 情感对话 |
| HowTo100M | 136M clips | 大规模教学 |

## 🔄 格式转换

### 统一输出格式（REAP-OMNI）

所有数据集都会转换为统一的 JSONL 格式：

```json
{
    "id": "librispeech_00001",
    "text": "转录文本或描述",
    "audio": "/path/to/audio.wav",
    "video": "/path/to/video.mp4",
    "modality": "audio"
}
```

### 基础转换命令

```bash
# CSV 转 JSONL
python convert_utils.py csv input.csv output.jsonl --text-col caption --audio-col path

# JSON 转 JSONL
python convert_utils.py json input.json output.jsonl --text-key text --video-key video_path

# 文件夹转 JSONL
python convert_utils.py folder ./my_data output.jsonl --name my_dataset

# 合并多个 JSONL
python convert_utils.py merge audio1.jsonl audio2.jsonl -o merged.jsonl

# 按模态分割
python convert_utils.py split all_data.jsonl -o ./split_output

# 验证格式
python convert_utils.py validate calibration/audio.jsonl
```

---

## 🤖 MS-SWIFT 格式转换

支持将数据转换为 [MS-SWIFT](https://github.com/modelscope/ms-swift) 训练框架所需的格式，用于 Qwen3-Omni 等多模态模型的 SFT 和 GRPO 训练。

### MS-SWIFT 数据格式说明

#### SFT 训练格式（包含完整问答对）

```jsonl
{"messages": [{"role": "user", "content": "<audio>What did the audio say?"}, {"role": "assistant", "content": "The speaker said hello."}], "audios": ["/path/to/audio.wav"]}
{"messages": [{"role": "user", "content": "<video>Describe this video"}, {"role": "assistant", "content": "A cat is playing."}], "videos": ["/path/to/video.mp4"]}
{"messages": [{"role": "user", "content": "<image>What is in this image?"}, {"role": "assistant", "content": "This is a dog."}], "images": ["/path/to/image.jpg"]}
{"messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]}
```

#### GRPO 训练格式（仅包含提示）

```jsonl
{"messages": [{"role": "user", "content": "<image>Solve this math problem"}], "images": ["/path/to/image.jpg"]}
{"messages": [{"role": "user", "content": "<audio>Transcribe and answer"}], "audios": ["/path/to/audio.wav"]}
{"messages": [{"role": "user", "content": "What is 1+1?"}]}
```

#### 格式关键规则

| 元素 | 说明 |
|------|------|
| `<image>` | 在 content 中标记图片插入位置 |
| `<video>` | 在 content 中标记视频插入位置 |
| `<audio>` | 在 content 中标记音频插入位置 |
| `images` | 图片文件路径列表（建议使用绝对路径） |
| `videos` | 视频文件路径列表 |
| `audios` | 音频文件路径列表 |

### 转换命令

```bash
# 将统一格式转换为 MS-SWIFT SFT 格式
python convert_utils.py msswift input.jsonl output_sft.jsonl --task sft

# 将统一格式转换为 MS-SWIFT GRPO 格式（只保留prompt）
python convert_utils.py msswift input.jsonl output_grpo.jsonl --task grpo

# 添加系统提示
python convert_utils.py msswift input.jsonl output.jsonl --task sft --system "You are a helpful assistant."

# 自定义用户消息模板
python convert_utils.py msswift input.jsonl output.jsonl --task sft --user-template "{modality_tag}Please describe: {text}"

# 将 QA 格式数据转换为 MS-SWIFT 格式
python convert_utils.py qa-msswift qa_data.jsonl output.jsonl \
    --question-key question \
    --answer-key answer \
    --image-key image \
    --audio-key audio

# 验证 MS-SWIFT 格式
python convert_utils.py validate-msswift output.jsonl
```

### Python API 使用

```python
from convert_utils import (
    convert_unified_to_msswift,
    convert_qa_to_msswift,
    create_msswift_sample,
    validate_msswift_format
)

# 批量转换统一格式到 MS-SWIFT
convert_unified_to_msswift(
    "unified_data.jsonl", 
    "msswift_sft.jsonl", 
    task_type="sft",
    system_prompt="You are a helpful assistant."
)

# 转换 QA 格式数据
convert_qa_to_msswift(
    "qa_data.jsonl",
    "msswift_qa.jsonl",
    question_key="question",
    answer_key="answer",
    audio_key="audio_path"
)

# 创建单个样本
sample = create_msswift_sample(
    user_content="<audio>What did the speaker say?",
    assistant_content="Hello world.",
    system_prompt="You are helpful.",
    audios=["/path/to/audio.wav"]
)

# 验证格式
stats = validate_msswift_format("output.jsonl")
print(f"Valid: {stats['valid']}, SFT: {stats['sft_samples']}, GRPO: {stats['grpo_samples']}")
```

### 完整工作流示例

```bash
# 1. 下载数据集
python download_datasets.py --dataset librispeech --output ./data --samples 1000

# 2. 转换为 MS-SWIFT SFT 格式
python convert_utils.py msswift ./data/calibration/audio.jsonl ./sft_data.jsonl --task sft

# 3. 验证格式
python convert_utils.py validate-msswift ./sft_data.jsonl

# 4. 使用 MS-SWIFT 训练（参见 ms-swift/bash_scripts/）
cd ../ms-swift
bash bash_scripts/qwen3_omni_sft_lora.sh
```

---

## 🔧 与 REAP-OMNI 集成

下载完成后，可以直接用于 REAP 专家剪枝：

```bash
python ../reap_expert_pruning.py \
    --model-path /path/to/model \
    --output-path /path/to/output \
    --audio-data ./calibration_data/calibration/audio.jsonl \
    --video-data ./calibration_data/calibration/video.jsonl \
    --mixed-data ./calibration_data/calibration/mixed.jsonl \
    --retention-rate 0.5 \
    --calibration-samples 100
```

## ⚠️ 注意事项

1. **数据集协议**: 部分数据集需要同意使用协议才能下载
   - Common Voice: 需要 HuggingFace 登录
   - GigaSpeech: 需要同意协议
   - VoxCeleb: 需要学术协议
   - LRS2/LRS3: 需要 BBC R&D 协议

2. **存储空间**: 完整数据集可能需要数TB空间，建议只下载需要的样本数

3. **网络要求**: 部分数据集从 HuggingFace 下载，建议使用稳定网络

4. **GPU 显存**: 使用校准数据进行模型推理时需要足够的 GPU 显存

5. **路径格式**: MS-SWIFT 格式中的媒体路径建议使用**绝对路径**

## 📚 参考资料

- [DATASETS_CATALOG.md](./DATASETS_CATALOG.md) - 详细的数据集信息
- [MS-SWIFT 官方文档](https://swift.readthedocs.io/) - MS-SWIFT 使用指南
- [MS-SWIFT 自定义数据集](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html) - 数据格式详细说明
