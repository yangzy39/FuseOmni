# REAP-OMNI 多模态数据集目录

本文档整理了适用于 REAP-OMNI 多模态模型剪枝校准的数据集，包含三种类型：
- **Audio-only (S1)**: 纯音频数据集，用于计算音频专家亲和度
- **Video-only (S2)**: 纯视频数据集，用于计算视频专家亲和度  
- **Mixed (S3)**: 音视频混合数据集，用于联合校准

---

## 一、Audio-only 数据集 (纯音频)

| 数据集名称 | 数据来源 | 数据描述 | 用于模型训练 | 数据类型 | 数据规模 | 发布时间 | License | HuggingFace链接 |
|-----------|---------|---------|-------------|---------|---------|---------|---------|----------------|
| **LibriSpeech** | LibriVox audiobooks | 英语朗读语音识别数据集，来自有声读物 | ✅ ASR模型训练 | Audio + Text | 960小时, 1000+小时总量 | 2015 | CC-BY-4.0 | [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) |
| **Common Voice 15.0** | Mozilla众包 | 多语言语音识别数据集，覆盖114种语言 | ✅ 多语言ASR | Audio + Text | 19,159小时验证语音 | 2023 | CC0-1.0 | [mozilla-foundation/common_voice_15_0](https://huggingface.co/datasets/mozilla-foundation/common_voice_15_0) |
| **GigaSpeech** | Audiobooks, Podcasts, YouTube | 大规模英语ASR数据集，覆盖多领域 | ✅ ASR训练 | Audio + Text | 10,000小时标注 + 33,000总量 | 2021 | Apache-2.0 | [speechcolab/gigaspeech](https://huggingface.co/datasets/speechcolab/gigaspeech) |
| **VoxPopuli** | European Parliament | 多语言语音语料库，来自欧洲议会录音 | ✅ ASR/表征学习 | Audio + Text | 400K+小时, 18种语言 | 2021 | CC0-1.0 | [facebook/voxpopuli](https://huggingface.co/datasets/facebook/voxpopuli) |
| **WenetSpeech** | YouTube, Podcasts | 大规模中文ASR数据集，10+领域 | ✅ 中文ASR | Audio + Text | 10,000+小时高质量标注 | 2021 | CC-BY-4.0 | [wenet-e2e/wenetspeech](https://huggingface.co/datasets/wenet-e2e/wenetspeech) |
| **WenetSpeech4TTS** | WenetSpeech衍生 | 中文TTS数据集，优化音频质量和边界 | ✅ TTS训练 | Audio + Text | 12,800小时 | 2024 | CC-BY-4.0 | [Wenetspeech4TTS/WenetSpeech4TTS](https://huggingface.co/datasets/Wenetspeech4TTS/WenetSpeech4TTS) |
| **AISHELL-1** | 众包录制 | 中文普通话ASR数据集 | ✅ 中文ASR | Audio + Text | 170小时 | 2017 | Apache-2.0 | [AISHELL/AISHELL-1](https://huggingface.co/datasets/AISHELL/AISHELL-1) |
| **CoVoST2** | Common Voice衍生 | 多语言语音翻译数据集，21→1 + 1→15 | ✅ Speech Translation | Audio + Text | 2,900小时 | 2020 | CC0-1.0 | [facebook/covost2](https://huggingface.co/datasets/facebook/covost2) |
| **MuST-C** | TED Talks | 多语言语音翻译语料库 | ✅ Speech Translation | Audio + Text | 385+小时/语言方向 | 2019 | CC-BY-4.0 | [may-ohta/MUST-C](https://huggingface.co/datasets/may-ohta/MUST-C) |
| **AudioCaps** | AudioSet衍生 | 音频描述数据集，每段10秒 | ✅ Audio Captioning | Audio + Text | 46K clips | 2019 | MIT | [audiocaps](https://github.com/cdjkim/audiocaps) |
| **WavCaps** | FreeSound, BBC, AudioSet | ChatGPT辅助的音频描述数据集 | ✅ Audio-Language | Audio + Text | 403,050 clips | 2023 | Academic | [cvssp/WavCaps](https://huggingface.co/datasets/cvssp/WavCaps) |
| **AudioSetCaps** | AudioSet, YouTube-8M, VGGSound | 大规模音频描述数据集 | ✅ Audio Captioning | Audio + Text | 6.1M pairs | 2024 | MIT | [AudioSetCaps](https://github.com/JishengBai/AudioSetCaps) |
| **Spoken-SQuAD** | SQuAD TTS生成 | 语音问答数据集 | ✅ Spoken QA | Audio + Text | 37K QA pairs | 2018 | MIT | [Spoken-SQuAD](https://github.com/Chia-Hsuan-Lee/Spoken-SQuAD) |
| **HeySQuAD** | SQuAD衍生 | 大规模语音问答数据集 | ✅ Spoken QA | Audio + Text | 76K human + 97K machine questions | 2024 | Apache-2.0 | [HeySQuAD](https://github.com/yijingjoanna/HeySQuAD) |
| **LibriTTS** | LibriVox audiobooks | 英语TTS数据集，高质量音频 | ✅ TTS训练 | Audio + Text | 585小时 | 2019 | CC-BY-4.0 | [openslr/libritts](https://huggingface.co/datasets/openslr/libritts) |
| **VCTK** | Edinburgh录制 | 多说话人英语语音数据集 | ✅ TTS/VC | Audio + Text | 44小时, 110说话人 | 2019 | CC-BY-4.0 | [VCTK](https://datashare.ed.ac.uk/handle/10283/3443) |

---

## 二、Video-only 数据集 (纯视频/视觉)

| 数据集名称 | 数据来源 | 数据描述 | 用于模型训练 | 数据类型 | 数据规模 | 发布时间 | License | HuggingFace链接 |
|-----------|---------|---------|-------------|---------|---------|---------|---------|----------------|
| **Kinetics-400** | YouTube | 人类动作识别数据集，400类 | ✅ Action Recognition | Video + Label | 306K clips, ~10s each | 2017 | Apache-2.0 | [kinetics-dataset](https://github.com/cvdfoundation/kinetics-dataset) |
| **Kinetics-700** | YouTube | 扩展版动作识别，700类 | ✅ Action Recognition | Video + Label | 650K clips | 2019 | Apache-2.0 | [kinetics-dataset](https://github.com/cvdfoundation/kinetics-dataset) |
| **UCF101** | YouTube | 动作识别经典数据集，101类 | ✅ Action Recognition | Video + Label | 13,320 clips, 27小时 | 2012 | N/A | [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) |
| **HMDB51** | Movies/Web | 人类动作数据集，51类 | ✅ Action Recognition | Video + Label | 6,849 clips | 2011 | CC-BY-4.0 | [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) |
| **MSR-VTT** | Commercial Search | 视频描述基准数据集 | ✅ Video Captioning | Video + Text | 10K clips, 200K句子 | 2016 | N/A | [MSR-VTT](https://cove.thecvf.com/datasets/839) |
| **VATEX** | Kinetics-600衍生 | 多语言视频描述数据集(EN/ZH) | ✅ Video Captioning | Video + Text | 41.3K clips, 826K captions | 2019 | CC-BY-4.0 | [VATEX](https://eric-xw.github.io/vatex-website/) |
| **YouCook2** | YouTube Cooking | 烹饪教学视频数据集 | ✅ Video Understanding | Video + Text | 2,000 videos, 89 recipes | 2018 | MIT | [merve/YouCook2](https://huggingface.co/datasets/merve/YouCook2) |
| **ActivityNet-QA** | ActivityNet | 视频问答数据集 | ✅ Video QA | Video + QA | 58K QA pairs, 5.8K videos | 2019 | N/A | [ActivityNet-QA](https://github.com/MILVLG/activitynet-qa) |
| **LongVideoBench** | Web Videos | 长视频理解基准 | ✅ Long Video QA | Video + Text + QA | 3,763 videos, 6,678 QA | 2024 | Apache-2.0 | [longvideobench/LongVideoBench](https://huggingface.co/datasets/longvideobench/LongVideoBench) |
| **FineVideo** | Web Videos | 精细视频描述数据集 | ✅ Video Captioning | Video + Text + QA | 43K videos, 3.4K小时 | 2024 | Apache-2.0 | [FineVideo](https://huggingface.co/blog/fine-video) |
| **TemporalBench** | Human Annotated | 时序理解基准数据集 | ✅ Temporal Understanding | Video + QA | 10K QA pairs | 2024 | MIT | [TemporalBench](https://github.com/mu-cai/TemporalBench) |
| **CinePile** | Movies | 长视频问答数据集 | ✅ Long Video QA | Video + QA | 305K MCQs | 2024 | N/A | [CinePile](https://huggingface.co/papers/2405.08813) |
| **MMBench-Video** | YouTube | 视频理解评估基准 | ✅ Video Understanding | Video + QA | 900 videos | 2024 | N/A | [MMBench-Video](https://mmbench-video.github.io/) |
| **VITATECS** | VATEX/MSRVTT | 时序概念理解诊断数据集 | ✅ Temporal Concepts | Video + Text | Counterfactual descriptions | 2024 | MIT | [VITATECS](https://github.com/lscpku/vitatecs) |
| **VIOLIN** | TV Shows/Movies | 视频语言推理数据集 | ✅ Video-Language Inference | Video + Text + Hypothesis | 95K pairs, 582小时 | 2020 | N/A | [VIOLIN](https://github.com/jimmy646/violin) |
| **VideoChat2-IT** | Multiple sources | 视频对话指令调优数据集 | ✅ Video Chat | Video + Dialogue | 1.9M annotations | 2024 | N/A | [OpenGVLab/VideoChat2-IT](https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT) |

---

## 三、Mixed 数据集 (音视频混合)

| 数据集名称 | 数据来源 | 数据描述 | 用于模型训练 | 数据类型 | 数据规模 | 发布时间 | License | HuggingFace链接 |
|-----------|---------|---------|-------------|---------|---------|---------|---------|----------------|
| **VoxCeleb1/2** | YouTube Interviews | 音视频说话人识别数据集 | ✅ Speaker Recognition | Audio + Video | 1M+ utterances, 7000+ speakers | 2017/2018 | CC-BY-4.0 | [VoxCeleb](https://robots.ox.ac.uk/~vgg/data/voxceleb) |
| **AudioSet** | YouTube | 大规模音频事件数据集 | ✅ Audio Classification | Audio + Video | 2M+ clips, 632类 | 2017 | CC-BY-4.0 | [AudioSet](https://research.google.com/audioset/) |
| **VGGSound** | YouTube | 音视频对应数据集 | ✅ Audio-Visual Learning | Audio + Video | 210K videos, 310类 | 2020 | CC-BY-4.0 | [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/) |
| **LRS2** | BBC Television | 音视频语音识别数据集 | ✅ AV-ASR/Lip Reading | Audio + Video + Text | 数千句子 | 2018 | Non-commercial | [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html) |
| **LRS3** | TED/TEDx | 音视频语音识别数据集 | ✅ AV-ASR/Lip Reading | Audio + Video + Text | 数千句子 | 2018 | CC-BY-4.0 | [LRS3](https://mmai.io/datasets/lip_reading/) |
| **AVSpeech** | YouTube Instructional | 单人说话视频数据集 | ✅ AV Speech | Audio + Video | 大规模clips | 2018 | N/A | [AVSpeech](https://looking-to-listen.github.io/avspeech/) |
| **How2** | YouTube Instructional | 多模态教学视频数据集 | ✅ Multimodal Learning | Audio + Video + Text | 80K clips, 2000小时 | 2018 | MIT | [How2](https://srvk.github.io/how2-dataset/) |
| **VALOR-1M** | Web Videos | 三模态预训练数据集 | ✅ Vision-Audio-Language | Audio + Video + Text | 1M videos with captions | 2023 | N/A | [VALOR](https://arxiv.org/abs/2304.08345) |
| **Video-MME** | YouTube | 多模态视频评估基准(含音频) | ✅ Multimodal Evaluation | Audio + Video + Text | 900 videos | 2024 | N/A | [Video-MME](https://arxiv.org/abs/2405.21075) |
| **JointAVBench** | Multiple sources | 音视频联合推理基准 | ✅ Joint AV Reasoning | Audio + Video + QA | 多类型问题 | 2024 | N/A | [JointAVBench](https://arxiv.org/abs/2512.12772) |
| **UGC-VideoCap** | TikTok | 短视频多模态描述数据集 | ✅ Omni-modal Captioning | Audio + Video + Text | 1,000 videos, 4,000 QA | 2025 | N/A | [openinterx/UGC-VideoCap](https://huggingface.co/datasets/openinterx/UGC-VideoCap) |
| **MELD** | TV Shows (Friends) | 多模态情感对话数据集 | ✅ Emotion Recognition | Audio + Video + Text | Episodes with emotions | 2019 | GPL-3.0 | [MELD](https://github.com/declare-lab/MELD) |
| **HowTo100M** | YouTube | 大规模教学视频数据集 | ✅ Text-Video Embedding | Audio + Video + Text | 136M clips, 1.22M videos | 2019 | N/A | [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/) |

---

## 四、数据集选择建议

### 用于 REAP-OMNI 校准的推荐组合

| 模态类型 | 推荐数据集 | 理由 |
|---------|-----------|------|
| **Audio-only (S1)** | LibriSpeech, Common Voice, GigaSpeech | 高质量ASR数据，覆盖朗读和自然对话 |
| **Video-only (S2)** | Kinetics-400, MSR-VTT, VATEX | 多样化视频内容，良好的视觉特征 |
| **Mixed (S3)** | VoxCeleb, LRS2/LRS3, How2 | 真实音视频对应，高质量标注 |

### 数据规模建议

根据 REAP-OMNI README 中的配置，建议每种模态使用 100-1000 个校准样本：

```bash
--calibration-samples 100  # 默认值
```

---

## 五、下载与使用

请参考同目录下的 `download_datasets.py` 脚本进行数据集下载和格式转换。

### 统一输出格式

```json
{
    "id": "dataset_name_00001",
    "text": "源文本内容（如有）",
    "audio": "path/to/audio.wav",
    "video": "path/to/video.mp4"
}
```
