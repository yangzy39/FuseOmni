#!/usr/bin/env python3
"""
REAP-OMNI 多模态数据集下载与转换脚本

本脚本实现以下功能：
1. 自动下载支持的数据集
2. 将所有数据转换为统一格式
3. 生成 JSONL 格式的校准数据

统一输出格式:
{
    "id": "dataset_name_00001",
    "text": "源文本（如有）",
    "audio": "path/to/audio.wav",  # 如有
    "video": "path/to/video.mp4"   # 如有
}

Usage:
    python download_datasets.py --dataset librispeech --output ./data --samples 100
    python download_datasets.py --dataset all --output ./data --samples 100
    python download_datasets.py --list  # 列出所有支持的数据集
"""

import os
import json
import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Any, Generator
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class UnifiedSample:
    """统一数据格式"""
    id: str
    text: Optional[str] = None
    audio: Optional[str] = None
    video: Optional[str] = None
    modality: str = "mixed"  # audio, video, mixed
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，排除None值"""
        result = {"id": self.id, "modality": self.modality}
        if self.text is not None:
            result["text"] = self.text
        if self.audio is not None:
            result["audio"] = self.audio
        if self.video is not None:
            result["video"] = self.video
        return result


class DatasetDownloader(ABC):
    """数据集下载器基类"""
    
    name: str = "base"
    modality: str = "mixed"  # audio, video, mixed
    description: str = ""
    url: str = ""
    
    def __init__(self, output_dir: Path, max_samples: int = 100):
        self.output_dir = output_dir
        self.max_samples = max_samples
        self.data_dir = output_dir / self.name
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def download(self) -> bool:
        """下载数据集，返回是否成功"""
        pass
    
    @abstractmethod
    def convert(self) -> Generator[UnifiedSample, None, None]:
        """转换为统一格式，yield UnifiedSample"""
        pass
    
    def process(self) -> List[UnifiedSample]:
        """完整处理流程"""
        logger.info(f"Processing dataset: {self.name}")
        
        if not self.download():
            logger.error(f"Failed to download {self.name}")
            return []
        
        samples = []
        for i, sample in enumerate(self.convert()):
            if i >= self.max_samples:
                break
            samples.append(sample)
            
        logger.info(f"Processed {len(samples)} samples from {self.name}")
        return samples


# ============== Audio-only Datasets ==============

class LibriSpeechDownloader(DatasetDownloader):
    """LibriSpeech ASR 数据集"""
    
    name = "librispeech"
    modality = "audio"
    description = "英语朗读语音识别数据集，来自LibriVox有声读物"
    url = "https://huggingface.co/datasets/openslr/librispeech_asr"
    
    def download(self) -> bool:
        try:
            from datasets import load_dataset
            logger.info("Loading LibriSpeech from HuggingFace...")
            self.dataset = load_dataset(
                "openslr/librispeech_asr", 
                "clean",
                split="train.100",
                trust_remote_code=True
            )
            return True
        except Exception as e:
            logger.error(f"Error loading LibriSpeech: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        import soundfile as sf
        
        audio_dir = self.data_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        for idx, item in enumerate(self.dataset):
            if idx >= self.max_samples:
                break
                
            # 保存音频文件
            audio_path = audio_dir / f"{self.name}_{idx:05d}.wav"
            audio_array = item["audio"]["array"]
            sample_rate = item["audio"]["sampling_rate"]
            sf.write(str(audio_path), audio_array, sample_rate)
            
            yield UnifiedSample(
                id=f"{self.name}_{idx:05d}",
                text=item["text"],
                audio=str(audio_path),
                modality="audio"
            )


class CommonVoiceDownloader(DatasetDownloader):
    """Mozilla Common Voice 数据集"""
    
    name = "common_voice"
    modality = "audio"
    description = "多语言众包语音识别数据集"
    url = "https://huggingface.co/datasets/mozilla-foundation/common_voice_15_0"
    
    def download(self) -> bool:
        try:
            from datasets import load_dataset
            logger.info("Loading Common Voice from HuggingFace...")
            # 加载英语子集
            self.dataset = load_dataset(
                "mozilla-foundation/common_voice_15_0",
                "en",
                split="train",
                trust_remote_code=True
            )
            return True
        except Exception as e:
            logger.error(f"Error loading Common Voice: {e}")
            logger.info("Common Voice requires login. Please run: huggingface-cli login")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        import soundfile as sf
        
        audio_dir = self.data_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        for idx, item in enumerate(self.dataset):
            if idx >= self.max_samples:
                break
            
            try:
                audio_path = audio_dir / f"{self.name}_{idx:05d}.wav"
                audio_array = item["audio"]["array"]
                sample_rate = item["audio"]["sampling_rate"]
                sf.write(str(audio_path), audio_array, sample_rate)
                
                yield UnifiedSample(
                    id=f"{self.name}_{idx:05d}",
                    text=item["sentence"],
                    audio=str(audio_path),
                    modality="audio"
                )
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue


class GigaSpeechDownloader(DatasetDownloader):
    """GigaSpeech 大规模ASR数据集"""
    
    name = "gigaspeech"
    modality = "audio"
    description = "10000小时多领域英语ASR数据集"
    url = "https://huggingface.co/datasets/speechcolab/gigaspeech"
    
    def download(self) -> bool:
        try:
            from datasets import load_dataset
            logger.info("Loading GigaSpeech XS subset from HuggingFace...")
            self.dataset = load_dataset(
                "speechcolab/gigaspeech",
                "xs",  # 使用最小子集
                split="train",
                trust_remote_code=True
            )
            return True
        except Exception as e:
            logger.error(f"Error loading GigaSpeech: {e}")
            logger.info("GigaSpeech requires agreement. Visit the HuggingFace page to accept terms.")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        import soundfile as sf
        
        audio_dir = self.data_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        for idx, item in enumerate(self.dataset):
            if idx >= self.max_samples:
                break
            
            try:
                audio_path = audio_dir / f"{self.name}_{idx:05d}.wav"
                audio_array = item["audio"]["array"]
                sample_rate = item["audio"]["sampling_rate"]
                sf.write(str(audio_path), audio_array, sample_rate)
                
                yield UnifiedSample(
                    id=f"{self.name}_{idx:05d}",
                    text=item["text"],
                    audio=str(audio_path),
                    modality="audio"
                )
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue


class WavCapsDownloader(DatasetDownloader):
    """WavCaps 音频描述数据集"""
    
    name = "wavcaps"
    modality = "audio"
    description = "ChatGPT辅助的音频描述数据集"
    url = "https://huggingface.co/datasets/cvssp/WavCaps"
    
    def download(self) -> bool:
        try:
            from datasets import load_dataset
            logger.info("Loading WavCaps from HuggingFace...")
            self.dataset = load_dataset(
                "cvssp/WavCaps",
                split="train",
                trust_remote_code=True
            )
            return True
        except Exception as e:
            logger.error(f"Error loading WavCaps: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        import soundfile as sf
        
        audio_dir = self.data_dir / "audio"
        audio_dir.mkdir(exist_ok=True)
        
        for idx, item in enumerate(self.dataset):
            if idx >= self.max_samples:
                break
            
            try:
                audio_path = audio_dir / f"{self.name}_{idx:05d}.wav"
                if "audio" in item and item["audio"] is not None:
                    audio_array = item["audio"]["array"]
                    sample_rate = item["audio"]["sampling_rate"]
                    sf.write(str(audio_path), audio_array, sample_rate)
                    
                    yield UnifiedSample(
                        id=f"{self.name}_{idx:05d}",
                        text=item.get("caption", ""),
                        audio=str(audio_path),
                        modality="audio"
                    )
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue


# ============== Video-only Datasets ==============

class Kinetics400Downloader(DatasetDownloader):
    """Kinetics-400 动作识别数据集"""
    
    name = "kinetics400"
    modality = "video"
    description = "人类动作识别数据集，400类"
    url = "https://github.com/cvdfoundation/kinetics-dataset"
    
    def download(self) -> bool:
        try:
            # Kinetics需要特殊处理，这里提供下载脚本路径
            logger.info("Kinetics-400 requires manual download.")
            logger.info("Please download from: https://github.com/cvdfoundation/kinetics-dataset")
            logger.info("Or use: pip install kinetics-dataset")
            
            # 尝试加载本地数据
            video_dir = self.data_dir / "videos"
            if video_dir.exists() and any(video_dir.iterdir()):
                self.video_files = list(video_dir.glob("*.mp4"))[:self.max_samples]
                return True
            return False
        except Exception as e:
            logger.error(f"Error with Kinetics-400: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        for idx, video_path in enumerate(self.video_files):
            if idx >= self.max_samples:
                break
            
            # 从文件名提取标签
            label = video_path.stem.rsplit("_", 1)[0] if "_" in video_path.stem else ""
            
            yield UnifiedSample(
                id=f"{self.name}_{idx:05d}",
                text=label,
                video=str(video_path),
                modality="video"
            )


class MSRVTTDownloader(DatasetDownloader):
    """MSR-VTT 视频描述数据集"""
    
    name = "msrvtt"
    modality = "video"
    description = "视频描述基准数据集"
    url = "https://cove.thecvf.com/datasets/839"
    
    def download(self) -> bool:
        try:
            logger.info("MSR-VTT requires manual download.")
            logger.info("Please download from: https://cove.thecvf.com/datasets/839")
            
            video_dir = self.data_dir / "videos"
            annotations_file = self.data_dir / "annotations.json"
            
            if video_dir.exists() and annotations_file.exists():
                with open(annotations_file) as f:
                    self.annotations = json.load(f)
                self.video_dir = video_dir
                return True
            return False
        except Exception as e:
            logger.error(f"Error with MSR-VTT: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        sentences = self.annotations.get("sentences", [])
        
        for idx, item in enumerate(sentences):
            if idx >= self.max_samples:
                break
            
            video_id = item.get("video_id", "")
            video_path = self.video_dir / f"{video_id}.mp4"
            
            if video_path.exists():
                yield UnifiedSample(
                    id=f"{self.name}_{idx:05d}",
                    text=item.get("caption", ""),
                    video=str(video_path),
                    modality="video"
                )


class LongVideoBenchDownloader(DatasetDownloader):
    """LongVideoBench 长视频理解数据集"""
    
    name = "longvideobench"
    modality = "video"
    description = "长视频理解基准数据集"
    url = "https://huggingface.co/datasets/longvideobench/LongVideoBench"
    
    def download(self) -> bool:
        try:
            from datasets import load_dataset
            logger.info("Loading LongVideoBench from HuggingFace...")
            self.dataset = load_dataset(
                "longvideobench/LongVideoBench",
                split="test",
                trust_remote_code=True
            )
            return True
        except Exception as e:
            logger.error(f"Error loading LongVideoBench: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        video_dir = self.data_dir / "videos"
        video_dir.mkdir(exist_ok=True)
        
        for idx, item in enumerate(self.dataset):
            if idx >= self.max_samples:
                break
            
            try:
                # 获取问题和答案作为文本
                question = item.get("question", "")
                
                yield UnifiedSample(
                    id=f"{self.name}_{idx:05d}",
                    text=question,
                    video=item.get("video_path", None),
                    modality="video"
                )
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue


# ============== Mixed Datasets (Audio + Video) ==============

class VoxCelebDownloader(DatasetDownloader):
    """VoxCeleb 音视频说话人数据集"""
    
    name = "voxceleb"
    modality = "mixed"
    description = "音视频说话人识别数据集"
    url = "https://robots.ox.ac.uk/~vgg/data/voxceleb"
    
    def download(self) -> bool:
        try:
            logger.info("VoxCeleb requires manual download with agreement.")
            logger.info("Please visit: https://robots.ox.ac.uk/~vgg/data/voxceleb")
            
            # 检查本地数据
            data_path = self.data_dir / "voxceleb1"
            if data_path.exists():
                self.data_path = data_path
                return True
            return False
        except Exception as e:
            logger.error(f"Error with VoxCeleb: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        # VoxCeleb 数据格式: id/video_id/clip_id.wav
        for idx, audio_file in enumerate(self.data_path.rglob("*.wav")):
            if idx >= self.max_samples:
                break
            
            # 查找对应的视频文件
            video_file = audio_file.with_suffix(".mp4")
            
            yield UnifiedSample(
                id=f"{self.name}_{idx:05d}",
                text=None,  # VoxCeleb 没有文本
                audio=str(audio_file),
                video=str(video_file) if video_file.exists() else None,
                modality="mixed"
            )


class How2Downloader(DatasetDownloader):
    """How2 多模态教学视频数据集"""
    
    name = "how2"
    modality = "mixed"
    description = "多模态教学视频数据集"
    url = "https://srvk.github.io/how2-dataset/"
    
    def download(self) -> bool:
        try:
            logger.info("How2 dataset requires download from official site.")
            logger.info("Please visit: https://srvk.github.io/how2-dataset/")
            
            data_path = self.data_dir / "how2"
            if data_path.exists():
                self.data_path = data_path
                return True
            return False
        except Exception as e:
            logger.error(f"Error with How2: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        # How2 通常有 video_id.mp4, video_id.wav, video_id.txt
        txt_files = list(self.data_path.glob("*.txt"))
        
        for idx, txt_file in enumerate(txt_files):
            if idx >= self.max_samples:
                break
            
            video_id = txt_file.stem
            audio_file = txt_file.with_suffix(".wav")
            video_file = txt_file.with_suffix(".mp4")
            
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            yield UnifiedSample(
                id=f"{self.name}_{idx:05d}",
                text=text,
                audio=str(audio_file) if audio_file.exists() else None,
                video=str(video_file) if video_file.exists() else None,
                modality="mixed"
            )


class LRS2Downloader(DatasetDownloader):
    """LRS2 音视频语音识别数据集"""
    
    name = "lrs2"
    modality = "mixed"
    description = "BBC音视频语音识别数据集"
    url = "https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html"
    
    def download(self) -> bool:
        try:
            logger.info("LRS2 requires agreement with BBC R&D.")
            logger.info("Please visit: https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html")
            
            data_path = self.data_dir / "lrs2"
            if data_path.exists():
                self.data_path = data_path
                return True
            return False
        except Exception as e:
            logger.error(f"Error with LRS2: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        # LRS2 格式: {split}/{video_id}/{clip_id}.mp4 + .txt
        mp4_files = list(self.data_path.rglob("*.mp4"))
        
        for idx, video_file in enumerate(mp4_files):
            if idx >= self.max_samples:
                break
            
            txt_file = video_file.with_suffix(".txt")
            text = ""
            if txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            
            # 音频嵌入在视频中
            yield UnifiedSample(
                id=f"{self.name}_{idx:05d}",
                text=text,
                audio=None,  # 音频在视频内
                video=str(video_file),
                modality="mixed"
            )


class AudioSetDownloader(DatasetDownloader):
    """AudioSet 大规模音频数据集"""
    
    name = "audioset"
    modality = "mixed"
    description = "大规模音频事件分类数据集"
    url = "https://research.google.com/audioset/"
    
    def download(self) -> bool:
        try:
            logger.info("AudioSet requires downloading from YouTube using official tools.")
            logger.info("Please visit: https://research.google.com/audioset/download.html")
            
            # 检查本地数据
            data_path = self.data_dir / "audioset"
            if data_path.exists():
                self.data_path = data_path
                # 尝试加载标签文件
                labels_file = data_path / "balanced_train_segments.csv"
                if labels_file.exists():
                    self.labels = self._load_labels(labels_file)
                    return True
            return False
        except Exception as e:
            logger.error(f"Error with AudioSet: {e}")
            return False
    
    def _load_labels(self, labels_file: Path) -> Dict:
        """加载AudioSet标签文件"""
        labels = {}
        with open(labels_file, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    ytid = parts[0].strip('"')
                    labels[ytid] = parts[3:]
        return labels
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        audio_files = list(self.data_path.glob("*.wav")) + list(self.data_path.glob("*.flac"))
        
        for idx, audio_file in enumerate(audio_files):
            if idx >= self.max_samples:
                break
            
            ytid = audio_file.stem
            label_ids = self.labels.get(ytid, [])
            
            yield UnifiedSample(
                id=f"{self.name}_{idx:05d}",
                text=",".join(label_ids),
                audio=str(audio_file),
                video=None,
                modality="mixed"
            )


class VGGSoundDownloader(DatasetDownloader):
    """VGGSound 音视频对应数据集"""
    
    name = "vggsound"
    modality = "mixed"
    description = "音视频对应数据集"
    url = "https://www.robots.ox.ac.uk/~vgg/data/vggsound/"
    
    def download(self) -> bool:
        try:
            logger.info("VGGSound requires downloading from official site.")
            logger.info("Please visit: https://www.robots.ox.ac.uk/~vgg/data/vggsound/")
            
            data_path = self.data_dir / "vggsound"
            if data_path.exists():
                self.data_path = data_path
                return True
            return False
        except Exception as e:
            logger.error(f"Error with VGGSound: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        video_files = list(self.data_path.glob("*.mp4"))
        
        for idx, video_file in enumerate(video_files):
            if idx >= self.max_samples:
                break
            
            # 从文件名提取标签 (格式: ytid_start_end_label.mp4)
            parts = video_file.stem.rsplit("_", 1)
            label = parts[-1] if len(parts) > 1 else ""
            
            yield UnifiedSample(
                id=f"{self.name}_{idx:05d}",
                text=label.replace("_", " "),
                audio=None,  # 音频在视频内
                video=str(video_file),
                modality="mixed"
            )


# ============== Dataset Registry ==============

DATASET_REGISTRY: Dict[str, type] = {
    # Audio-only
    "librispeech": LibriSpeechDownloader,
    "common_voice": CommonVoiceDownloader,
    "gigaspeech": GigaSpeechDownloader,
    "wavcaps": WavCapsDownloader,
    
    # Video-only
    "kinetics400": Kinetics400Downloader,
    "msrvtt": MSRVTTDownloader,
    "longvideobench": LongVideoBenchDownloader,
    
    # Mixed
    "voxceleb": VoxCelebDownloader,
    "how2": How2Downloader,
    "lrs2": LRS2Downloader,
    "audioset": AudioSetDownloader,
    "vggsound": VGGSoundDownloader,
}


def list_datasets():
    """列出所有支持的数据集"""
    print("\n" + "=" * 80)
    print("REAP-OMNI 支持的数据集")
    print("=" * 80)
    
    audio_datasets = []
    video_datasets = []
    mixed_datasets = []
    
    for name, cls in DATASET_REGISTRY.items():
        info = {"name": name, "description": cls.description, "url": cls.url}
        if cls.modality == "audio":
            audio_datasets.append(info)
        elif cls.modality == "video":
            video_datasets.append(info)
        else:
            mixed_datasets.append(info)
    
    print("\n📢 Audio-only 数据集 (S1):")
    print("-" * 40)
    for ds in audio_datasets:
        print(f"  • {ds['name']}: {ds['description']}")
        print(f"    URL: {ds['url']}")
    
    print("\n🎬 Video-only 数据集 (S2):")
    print("-" * 40)
    for ds in video_datasets:
        print(f"  • {ds['name']}: {ds['description']}")
        print(f"    URL: {ds['url']}")
    
    print("\n🔀 Mixed 数据集 (S3):")
    print("-" * 40)
    for ds in mixed_datasets:
        print(f"  • {ds['name']}: {ds['description']}")
        print(f"    URL: {ds['url']}")
    
    print("\n" + "=" * 80)


def save_jsonl(samples: List[UnifiedSample], output_path: Path, modality: str):
    """保存为JSONL格式"""
    output_file = output_path / f"{modality}.jsonl"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(samples)} samples to {output_file}")
    return output_file


def process_datasets(
    datasets: List[str],
    output_dir: Path,
    max_samples: int = 100
) -> Dict[str, List[UnifiedSample]]:
    """处理多个数据集"""
    
    results = {
        "audio": [],
        "video": [],
        "mixed": []
    }
    
    for ds_name in datasets:
        if ds_name not in DATASET_REGISTRY:
            logger.warning(f"Unknown dataset: {ds_name}")
            continue
        
        downloader_cls = DATASET_REGISTRY[ds_name]
        downloader = downloader_cls(output_dir, max_samples)
        
        try:
            samples = downloader.process()
            results[downloader.modality].extend(samples)
        except Exception as e:
            logger.error(f"Error processing {ds_name}: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="REAP-OMNI 多模态数据集下载与转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 列出所有支持的数据集
  python download_datasets.py --list
  
  # 下载单个数据集
  python download_datasets.py --dataset librispeech --output ./data --samples 100
  
  # 下载多个数据集
  python download_datasets.py --dataset librispeech gigaspeech --output ./data
  
  # 下载所有数据集
  python download_datasets.py --dataset all --output ./data
  
  # 按模态下载
  python download_datasets.py --modality audio --output ./data
        """
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="列出所有支持的数据集"
    )
    
    parser.add_argument(
        "--dataset", "-d",
        nargs="+",
        default=[],
        help="要下载的数据集名称，使用 'all' 下载全部"
    )
    
    parser.add_argument(
        "--modality", "-m",
        choices=["audio", "video", "mixed", "all"],
        default=None,
        help="按模态类型下载"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./data"),
        help="输出目录"
    )
    
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=100,
        help="每个数据集的最大样本数"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.list:
        list_datasets()
        return
    
    # 确定要处理的数据集
    datasets_to_process = []
    
    if args.modality:
        if args.modality == "all":
            datasets_to_process = list(DATASET_REGISTRY.keys())
        else:
            for name, cls in DATASET_REGISTRY.items():
                if cls.modality == args.modality:
                    datasets_to_process.append(name)
    elif args.dataset:
        if "all" in args.dataset:
            datasets_to_process = list(DATASET_REGISTRY.keys())
        else:
            datasets_to_process = args.dataset
    else:
        parser.print_help()
        return
    
    if not datasets_to_process:
        logger.error("No datasets to process!")
        return
    
    logger.info(f"Processing datasets: {datasets_to_process}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Max samples per dataset: {args.samples}")
    
    # 创建输出目录
    args.output.mkdir(parents=True, exist_ok=True)
    calibration_dir = args.output / "calibration"
    calibration_dir.mkdir(exist_ok=True)
    
    # 处理数据集
    results = process_datasets(datasets_to_process, args.output, args.samples)
    
    # 保存为JSONL
    for modality, samples in results.items():
        if samples:
            save_jsonl(samples, calibration_dir, modality)
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("处理完成!")
    print("=" * 80)
    print(f"  Audio samples: {len(results['audio'])}")
    print(f"  Video samples: {len(results['video'])}")
    print(f"  Mixed samples: {len(results['mixed'])}")
    print(f"\n校准数据已保存到: {calibration_dir}")
    print("\n可用于 REAP-OMNI 的命令:")
    print(f"  python reap_expert_pruning.py \\")
    print(f"      --audio-data {calibration_dir / 'audio.jsonl'} \\")
    print(f"      --video-data {calibration_dir / 'video.jsonl'} \\")
    print(f"      --mixed-data {calibration_dir / 'mixed.jsonl'}")


if __name__ == "__main__":
    main()
