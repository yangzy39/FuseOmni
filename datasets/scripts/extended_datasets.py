"""
REAP-OMNI 扩展数据集下载器

本模块包含更多数据集的下载器实现，作为 download_datasets.py 的扩展。
"""

import os
import json
import logging
from pathlib import Path
from typing import Generator, Optional
from dataclasses import dataclass

# 从主模块导入
try:
    from download_datasets import (
        DatasetDownloader, 
        UnifiedSample, 
        DATASET_REGISTRY
    )
except ImportError:
    from .download_datasets import (
        DatasetDownloader, 
        UnifiedSample, 
        DATASET_REGISTRY
    )

logger = logging.getLogger(__name__)


# ============== 额外的 Audio-only 数据集 ==============

class VoxPopuliDownloader(DatasetDownloader):
    """VoxPopuli 多语言语音数据集"""
    
    name = "voxpopuli"
    modality = "audio"
    description = "欧洲议会多语言语音语料库"
    url = "https://huggingface.co/datasets/facebook/voxpopuli"
    
    def download(self) -> bool:
        try:
            from datasets import load_dataset
            logger.info("Loading VoxPopuli from HuggingFace...")
            self.dataset = load_dataset(
                "facebook/voxpopuli",
                "en",  # 英语子集
                split="train",
                trust_remote_code=True
            )
            return True
        except Exception as e:
            logger.error(f"Error loading VoxPopuli: {e}")
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
                    text=item.get("normalized_text", item.get("raw_text", "")),
                    audio=str(audio_path),
                    modality="audio"
                )
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue


class AishellDownloader(DatasetDownloader):
    """AISHELL-1 中文语音数据集"""
    
    name = "aishell"
    modality = "audio"
    description = "中文普通话语音识别数据集"
    url = "https://huggingface.co/datasets/AISHELL/AISHELL-1"
    
    def download(self) -> bool:
        try:
            from datasets import load_dataset
            logger.info("Loading AISHELL-1 from HuggingFace...")
            self.dataset = load_dataset(
                "AISHELL/AISHELL-1",
                split="train",
                trust_remote_code=True
            )
            return True
        except Exception as e:
            logger.error(f"Error loading AISHELL-1: {e}")
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
                    text=item.get("text", ""),
                    audio=str(audio_path),
                    modality="audio"
                )
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue


class CoVoST2Downloader(DatasetDownloader):
    """CoVoST2 语音翻译数据集"""
    
    name = "covost2"
    modality = "audio"
    description = "多语言语音翻译数据集"
    url = "https://huggingface.co/datasets/facebook/covost2"
    
    def download(self) -> bool:
        try:
            from datasets import load_dataset
            logger.info("Loading CoVoST2 from HuggingFace...")
            # 加载法语到英语的翻译
            self.dataset = load_dataset(
                "facebook/covost2",
                "fr_en",
                split="train",
                trust_remote_code=True
            )
            return True
        except Exception as e:
            logger.error(f"Error loading CoVoST2: {e}")
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
                
                # 使用翻译后的文本
                yield UnifiedSample(
                    id=f"{self.name}_{idx:05d}",
                    text=item.get("translation", item.get("sentence", "")),
                    audio=str(audio_path),
                    modality="audio"
                )
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue


# ============== 额外的 Video-only 数据集 ==============

class YouCook2Downloader(DatasetDownloader):
    """YouCook2 烹饪视频数据集"""
    
    name = "youcook2"
    modality = "video"
    description = "烹饪教学视频数据集"
    url = "https://huggingface.co/datasets/merve/YouCook2"
    
    def download(self) -> bool:
        try:
            from datasets import load_dataset
            logger.info("Loading YouCook2 from HuggingFace...")
            self.dataset = load_dataset(
                "merve/YouCook2",
                split="train",
                trust_remote_code=True
            )
            return True
        except Exception as e:
            logger.error(f"Error loading YouCook2: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        for idx, item in enumerate(self.dataset):
            if idx >= self.max_samples:
                break
            
            try:
                yield UnifiedSample(
                    id=f"{self.name}_{idx:05d}",
                    text=item.get("caption", item.get("text", "")),
                    video=item.get("video_path", None),
                    modality="video"
                )
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue


class VATEXDownloader(DatasetDownloader):
    """VATEX 多语言视频描述数据集"""
    
    name = "vatex"
    modality = "video"
    description = "多语言视频描述数据集 (EN/ZH)"
    url = "https://eric-xw.github.io/vatex-website/"
    
    def download(self) -> bool:
        try:
            logger.info("VATEX requires manual download.")
            logger.info("Please visit: https://eric-xw.github.io/vatex-website/")
            
            # 检查本地标注文件
            annotations_file = self.data_dir / "vatex_training_v1.0.json"
            video_dir = self.data_dir / "videos"
            
            if annotations_file.exists():
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    self.annotations = json.load(f)
                self.video_dir = video_dir
                return True
            return False
        except Exception as e:
            logger.error(f"Error with VATEX: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        for idx, item in enumerate(self.annotations):
            if idx >= self.max_samples:
                break
            
            video_id = item.get("videoID", "")
            video_path = self.video_dir / f"{video_id}.mp4"
            
            # 获取英文描述
            en_cap = item.get("enCap", [""])[0] if item.get("enCap") else ""
            
            yield UnifiedSample(
                id=f"{self.name}_{idx:05d}",
                text=en_cap,
                video=str(video_path) if video_path.exists() else None,
                modality="video"
            )


class ActivityNetQADownloader(DatasetDownloader):
    """ActivityNet-QA 视频问答数据集"""
    
    name = "activitynet_qa"
    modality = "video"
    description = "视频问答数据集"
    url = "https://github.com/MILVLG/activitynet-qa"
    
    def download(self) -> bool:
        try:
            logger.info("ActivityNet-QA requires manual download.")
            logger.info("Please visit: https://github.com/MILVLG/activitynet-qa")
            
            qa_file = self.data_dir / "train_qa.json"
            video_dir = self.data_dir / "videos"
            
            if qa_file.exists():
                with open(qa_file, 'r', encoding='utf-8') as f:
                    self.qa_data = json.load(f)
                self.video_dir = video_dir
                return True
            return False
        except Exception as e:
            logger.error(f"Error with ActivityNet-QA: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        for idx, item in enumerate(self.qa_data):
            if idx >= self.max_samples:
                break
            
            video_id = item.get("video_name", "")
            video_path = self.video_dir / f"{video_id}.mp4"
            
            question = item.get("question", "")
            answer = item.get("answer", "")
            text = f"Q: {question} A: {answer}"
            
            yield UnifiedSample(
                id=f"{self.name}_{idx:05d}",
                text=text,
                video=str(video_path) if video_path.exists() else None,
                modality="video"
            )


class VideoChat2ITDownloader(DatasetDownloader):
    """VideoChat2-IT 视频对话数据集"""
    
    name = "videochat2_it"
    modality = "video"
    description = "视频对话指令调优数据集"
    url = "https://huggingface.co/datasets/OpenGVLab/VideoChat2-IT"
    
    def download(self) -> bool:
        try:
            from datasets import load_dataset
            logger.info("Loading VideoChat2-IT from HuggingFace...")
            self.dataset = load_dataset(
                "OpenGVLab/VideoChat2-IT",
                split="train",
                trust_remote_code=True
            )
            return True
        except Exception as e:
            logger.error(f"Error loading VideoChat2-IT: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        for idx, item in enumerate(self.dataset):
            if idx >= self.max_samples:
                break
            
            try:
                # 提取对话内容
                conversations = item.get("conversations", [])
                text = " ".join([c.get("value", "") for c in conversations[:2]])
                
                yield UnifiedSample(
                    id=f"{self.name}_{idx:05d}",
                    text=text,
                    video=item.get("video", None),
                    modality="video"
                )
            except Exception as e:
                logger.warning(f"Error processing sample {idx}: {e}")
                continue


# ============== 额外的 Mixed 数据集 ==============

class LRS3Downloader(DatasetDownloader):
    """LRS3 TED音视频语音数据集"""
    
    name = "lrs3"
    modality = "mixed"
    description = "TED/TEDx音视频语音识别数据集"
    url = "https://mmai.io/datasets/lip_reading/"
    
    def download(self) -> bool:
        try:
            logger.info("LRS3 requires agreement and download.")
            logger.info("Please visit: https://mmai.io/datasets/lip_reading/")
            
            data_path = self.data_dir / "lrs3"
            if data_path.exists():
                self.data_path = data_path
                return True
            return False
        except Exception as e:
            logger.error(f"Error with LRS3: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        mp4_files = list(self.data_path.rglob("*.mp4"))
        
        for idx, video_file in enumerate(mp4_files):
            if idx >= self.max_samples:
                break
            
            txt_file = video_file.with_suffix(".txt")
            text = ""
            if txt_file.exists():
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            
            yield UnifiedSample(
                id=f"{self.name}_{idx:05d}",
                text=text,
                audio=None,
                video=str(video_file),
                modality="mixed"
            )


class MELDDownloader(DatasetDownloader):
    """MELD 多模态情感对话数据集"""
    
    name = "meld"
    modality = "mixed"
    description = "多模态情感对话数据集 (Friends)"
    url = "https://github.com/declare-lab/MELD"
    
    def download(self) -> bool:
        try:
            logger.info("MELD requires download from GitHub.")
            logger.info("Please visit: https://github.com/declare-lab/MELD")
            
            data_path = self.data_dir / "meld"
            csv_file = data_path / "train_sent_emo.csv"
            
            if csv_file.exists():
                import pandas as pd
                self.data = pd.read_csv(csv_file)
                self.data_path = data_path
                return True
            return False
        except Exception as e:
            logger.error(f"Error with MELD: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        for idx, row in self.data.iterrows():
            if idx >= self.max_samples:
                break
            
            dialogue_id = row.get("Dialogue_ID", "")
            utterance_id = row.get("Utterance_ID", "")
            video_file = self.data_path / "videos" / f"dia{dialogue_id}_utt{utterance_id}.mp4"
            
            yield UnifiedSample(
                id=f"{self.name}_{idx:05d}",
                text=row.get("Utterance", ""),
                audio=None,
                video=str(video_file) if video_file.exists() else None,
                modality="mixed"
            )


class VALORDownloader(DatasetDownloader):
    """VALOR-1M 三模态预训练数据集"""
    
    name = "valor"
    modality = "mixed"
    description = "Vision-Audio-Language三模态数据集"
    url = "https://arxiv.org/abs/2304.08345"
    
    def download(self) -> bool:
        try:
            logger.info("VALOR-1M requires downloading from official sources.")
            logger.info("Please check: https://arxiv.org/abs/2304.08345")
            
            data_path = self.data_dir / "valor"
            annotations_file = data_path / "valor_annotations.json"
            
            if annotations_file.exists():
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    self.annotations = json.load(f)
                self.data_path = data_path
                return True
            return False
        except Exception as e:
            logger.error(f"Error with VALOR: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        for idx, item in enumerate(self.annotations):
            if idx >= self.max_samples:
                break
            
            video_id = item.get("video_id", "")
            video_path = self.data_path / "videos" / f"{video_id}.mp4"
            audio_path = self.data_path / "audio" / f"{video_id}.wav"
            
            yield UnifiedSample(
                id=f"{self.name}_{idx:05d}",
                text=item.get("caption", ""),
                audio=str(audio_path) if audio_path.exists() else None,
                video=str(video_path) if video_path.exists() else None,
                modality="mixed"
            )


class HowTo100MDownloader(DatasetDownloader):
    """HowTo100M 大规模教学视频数据集"""
    
    name = "howto100m"
    modality = "mixed"
    description = "1.36亿视频片段的教学视频数据集"
    url = "https://www.di.ens.fr/willow/research/howto100m/"
    
    def download(self) -> bool:
        try:
            logger.info("HowTo100M is a very large dataset.")
            logger.info("Please visit: https://www.di.ens.fr/willow/research/howto100m/")
            
            data_path = self.data_dir / "howto100m"
            captions_file = data_path / "caption_howto100m_with_cap.json"
            
            if captions_file.exists():
                with open(captions_file, 'r', encoding='utf-8') as f:
                    self.captions = json.load(f)
                self.data_path = data_path
                return True
            return False
        except Exception as e:
            logger.error(f"Error with HowTo100M: {e}")
            return False
    
    def convert(self) -> Generator[UnifiedSample, None, None]:
        video_ids = list(self.captions.keys())[:self.max_samples]
        
        for idx, video_id in enumerate(video_ids):
            if idx >= self.max_samples:
                break
            
            video_path = self.data_path / "videos" / f"{video_id}.mp4"
            
            # 获取第一个caption
            caps = self.captions.get(video_id, {}).get("text", [""])
            text = caps[0] if caps else ""
            
            yield UnifiedSample(
                id=f"{self.name}_{idx:05d}",
                text=text,
                audio=None,
                video=str(video_path) if video_path.exists() else None,
                modality="mixed"
            )


# 注册扩展数据集
EXTENDED_DATASETS = {
    # Audio
    "voxpopuli": VoxPopuliDownloader,
    "aishell": AishellDownloader,
    "covost2": CoVoST2Downloader,
    
    # Video
    "youcook2": YouCook2Downloader,
    "vatex": VATEXDownloader,
    "activitynet_qa": ActivityNetQADownloader,
    "videochat2_it": VideoChat2ITDownloader,
    
    # Mixed
    "lrs3": LRS3Downloader,
    "meld": MELDDownloader,
    "valor": VALORDownloader,
    "howto100m": HowTo100MDownloader,
}

# 合并到主注册表
DATASET_REGISTRY.update(EXTENDED_DATASETS)


def register_all_datasets():
    """注册所有扩展数据集到主注册表"""
    from download_datasets import DATASET_REGISTRY as main_registry
    main_registry.update(EXTENDED_DATASETS)
    return main_registry


if __name__ == "__main__":
    # 打印扩展数据集信息
    print("Extended datasets available:")
    for name, cls in EXTENDED_DATASETS.items():
        print(f"  - {name}: {cls.description}")
