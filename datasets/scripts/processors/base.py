#!/usr/bin/env python3
"""
Base classes for dataset processors.

Each dataset has its own unique format. Processors understand
the specific structure and convert to unified MS-SWIFT format.

Architecture:
    1. download.sh uses huggingface-cli to download raw datasets
    2. Each processor reads its dataset's specific format
    3. Converts to MS-SWIFT SFT/GRPO format with absolute paths
"""

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Literal

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Configuration for dataset processor."""
    
    # Dataset identification
    name: str
    hf_id: str  # e.g., "openslr/librispeech_asr"
    
    # Processing options
    max_samples: int = -1  # -1 means all samples
    split: str = "train"
    subset: Optional[str] = None  # e.g., "clean" for librispeech
    
    # Output options
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    task_type: Literal["sft", "grpo"] = "sft"
    system_prompt: Optional[str] = None
    
    # Audio processing
    target_sample_rate: int = 16000
    audio_format: str = "wav"


@dataclass
class Sample:
    """Unified sample representation."""
    
    id: str
    text: str
    audio_path: Optional[Path] = None
    video_path: Optional[Path] = None
    
    # Metadata
    speaker_id: Optional[str] = None
    language: Optional[str] = None
    duration: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def normalize_path(path: str | Path) -> Path:
    """Normalize path for cross-platform compatibility."""
    return Path(path).resolve()


def save_audio_wav(audio_array: np.ndarray, sample_rate: int, output_path: Path) -> bool:
    """Save audio array to WAV file."""
    try:
        import soundfile as sf
        audio = np.asarray(audio_array, dtype=np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))
        sf.write(str(output_path), audio, sample_rate, format="WAV")
        return True
    except ImportError:
        try:
            from scipy.io import wavfile
            audio = np.asarray(audio_array, dtype=np.float32)
            if audio.max() <= 1.0 and audio.min() >= -1.0:
                audio = (audio * 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)
            wavfile.write(str(output_path), sample_rate, audio)
            return True
        except ImportError:
            logger.error("Neither soundfile nor scipy available")
            return False


def create_msswift_sample(
    user_content: str,
    assistant_content: Optional[str] = None,
    system_prompt: Optional[str] = None,
    audios: Optional[List[str]] = None,
    videos: Optional[List[str]] = None,
    images: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Create a single MS-SWIFT format sample."""
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": user_content})
    
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})
    
    result = {"messages": messages}
    
    if audios:
        result["audios"] = audios
    if videos:
        result["videos"] = videos
    if images:
        result["images"] = images
    
    return result


class BaseProcessor(ABC):
    """
    Base class for dataset processors.
    
    Each processor must implement:
        - get_dataset_info(): Return dataset metadata
        - iter_samples(): Iterate over raw samples
        - process_sample(): Convert raw sample to Sample object
    """
    
    def __init__(self, data_dir: Path, config: ProcessorConfig):
        """
        Initialize processor.
        
        Args:
            data_dir: Directory containing downloaded dataset
            config: Processing configuration
        """
        self.data_dir = normalize_path(data_dir)
        self.config = config
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
    
    @abstractmethod
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Return dataset metadata.
        
        Returns:
            Dict with keys: name, description, modality, splits, etc.
        """
        pass
    
    @abstractmethod
    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over raw samples from the dataset.
        
        Yields:
            Raw sample dictionaries in dataset's native format
        """
        pass
    
    @abstractmethod
    def process_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Sample]:
        """
        Convert a raw sample to unified Sample format.
        
        Args:
            raw_sample: Raw sample in dataset's native format
            idx: Sample index
            
        Returns:
            Sample object or None if processing failed
        """
        pass
    
    def get_user_prompt(self, sample: Sample) -> str:
        """
        Get user prompt for the sample.
        Override this to customize prompts per dataset.
        """
        if sample.audio_path:
            return "<audio>What did the audio say?"
        elif sample.video_path:
            return "<video>Describe what happens in this video."
        else:
            return "Please respond."
    
    def to_msswift(self, sample: Sample) -> Dict[str, Any]:
        """Convert Sample to MS-SWIFT format."""
        user_content = self.get_user_prompt(sample)
        
        # For SFT, include response; for GRPO, only prompt
        assistant_content = sample.text if self.config.task_type == "sft" else None
        
        audios = [str(sample.audio_path)] if sample.audio_path else None
        videos = [str(sample.video_path)] if sample.video_path else None
        
        return create_msswift_sample(
            user_content=user_content,
            assistant_content=assistant_content,
            system_prompt=self.config.system_prompt,
            audios=audios,
            videos=videos,
        )
    
    def process(self) -> Dict[str, int]:
        """
        Main processing method.
        
        Returns:
            Statistics dictionary with processed/skipped/error counts
        """
        output_dir = self.config.output_dir / self.config.name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"{self.config.task_type}.jsonl"
        
        stats = {"total": 0, "processed": 0, "skipped": 0, "errors": 0}
        
        logger.info(f"Processing {self.config.name} from {self.data_dir}")
        logger.info(f"Output: {output_file}")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for idx, raw_sample in enumerate(self.iter_samples()):
                # Check max samples
                if self.config.max_samples > 0 and stats["processed"] >= self.config.max_samples:
                    break
                
                stats["total"] += 1
                
                try:
                    sample = self.process_sample(raw_sample, idx)
                    
                    if sample is None:
                        stats["skipped"] += 1
                        continue
                    
                    msswift_sample = self.to_msswift(sample)
                    f.write(json.dumps(msswift_sample, ensure_ascii=False) + "\n")
                    stats["processed"] += 1
                    
                    if stats["processed"] % 100 == 0:
                        logger.info(f"  Processed {stats['processed']} samples...")
                        
                except Exception as e:
                    logger.warning(f"Error processing sample {idx}: {e}")
                    stats["errors"] += 1
        
        logger.info(
            f"Completed {self.config.name}: "
            f"{stats['processed']} processed, "
            f"{stats['skipped']} skipped, "
            f"{stats['errors']} errors"
        )
        
        # Save metadata
        meta_file = output_dir / "metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({
                "dataset": self.get_dataset_info(),
                "config": {
                    "name": self.config.name,
                    "hf_id": self.config.hf_id,
                    "split": self.config.split,
                    "subset": self.config.subset,
                    "max_samples": self.config.max_samples,
                    "task_type": self.config.task_type,
                },
                "stats": stats,
            }, f, indent=2, ensure_ascii=False)
        
        return stats


class ParquetProcessor(BaseProcessor):
    """
    Base processor for datasets stored in parquet format.
    
    Many HuggingFace datasets use parquet files after download.
    This provides common utilities for reading parquet data.
    """
    
    def find_parquet_files(self, pattern: str = "*.parquet") -> List[Path]:
        """Find all parquet files matching pattern."""
        files = list(self.data_dir.rglob(pattern))
        return sorted(files)
    
    def iter_parquet_rows(self, file_path: Path) -> Iterator[Dict[str, Any]]:
        """Iterate over rows in a parquet file."""
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(file_path)
            for batch in table.to_batches():
                for row_idx in range(batch.num_rows):
                    yield {col: batch.column(col)[row_idx].as_py() 
                           for col in batch.schema.names}
        except ImportError:
            try:
                import pandas as pd
                df = pd.read_parquet(file_path)
                for _, row in df.iterrows():
                    yield row.to_dict()
            except ImportError:
                raise ImportError("pyarrow or pandas required for parquet processing")
    
    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        """Default implementation: iterate over all parquet files."""
        parquet_files = self.find_parquet_files()
        
        if not parquet_files:
            logger.warning(f"No parquet files found in {self.data_dir}")
            return
        
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        for pq_file in parquet_files:
            logger.debug(f"Reading {pq_file}")
            yield from self.iter_parquet_rows(pq_file)
