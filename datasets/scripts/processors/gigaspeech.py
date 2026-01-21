#!/usr/bin/env python3
"""
GigaSpeech dataset processor.

Dataset: speechcolab/gigaspeech
Format: Audio segments + transcriptions
Task: ASR
Structure:
    - audio: audio data
    - text: transcription
    - segment_id: unique segment identifier
    - speaker: speaker ID (optional)
    - begin_time, end_time: timestamps

Subsets: xs (10h), s (250h), m (1000h), l (2500h), xl (10000h)
Requires HF authentication.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from .base import (
    BaseProcessor,
    ParquetProcessor,
    ProcessorConfig,
    Sample,
    normalize_path,
    save_audio_wav,
)

logger = logging.getLogger(__name__)


class GigaSpeechProcessor(ParquetProcessor):
    """
    Processor for GigaSpeech ASR dataset.
    
    GigaSpeech is a large-scale English ASR dataset with audio from
    audiobooks, podcasts, and YouTube.
    
    Structure after download:
        gigaspeech/
        ├── xs/  (or s, m, l, xl)
        │   ├── train/
        │   │   ├── *.parquet
        │   │   └── ...
        │   ├── validation/
        │   └── test/
        └── audio/
            └── ...
    
    Parquet schema:
        - segment_id: str
        - audio: struct{array, sampling_rate, path}
        - text: str
        - speaker: str (optional)
        - begin_time: float
        - end_time: float
    """
    
    def __init__(self, data_dir: Path, config: ProcessorConfig):
        super().__init__(data_dir, config)
        
        self.audio_output_dir = config.output_dir / config.name / "audio"
        self.audio_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default subset is 'xs' (smallest)
        self.subset = config.subset or "xs"
    
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "name": "gigaspeech",
            "full_name": "GigaSpeech",
            "hf_id": "speechcolab/gigaspeech",
            "modality": "audio",
            "language": "en",
            "task": "asr",
            "description": "Large-scale English ASR from diverse sources",
            "license": "Apache-2.0",
            "requires_auth": True,
            "subsets": ["xs", "s", "m", "l", "xl"],
        }
    
    def find_split_parquets(self) -> list[Path]:
        """Find parquet files for the configured split and subset."""
        split = self.config.split
        
        patterns = [
            self.data_dir / self.subset / split,
            self.data_dir / "data" / self.subset / split,
            self.data_dir / split,
        ]
        
        for pattern in patterns:
            if pattern.exists():
                files = list(pattern.glob("*.parquet"))
                if files:
                    return sorted(files)
        
        # Fallback
        return sorted(self.data_dir.rglob(f"*{split}*.parquet"))
    
    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples from parquet files."""
        parquet_files = self.find_split_parquets()
        
        if not parquet_files:
            logger.error(f"No parquet files found for {self.subset}/{self.config.split}")
            return
        
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        for pq_file in parquet_files:
            yield from self.iter_parquet_rows(pq_file)
    
    def process_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Sample]:
        """Convert raw GigaSpeech sample to unified format."""
        try:
            # Get text
            text = raw_sample.get("text", "")
            if not text:
                return None
            
            # Get sample ID
            sample_id = raw_sample.get("segment_id", f"gigaspeech_{idx:08d}")
            
            # Get audio
            audio_data = raw_sample.get("audio", {})
            
            if not isinstance(audio_data, dict):
                return None
            
            audio_bytes = audio_data.get("bytes")
            audio_array = audio_data.get("array")
            sample_rate = audio_data.get("sampling_rate", 16000)
            
            audio_filename = f"{sample_id}.wav"
            audio_path = self.audio_output_dir / audio_filename
            
            if audio_bytes:
                audio_path.write_bytes(audio_bytes)
            elif audio_array is not None:
                import numpy as np
                if not save_audio_wav(np.array(audio_array), sample_rate, audio_path):
                    return None
            else:
                return None
            
            return Sample(
                id=sample_id,
                text=text,
                audio_path=normalize_path(audio_path),
                speaker_id=raw_sample.get("speaker", ""),
                language="en",
                duration=raw_sample.get("end_time", 0) - raw_sample.get("begin_time", 0),
                extra={
                    "begin_time": raw_sample.get("begin_time"),
                    "end_time": raw_sample.get("end_time"),
                },
            )
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            return None
    
    def get_user_prompt(self, sample: Sample) -> str:
        """GigaSpeech specific prompt."""
        return "<audio>Transcribe the following English speech."
