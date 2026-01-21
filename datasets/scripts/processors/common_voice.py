#!/usr/bin/env python3
"""
Common Voice dataset processor.

Dataset: mozilla-foundation/common_voice_17_0 (or other versions)

As of CV 17.0, the dataset is distributed via Mozilla Data Collective (MDC).
The archive structure is:
    cv-corpus-17.0-<date>/
    ├── <language>/           # e.g., "en", "zh-CN"
    │   ├── clips/            # MP3 audio files
    │   │   └── common_voice_<lang>_*.mp3
    │   ├── train.tsv         # Training split metadata
    │   ├── test.tsv          # Test split metadata
    │   ├── validated.tsv     # Validated samples
    │   └── dev.tsv           # Development split
    └── ...

TSV Columns:
    - client_id: anonymized speaker ID
    - path: MP3 filename (e.g., "common_voice_en_12345.mp3")
    - sentence: transcription text
    - up_votes, down_votes: quality votes
    - age, gender, accent: speaker metadata (optional)
    - segment: segment info (optional)

Note: Earlier versions may still be on HuggingFace with parquet format.
This processor supports both TSV+MP3 and parquet formats.
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


class CommonVoiceProcessor(ParquetProcessor):
    """
    Processor for Mozilla Common Voice dataset.
    
    Supports two formats:
    
    1. MDC Archive format (CV 17.0+):
        cv-corpus-17.0-*/
        └── <lang>/
            ├── clips/                   # MP3 audio files
            ├── train.tsv, test.tsv, ... # Split metadata
    
    2. Older HuggingFace parquet format:
        common_voice_*/
        └── <lang>/
            └── <split>/*.parquet
    
    TSV schema (MDC format):
        - client_id, path, sentence, up_votes, down_votes,
          age, gender, accent, segment
    """
    
    def __init__(self, data_dir: Path, config: ProcessorConfig):
        super().__init__(data_dir, config)
        
        self.audio_output_dir = config.output_dir / config.name / "audio"
        self.audio_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Language/locale from subset config
        self.locale = config.subset or "en"
        
        # Detect format
        self._format = self._detect_format()
    
    def _detect_format(self) -> str:
        """Detect whether this is TSV or parquet format."""
        # Check for TSV files
        tsv_files = list(self.data_dir.rglob("*.tsv"))
        if tsv_files:
            return "tsv"
        
        # Check for parquet files
        parquet_files = list(self.data_dir.rglob("*.parquet"))
        if parquet_files:
            return "parquet"
        
        return "unknown"
    
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "name": "common_voice",
            "full_name": "Mozilla Common Voice",
            "hf_id": "mozilla-foundation/common_voice_17_0",
            "modality": "audio",
            "language": self.locale,
            "task": "asr",
            "description": "Multilingual crowdsourced speech dataset",
            "license": "CC0-1.0",
            "format": self._format,
        }
    
    def find_audio_dir(self) -> Optional[Path]:
        """Find the directory containing audio clips."""
        patterns = [
            self.data_dir / self.locale / "clips",
            self.data_dir / "clips",
            self.data_dir / self.locale / "audio",
        ]
        
        for pattern in patterns:
            if pattern.exists():
                return pattern
        
        # Search for mp3 files
        mp3_files = list(self.data_dir.rglob("*.mp3"))
        if mp3_files:
            return mp3_files[0].parent
        
        return None
    
    def find_tsv_file(self) -> Optional[Path]:
        """Find the TSV file for the configured split."""
        split = self.config.split
        
        # Map split names
        split_map = {
            "train": "train.tsv",
            "test": "test.tsv",
            "validation": "validated.tsv",
            "validated": "validated.tsv",
            "dev": "dev.tsv",
        }
        tsv_name = split_map.get(split, f"{split}.tsv")
        
        patterns = [
            self.data_dir / self.locale / tsv_name,
            self.data_dir / tsv_name,
        ]
        
        for pattern in patterns:
            if pattern.exists():
                return pattern
        
        # Search recursively
        found = list(self.data_dir.rglob(tsv_name))
        if found:
            return found[0]
        
        return None
    
    def iter_tsv_samples(self, tsv_path: Path) -> Iterator[Dict[str, Any]]:
        """Iterate over rows in a TSV file."""
        import csv
        
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                yield dict(row)
    
    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples - detect format and use appropriate method."""
        # Find audio directory first
        self.audio_source_dir = self.find_audio_dir()
        if self.audio_source_dir:
            logger.info(f"Audio source directory: {self.audio_source_dir}")
        
        if self._format == "tsv":
            tsv_file = self.find_tsv_file()
            if tsv_file:
                logger.info(f"Using TSV file: {tsv_file}")
                yield from self.iter_tsv_samples(tsv_file)
            else:
                logger.error(f"No TSV file found for split {self.config.split}")
        elif self._format == "parquet":
            yield from self._iter_parquet_samples()
        else:
            logger.error(f"Unknown format in {self.data_dir}")
    
    def _iter_parquet_samples(self) -> Iterator[Dict[str, Any]]:
        """Iterate over parquet files (older HF format)."""
        parquet_files = self.find_parquet_files()
        if parquet_files:
            logger.info(f"Found {len(parquet_files)} parquet files")
            for pq_file in parquet_files:
                yield from self.iter_parquet_rows(pq_file)
    
    def process_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Sample]:
        """Convert raw Common Voice sample to unified format."""
        try:
            # Get transcription
            text = raw_sample.get("sentence", "")
            if not text:
                return None
            
            sample_id = f"cv_{self.locale}_{idx:08d}"
            audio_path = None
            
            # TSV format: 'path' contains MP3 filename
            audio_filename = raw_sample.get("path", "")
            
            if audio_filename and self.audio_source_dir:
                # Find source audio file
                source_audio = self.audio_source_dir / audio_filename
                if source_audio.exists():
                    audio_path = source_audio
                else:
                    # Try just the filename without path
                    source_audio = self.audio_source_dir / Path(audio_filename).name
                    if source_audio.exists():
                        audio_path = source_audio
            
            # Parquet format: 'audio' contains embedded data
            audio_data = raw_sample.get("audio", {})
            if isinstance(audio_data, dict) and audio_data and audio_path is None:
                audio_bytes = audio_data.get("bytes")
                audio_array = audio_data.get("array")
                sample_rate = audio_data.get("sampling_rate", 16000)
                
                if audio_bytes:
                    # MP3 bytes - save directly
                    mp3_path = self.audio_output_dir / f"{sample_id}.mp3"
                    mp3_path.write_bytes(audio_bytes)
                    audio_path = mp3_path
                elif audio_array is not None:
                    import numpy as np
                    wav_path = self.audio_output_dir / f"{sample_id}.wav"
                    if save_audio_wav(np.array(audio_array), sample_rate, wav_path):
                        audio_path = wav_path
            
            if audio_path is None:
                return None
            
            return Sample(
                id=sample_id,
                text=text,
                audio_path=normalize_path(audio_path),
                speaker_id=raw_sample.get("client_id", ""),
                language=raw_sample.get("locale", self.locale),
                extra={
                    "age": raw_sample.get("age"),
                    "gender": raw_sample.get("gender"),
                    "accent": raw_sample.get("accent"),
                    "up_votes": int(raw_sample.get("up_votes", 0) or 0),
                    "down_votes": int(raw_sample.get("down_votes", 0) or 0),
                },
            )
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            return None
    
    def get_user_prompt(self, sample: Sample) -> str:
        """Common Voice specific prompt with language awareness."""
        lang = sample.language or self.locale
        if lang == "en":
            return "<audio>Transcribe this English speech."
        elif lang == "zh-CN" or lang == "zh":
            return "<audio>请转写这段中文语音。"
        else:
            return f"<audio>Transcribe this {lang} speech."
