#!/usr/bin/env python3
"""
WavCaps dataset processor.

Dataset: cvssp/WavCaps
Format: Audio files + JSON annotations
Task: Audio captioning (not ASR)
Structure:
    - audio: audio file data
    - caption: text description of the audio content
    - id: unique identifier
    - source: original source (freesound, bbc, audioset, etc.)

This is an audio CAPTIONING dataset, not transcription.
The text describes WHAT IS HEARD, not spoken words.
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


class WavCapsProcessor(ParquetProcessor):
    """
    Processor for WavCaps audio captioning dataset.
    
    WavCaps contains audio clips with natural language descriptions.
    Sources include FreeSound, BBC Sound Effects, AudioSet, etc.
    
    Structure after download:
        WavCaps/
        ├── data/
        │   ├── FreeSound/
        │   │   ├── audio/
        │   │   └── metadata.json
        │   ├── BBC_Sound_Effects/
        │   └── ...
        └── *.parquet (if available)
    
    Parquet/JSON schema:
        - id: str
        - caption: str (natural language description)
        - audio: audio data (bytes or array)
        - source: str (freesound, bbc, audioset, etc.)
        - duration: float (optional)
    """
    
    def __init__(self, data_dir: Path, config: ProcessorConfig):
        super().__init__(data_dir, config)
        
        self.audio_output_dir = config.output_dir / config.name / "audio"
        self.audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "name": "wavcaps",
            "full_name": "WavCaps Audio Captioning Dataset",
            "hf_id": "cvssp/WavCaps",
            "modality": "audio",
            "language": "en",
            "task": "audio_captioning",
            "description": "Audio clips with natural language descriptions",
            "license": "Academic",
        }
    
    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples - try parquet first, then JSON."""
        # Try parquet files
        parquet_files = self.find_parquet_files()
        if parquet_files:
            logger.info(f"Found {len(parquet_files)} parquet files")
            for pq_file in parquet_files:
                yield from self.iter_parquet_rows(pq_file)
            return
        
        # Try JSON metadata files
        json_files = list(self.data_dir.rglob("*.json"))
        if json_files:
            logger.info(f"Found {len(json_files)} JSON files")
            for json_file in json_files:
                yield from self._iter_json_samples(json_file)
            return
        
        logger.error(f"No data files found in {self.data_dir}")
    
    def _iter_json_samples(self, json_path: Path) -> Iterator[Dict[str, Any]]:
        """Iterate over samples from a JSON file."""
        import json
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                yield from data
            elif isinstance(data, dict):
                if "data" in data:
                    yield from data["data"]
                elif "samples" in data:
                    yield from data["samples"]
                else:
                    # Single sample dict
                    yield data
        except Exception as e:
            logger.warning(f"Error reading {json_path}: {e}")
    
    def process_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Sample]:
        """Convert raw WavCaps sample to unified format."""
        try:
            # Get caption
            caption = raw_sample.get("caption", "")
            if not caption:
                caption = raw_sample.get("text", "")
            if not caption:
                return None
            
            # Get sample ID
            sample_id = raw_sample.get("id", f"wavcaps_{idx:08d}")
            
            # Get audio
            audio_data = raw_sample.get("audio", {})
            audio_path = None
            
            if isinstance(audio_data, dict):
                audio_bytes = audio_data.get("bytes")
                audio_array = audio_data.get("array")
                sample_rate = audio_data.get("sampling_rate", 32000)  # WavCaps often uses 32kHz
                
                audio_filename = f"{sample_id}.wav"
                output_audio_path = self.audio_output_dir / audio_filename
                
                if audio_bytes:
                    output_audio_path.write_bytes(audio_bytes)
                    audio_path = output_audio_path
                elif audio_array is not None:
                    import numpy as np
                    if save_audio_wav(np.array(audio_array), sample_rate, output_audio_path):
                        audio_path = output_audio_path
            elif isinstance(audio_data, (str, Path)):
                # Direct path reference
                source_path = Path(audio_data)
                if source_path.is_absolute() and source_path.exists():
                    audio_path = source_path
                else:
                    # Try relative to data_dir
                    full_path = self.data_dir / source_path
                    if full_path.exists():
                        audio_path = full_path
            
            if audio_path is None:
                return None
            
            return Sample(
                id=sample_id,
                text=caption,
                audio_path=normalize_path(audio_path),
                language="en",
                extra={
                    "source": raw_sample.get("source", ""),
                    "duration": raw_sample.get("duration"),
                },
            )
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            return None
    
    def get_user_prompt(self, sample: Sample) -> str:
        """WavCaps specific prompt for audio captioning."""
        return "<audio>Describe what you hear in this audio."
