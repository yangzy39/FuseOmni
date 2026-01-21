#!/usr/bin/env python3
"""
LibriSpeech dataset processor.

Dataset: openslr/librispeech_asr
Format: Parquet files with embedded audio bytes
Structure:
    - audio: dict with 'array' (samples), 'sampling_rate', 'path'
    - text: transcription string
    - speaker_id: int
    - chapter_id: int
    - id: unique identifier string

Subsets: clean, other
Splits: train.clean.100, train.clean.360, train.other.500, 
        validation.clean, validation.other, test.clean, test.other
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


class LibriSpeechProcessor(ParquetProcessor):
    """
    Processor for LibriSpeech ASR dataset.
    
    After huggingface-cli download, the structure is:
        librispeech_asr/
        ├── clean/
        │   ├── train.clean.100/
        │   │   ├── train.clean.100-00000-of-XXXXX.parquet
        │   │   └── ...
        │   ├── validation.clean/
        │   └── test.clean/
        └── other/
            ├── train.other.500/
            └── ...
    
    Parquet schema:
        - file: str (original filename)
        - audio: struct{bytes, path, sampling_rate} 
        - text: str
        - speaker_id: int
        - chapter_id: int
        - id: str
    """
    
    def __init__(self, data_dir: Path, config: ProcessorConfig):
        super().__init__(data_dir, config)
        
        # Setup audio output directory
        self.audio_output_dir = config.output_dir / config.name / "audio"
        self.audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "name": "librispeech",
            "full_name": "LibriSpeech ASR",
            "hf_id": "openslr/librispeech_asr",
            "modality": "audio",
            "language": "en",
            "task": "asr",
            "description": "English audiobook ASR dataset from LibriVox",
            "license": "CC-BY-4.0",
            "subsets": ["clean", "other"],
        }
    
    def find_split_dir(self) -> Optional[Path]:
        """
        Find the directory for the configured split.
        
        LibriSpeech structure after huggingface-cli download:
            librispeech_asr/
            ├── clean/
            │   ├── train.100/  (or train.360, validation, test)
            │   │   └── *.parquet
            │   └── ...
            ├── other/
            │   └── train.500/, validation/, test/
            └── all/
                └── train.clean.100/, train.clean.360/, train.other.500/, etc.
        """
        split = self.config.split
        subset = self.config.subset or "clean"
        
        # Normalize split name (handle both "train.clean.100" and "train.100" formats)
        # For "all" config, splits are like "train.clean.100"
        # For "clean" config, splits are like "train.100"
        
        # Pattern 1: all/split/ (e.g., all/train.clean.100/)
        candidate = self.data_dir / "all" / split
        if candidate.exists():
            return candidate
        
        # Pattern 2: subset/split/ with short name (e.g., clean/train.100/)
        # Convert "train.clean.100" -> "train.100" for subset configs
        short_split = split.replace(f".{subset}", "")
        candidate = self.data_dir / subset / short_split
        if candidate.exists():
            return candidate
        
        # Pattern 3: Direct split (e.g., train.100/)
        candidate = self.data_dir / short_split
        if candidate.exists():
            return candidate
        
        # Pattern 4: subset/split/ with original name
        candidate = self.data_dir / subset / split
        if candidate.exists():
            return candidate
        
        # Pattern 5: Look for any parquet file matching split name
        for pattern in [f"*{split}*", f"*{short_split}*"]:
            parquet_files = list(self.data_dir.rglob(f"{pattern}/*.parquet"))
            if parquet_files:
                return parquet_files[0].parent
        
        logger.warning(f"Could not find split directory for {split} (subset={subset})")
        return None
    
    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples from parquet files."""
        split_dir = self.find_split_dir()
        
        if split_dir is None:
            # Fall back to searching all parquet files
            parquet_files = self.find_parquet_files()
        else:
            parquet_files = list(split_dir.glob("*.parquet"))
        
        if not parquet_files:
            logger.error(f"No parquet files found in {self.data_dir}")
            return
        
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        for pq_file in parquet_files:
            yield from self.iter_parquet_rows(pq_file)
    
    def process_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Sample]:
        """Convert raw LibriSpeech sample to unified format."""
        try:
            # Extract audio data
            audio_data = raw_sample.get("audio", {})
            
            if isinstance(audio_data, dict):
                # Format 1: dict with 'bytes' or 'array'
                audio_bytes = audio_data.get("bytes")
                audio_array = audio_data.get("array")
                sample_rate = audio_data.get("sampling_rate", 16000)
                original_path = audio_data.get("path", "")
            else:
                logger.warning(f"Unexpected audio format: {type(audio_data)}")
                return None
            
            # Get text
            text = raw_sample.get("text", "")
            if not text:
                logger.debug(f"Sample {idx} has no text, skipping")
                return None
            
            # Generate output audio path
            sample_id = raw_sample.get("id", f"librispeech_{idx:08d}")
            audio_filename = f"{sample_id}.wav"
            audio_path = self.audio_output_dir / audio_filename
            
            # Save audio file
            if audio_bytes:
                # Audio stored as FLAC bytes in parquet
                # The original files are .flac, so bytes are FLAC-encoded
                # Save as .flac first, then convert to .wav if needed
                flac_path = self.audio_output_dir / f"{sample_id}.flac"
                flac_path.write_bytes(audio_bytes)
                
                # Try to convert to WAV using soundfile
                try:
                    import soundfile as sf
                    audio_array_loaded, sr = sf.read(str(flac_path))
                    sf.write(str(audio_path), audio_array_loaded, sr, format="WAV")
                    flac_path.unlink()  # Remove temp flac
                except Exception as e:
                    # If conversion fails, just use the flac file
                    logger.debug(f"FLAC->WAV conversion failed, keeping FLAC: {e}")
                    audio_path = flac_path
                    
            elif audio_array is not None:
                # Audio stored as numpy array
                import numpy as np
                if not save_audio_wav(np.array(audio_array), sample_rate, audio_path):
                    return None
            else:
                logger.warning(f"Sample {idx} has no audio data")
                return None
            
            return Sample(
                id=sample_id,
                text=text,
                audio_path=normalize_path(audio_path),
                speaker_id=str(raw_sample.get("speaker_id", "")),
                language="en",
                extra={
                    "chapter_id": raw_sample.get("chapter_id"),
                    "file": raw_sample.get("file", ""),
                },
            )
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            return None
    
    def get_user_prompt(self, sample: Sample) -> str:
        """LibriSpeech specific prompt."""
        return "<audio>Transcribe the following audio exactly as spoken."
