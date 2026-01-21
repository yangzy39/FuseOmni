#!/usr/bin/env python3
"""
AISHELL-1 dataset processor.

Dataset: AISHELL/AISHELL-1 (or similar paths)
Format: Audio files + transcription files
Task: Chinese Mandarin ASR
Structure:
    - audio: WAV audio data
    - text: Chinese transcription
    - speaker_id: speaker identifier

AISHELL-1 is a commonly used Chinese ASR benchmark dataset.
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


class AISHELL1Processor(ParquetProcessor):
    """
    Processor for AISHELL-1 Chinese ASR dataset.
    
    Structure after download (may vary):
        aishell/
        ├── data/
        │   ├── train/
        │   │   ├── *.parquet
        │   │   └── ...
        │   ├── dev/
        │   └── test/
        └── audio/
            └── wav/
                └── train/
                    └── S0001/
                        └── *.wav
    
    Or original format:
        data_aishell/
        ├── wav/
        │   └── train/
        │       └── S0001/
        │           └── BAC009S0001W0001.wav
        └── transcript/
            └── aishell_transcript_v0.8.txt
    
    Parquet schema (if available):
        - audio: struct{array, sampling_rate, path}
        - text: str (Chinese characters)
        - speaker_id: str
    """
    
    def __init__(self, data_dir: Path, config: ProcessorConfig):
        super().__init__(data_dir, config)
        
        self.audio_output_dir = config.output_dir / config.name / "audio"
        self.audio_output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "name": "aishell1",
            "full_name": "AISHELL-1",
            "hf_id": "AISHELL/AISHELL-1",
            "modality": "audio",
            "language": "zh",
            "task": "asr",
            "description": "Chinese Mandarin ASR dataset",
            "license": "Apache-2.0",
        }
    
    def iter_samples(self) -> Iterator[Dict[str, Any]]:
        """Iterate over samples - try parquet first, then original format."""
        # Try parquet files
        parquet_files = self.find_parquet_files()
        if parquet_files:
            logger.info(f"Found {len(parquet_files)} parquet files")
            for pq_file in parquet_files:
                yield from self.iter_parquet_rows(pq_file)
            return
        
        # Try original AISHELL format
        yield from self._iter_original_format()
    
    def _iter_original_format(self) -> Iterator[Dict[str, Any]]:
        """Iterate using original AISHELL directory structure."""
        # Find transcript file
        transcript_files = list(self.data_dir.rglob("*transcript*.txt"))
        if not transcript_files:
            logger.error("No transcript file found")
            return
        
        transcript_file = transcript_files[0]
        logger.info(f"Using transcript: {transcript_file}")
        
        # Load transcripts
        transcripts = {}
        with open(transcript_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt_id, text = parts
                    transcripts[utt_id] = text
        
        # Find audio files
        wav_files = list(self.data_dir.rglob("*.wav"))
        logger.info(f"Found {len(wav_files)} WAV files")
        
        for wav_file in wav_files:
            utt_id = wav_file.stem
            if utt_id in transcripts:
                yield {
                    "id": utt_id,
                    "audio_path": str(wav_file),
                    "text": transcripts[utt_id],
                    "speaker_id": utt_id[:7] if len(utt_id) > 7 else "",  # e.g., S0001
                }
    
    def process_sample(self, raw_sample: Dict[str, Any], idx: int) -> Optional[Sample]:
        """Convert raw AISHELL sample to unified format."""
        try:
            text = raw_sample.get("text", "")
            if not text:
                return None
            
            sample_id = raw_sample.get("id", f"aishell_{idx:08d}")
            
            # Handle audio
            audio_data = raw_sample.get("audio", {})
            audio_path_raw = raw_sample.get("audio_path")
            
            if isinstance(audio_data, dict) and audio_data:
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
            elif audio_path_raw:
                # Use existing audio file
                audio_path = Path(audio_path_raw)
                if not audio_path.exists():
                    return None
            else:
                return None
            
            return Sample(
                id=sample_id,
                text=text,
                audio_path=normalize_path(audio_path),
                speaker_id=raw_sample.get("speaker_id", ""),
                language="zh",
            )
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            return None
    
    def get_user_prompt(self, sample: Sample) -> str:
        """AISHELL specific prompt."""
        return "<audio>请转写这段中文语音。"
