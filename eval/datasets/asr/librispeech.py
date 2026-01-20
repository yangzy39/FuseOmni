"""
LibriSpeech ASR dataset.
"""

from typing import Iterator, Any
from pathlib import Path
import os

from ..base import BaseDataset
from ..common_types import normalize_text
from ...schema import EvalSample
from ...registry import register_dataset


@register_dataset("librispeech_clean")
class LibriSpeechCleanDataset(BaseDataset):
    """
    LibriSpeech test-clean dataset for ASR evaluation.
    
    Contains ~2,620 samples of clean read English speech.
    
    Source: https://huggingface.co/datasets/openslr/librispeech_asr
    """
    
    name = "librispeech_clean"
    task_type = "asr"
    default_split = "test.clean"
    metrics = ["wer", "cer"]
    language = "en"
    description = "LibriSpeech test-clean (clean read English speech)"
    
    prompt_template = "Transcribe the following audio exactly as spoken."
    
    def load_from_hf(self) -> Iterator[EvalSample]:
        """Load LibriSpeech test-clean from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        # Load the dataset
        dataset = load_dataset(
            "openslr/librispeech_asr",
            "clean",
            split="test",
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            # Get audio path or use in-memory audio
            audio = sample["audio"]
            audio_path = audio.get("path", f"librispeech_clean_{idx}")
            
            yield EvalSample(
                id=f"librispeech_clean_{idx}",
                audio_path=audio_path,
                text_prompt=self.build_prompt(sample),
                reference=self.get_reference(sample),
                meta={
                    "speaker_id": sample.get("speaker_id"),
                    "chapter_id": sample.get("chapter_id"),
                    "audio_array": audio["array"],
                    "sampling_rate": audio["sampling_rate"],
                },
            )
            count += 1
    
    def get_reference(self, sample: Any) -> str:
        """Extract transcript from sample."""
        return sample["text"]
    
    def get_reference_from_local(self, sample: dict) -> str:
        """Get reference from local manifest."""
        return sample.get("text", "")
    
    def postprocess_prediction(self, pred_text: str, sample: EvalSample) -> str:
        """Normalize prediction for WER computation."""
        return normalize_text(pred_text, lowercase=True, remove_punctuation=True)


@register_dataset("librispeech_other")
class LibriSpeechOtherDataset(BaseDataset):
    """
    LibriSpeech test-other dataset for ASR evaluation.
    
    Contains ~2,939 samples of more challenging read English speech.
    
    Source: https://huggingface.co/datasets/openslr/librispeech_asr
    """
    
    name = "librispeech_other"
    task_type = "asr"
    default_split = "test.other"
    metrics = ["wer", "cer"]
    language = "en"
    description = "LibriSpeech test-other (more challenging read English speech)"
    
    prompt_template = "Transcribe the following audio exactly as spoken."
    
    def load_from_hf(self) -> Iterator[EvalSample]:
        """Load LibriSpeech test-other from HuggingFace."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            "openslr/librispeech_asr",
            "other",
            split="test",
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"librispeech_other_{idx}")
            
            yield EvalSample(
                id=f"librispeech_other_{idx}",
                audio_path=audio_path,
                text_prompt=self.build_prompt(sample),
                reference=self.get_reference(sample),
                meta={
                    "speaker_id": sample.get("speaker_id"),
                    "chapter_id": sample.get("chapter_id"),
                    "audio_array": audio["array"],
                    "sampling_rate": audio["sampling_rate"],
                },
            )
            count += 1
    
    def get_reference(self, sample: Any) -> str:
        return sample["text"]
    
    def get_reference_from_local(self, sample: dict) -> str:
        return sample.get("text", "")
    
    def postprocess_prediction(self, pred_text: str, sample: EvalSample) -> str:
        return normalize_text(pred_text, lowercase=True, remove_punctuation=True)
