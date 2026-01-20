"""
GigaSpeech ASR dataset.
"""

from typing import Iterator, Any
from pathlib import Path

from ..base import BaseDataset
from ..common_types import normalize_text
from ...schema import EvalSample
from ...registry import register_dataset


@register_dataset("gigaspeech")
class GigaSpeechDataset(BaseDataset):
    """
    GigaSpeech English ASR dataset.
    
    Large-scale English ASR dataset with diverse audio sources.
    
    Source: https://huggingface.co/datasets/speechcolab/gigaspeech
    Note: Requires authentication.
    """
    
    name = "gigaspeech"
    task_type = "asr"
    default_split = "test"
    metrics = ["wer", "cer"]
    language = "en"
    description = "GigaSpeech English ASR (diverse sources)"
    
    prompt_template = "Transcribe the following English audio."
    
    def __init__(self, subset: str = "xs", **kwargs):
        """
        Args:
            subset: Dataset subset (xs, s, m, l, xl)
        """
        super().__init__(**kwargs)
        self.subset = subset
    
    def load_from_hf(self) -> Iterator[EvalSample]:
        """Load GigaSpeech test set."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            "speechcolab/gigaspeech",
            self.subset,
            split=self.split,
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"gigaspeech_{idx}")
            
            yield EvalSample(
                id=f"gigaspeech_{idx}",
                audio_path=audio_path,
                text_prompt=self.build_prompt(sample),
                reference=self.get_reference(sample),
                meta={
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
