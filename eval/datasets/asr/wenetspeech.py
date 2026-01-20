"""
WenetSpeech Chinese ASR dataset.
"""

from typing import Iterator, Any
from pathlib import Path

from ..base import BaseDataset
from ..common_types import normalize_text, normalize_chinese
from ...schema import EvalSample
from ...registry import register_dataset


@register_dataset("wenetspeech")
class WenetSpeechDataset(BaseDataset):
    """
    WenetSpeech Mandarin Chinese ASR dataset.
    
    10000+ hours of multi-domain Mandarin speech.
    
    Source: https://huggingface.co/datasets/wenet/wenetspeech
    """
    
    name = "wenetspeech"
    task_type = "asr"
    default_split = "test_net"
    metrics = ["cer"]
    language = "zh"
    description = "WenetSpeech Mandarin Chinese ASR"
    
    prompt_template = "请将以下中文语音转录为文字。"
    
    def load_from_hf(self) -> Iterator[EvalSample]:
        """Load WenetSpeech test set."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            "wenet/wenetspeech",
            split=self.split,
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"wenetspeech_{idx}")
            
            yield EvalSample(
                id=f"wenetspeech_{idx}",
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
        text = normalize_text(pred_text, lowercase=False, remove_punctuation=True)
        return normalize_chinese(text)
