"""
Common Voice ASR dataset.
"""

from typing import Iterator, Any
from pathlib import Path

from ..base import BaseDataset
from ..common_types import normalize_text
from ...schema import EvalSample
from ...registry import register_dataset


@register_dataset("common_voice_en")
class CommonVoiceEnglishDataset(BaseDataset):
    """
    Common Voice English test set for ASR evaluation.
    
    Source: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
    Note: Requires authentication and agreement to terms.
    """
    
    name = "common_voice_en"
    task_type = "asr"
    default_split = "test"
    metrics = ["wer", "cer"]
    language = "en"
    description = "Common Voice English test set"
    
    prompt_template = "Transcribe the following English speech."
    
    def __init__(self, version: str = "17.0", **kwargs):
        super().__init__(**kwargs)
        self.version = version
    
    def load_from_hf(self) -> Iterator[EvalSample]:
        """Load Common Voice English test set."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            f"mozilla-foundation/common_voice_{self.version.replace('.', '_')}",
            "en",
            split=self.split,
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"common_voice_en_{idx}")
            
            yield EvalSample(
                id=f"common_voice_en_{idx}",
                audio_path=audio_path,
                text_prompt=self.build_prompt(sample),
                reference=self.get_reference(sample),
                meta={
                    "age": sample.get("age"),
                    "gender": sample.get("gender"),
                    "accent": sample.get("accent"),
                    "audio_array": audio["array"],
                    "sampling_rate": audio["sampling_rate"],
                },
            )
            count += 1
    
    def get_reference(self, sample: Any) -> str:
        return sample["sentence"]
    
    def get_reference_from_local(self, sample: dict) -> str:
        return sample.get("text", sample.get("sentence", ""))
    
    def postprocess_prediction(self, pred_text: str, sample: EvalSample) -> str:
        return normalize_text(pred_text, lowercase=True, remove_punctuation=True)


@register_dataset("common_voice_zh")
class CommonVoiceChineseDataset(BaseDataset):
    """
    Common Voice Chinese (zh-CN) test set for ASR evaluation.
    """
    
    name = "common_voice_zh"
    task_type = "asr"
    default_split = "test"
    metrics = ["cer"]  # CER is more appropriate for Chinese
    language = "zh"
    description = "Common Voice Chinese test set"
    
    prompt_template = "请转录以下中文语音。"
    
    def __init__(self, version: str = "17.0", **kwargs):
        super().__init__(**kwargs)
        self.version = version
    
    def load_from_hf(self) -> Iterator[EvalSample]:
        """Load Common Voice Chinese test set."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            f"mozilla-foundation/common_voice_{self.version.replace('.', '_')}",
            "zh-CN",
            split=self.split,
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"common_voice_zh_{idx}")
            
            yield EvalSample(
                id=f"common_voice_zh_{idx}",
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
        return sample["sentence"]
    
    def get_reference_from_local(self, sample: dict) -> str:
        return sample.get("text", sample.get("sentence", ""))
    
    def postprocess_prediction(self, pred_text: str, sample: EvalSample) -> str:
        from ..common_types import normalize_chinese
        text = normalize_text(pred_text, lowercase=False, remove_punctuation=True)
        return normalize_chinese(text)
