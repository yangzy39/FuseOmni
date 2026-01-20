"""
CoVoST2 Speech Translation dataset.
"""

from typing import Iterator, Any
from pathlib import Path

from ..base import BaseDataset
from ...schema import EvalSample
from ...registry import register_dataset


@register_dataset("covost2_en_zh")
class CoVoST2EnZhDataset(BaseDataset):
    """
    CoVoST2 English to Chinese speech translation.
    
    Source: https://huggingface.co/datasets/facebook/covost2
    """
    
    name = "covost2_en_zh"
    task_type = "translation"
    default_split = "test"
    metrics = ["bleu", "chrf"]
    language = "en-zh"
    description = "CoVoST2 English to Chinese speech translation"
    
    prompt_template = "Translate the following English speech to Chinese."
    
    def load(self) -> Iterator[EvalSample]:
        """Load CoVoST2 en-zh dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            "facebook/covost2",
            "en_zh-CN",
            split=self.split,
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"covost2_en_zh_{idx}")
            
            yield EvalSample(
                id=f"covost2_en_zh_{idx}",
                audio_path=audio_path,
                text_prompt=self.build_prompt(sample),
                reference=self.get_reference(sample),
                meta={
                    "source_text": sample.get("sentence"),
                    "audio_array": audio["array"],
                    "sampling_rate": audio["sampling_rate"],
                },
            )
            count += 1
    
    def get_reference(self, sample: Any) -> str:
        return sample["translation"]
    
    def postprocess_prediction(self, pred_text: str, sample: EvalSample) -> str:
        return pred_text.strip()


@register_dataset("covost2_zh_en")
class CoVoST2ZhEnDataset(BaseDataset):
    """
    CoVoST2 Chinese to English speech translation.
    """
    
    name = "covost2_zh_en"
    task_type = "translation"
    default_split = "test"
    metrics = ["bleu", "chrf"]
    language = "zh-en"
    description = "CoVoST2 Chinese to English speech translation"
    
    prompt_template = "Translate the following Chinese speech to English."
    
    def load(self) -> Iterator[EvalSample]:
        """Load CoVoST2 zh-en dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            "facebook/covost2",
            "zh-CN_en",
            split=self.split,
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"covost2_zh_en_{idx}")
            
            yield EvalSample(
                id=f"covost2_zh_en_{idx}",
                audio_path=audio_path,
                text_prompt=self.build_prompt(sample),
                reference=self.get_reference(sample),
                meta={
                    "source_text": sample.get("sentence"),
                    "audio_array": audio["array"],
                    "sampling_rate": audio["sampling_rate"],
                },
            )
            count += 1
    
    def get_reference(self, sample: Any) -> str:
        return sample["translation"]
    
    def postprocess_prediction(self, pred_text: str, sample: EvalSample) -> str:
        return pred_text.strip()
