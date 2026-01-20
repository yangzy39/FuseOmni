"""
FLEURS Speech Translation dataset.
"""

from typing import Iterator, Any
from pathlib import Path

from ..base import BaseDataset
from ...schema import EvalSample
from ...registry import register_dataset


@register_dataset("fleurs_en_zh")
class FLEURSEnZhDataset(BaseDataset):
    """
    FLEURS English to Chinese speech translation.
    
    Source: https://huggingface.co/datasets/google/fleurs
    """
    
    name = "fleurs_en_zh"
    task_type = "translation"
    default_split = "test"
    metrics = ["bleu", "chrf"]
    language = "en-zh"
    description = "FLEURS English to Chinese speech translation"
    
    prompt_template = "Translate the following English speech to Chinese."
    
    def load(self) -> Iterator[EvalSample]:
        """Load FLEURS en-zh dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        # Load English audio
        en_dataset = load_dataset(
            "google/fleurs",
            "en_us",
            split=self.split,
            trust_remote_code=True,
        )
        
        # Load Chinese translations
        zh_dataset = load_dataset(
            "google/fleurs",
            "cmn_hans_cn",
            split=self.split,
            trust_remote_code=True,
        )
        
        # Create id to translation mapping
        zh_translations = {s["id"]: s["transcription"] for s in zh_dataset}
        
        count = 0
        for idx, sample in enumerate(en_dataset):
            if self.limit and count >= self.limit:
                break
            
            sample_id = sample["id"]
            if sample_id not in zh_translations:
                continue
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"fleurs_en_zh_{idx}")
            
            yield EvalSample(
                id=f"fleurs_en_zh_{idx}",
                audio_path=audio_path,
                text_prompt=self.build_prompt(sample),
                reference=zh_translations[sample_id],
                meta={
                    "source_text": sample.get("transcription"),
                    "audio_array": audio["array"],
                    "sampling_rate": audio["sampling_rate"],
                },
            )
            count += 1
    
    def get_reference(self, sample: Any) -> str:
        return sample.get("translation", "")
    
    def postprocess_prediction(self, pred_text: str, sample: EvalSample) -> str:
        return pred_text.strip()
