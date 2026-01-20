"""
IEMOCAP Emotion Recognition dataset.
"""

from typing import Iterator, Any
from pathlib import Path

from ..base import BaseDataset
from ...schema import EvalSample
from ...registry import register_dataset


EMOTION_LABELS = ["angry", "happy", "sad", "neutral", "frustrated", "excited", "fearful", "surprised", "disgusted"]


@register_dataset("iemocap")
class IEMOCAPDataset(BaseDataset):
    """
    IEMOCAP Emotion Recognition dataset.
    
    Interactive Emotional Dyadic Motion Capture database.
    
    Source: Requires manual download due to licensing.
    """
    
    name = "iemocap"
    task_type = "emotion"
    default_split = "test"
    metrics = ["accuracy", "macro_f1"]
    language = "en"
    description = "IEMOCAP - Interactive emotional speech recognition"
    
    prompt_template = """Listen to the audio and identify the emotion expressed.

Choose from: angry, happy, sad, neutral, frustrated, excited, fearful, surprised, disgusted

Answer with only the emotion label."""
    
    def load(self) -> Iterator[EvalSample]:
        """Load IEMOCAP dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        # Try to load from HuggingFace (if available)
        try:
            dataset = load_dataset(
                "Zahra99/IEMOCAP",
                split=self.split,
                trust_remote_code=True,
            )
        except Exception:
            # Fall back to local manifest if HF dataset not available
            if self.data_dir is None:
                raise ValueError(
                    "IEMOCAP requires either a HuggingFace dataset or local data_dir. "
                    "Due to licensing, you may need to download it manually."
                )
            return self._load_from_local()
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"iemocap_{idx}")
            
            yield EvalSample(
                id=f"iemocap_{idx}",
                audio_path=audio_path,
                text_prompt=self.build_prompt(sample),
                reference=self.get_reference(sample),
                meta={
                    "speaker": sample.get("speaker"),
                    "audio_array": audio["array"],
                    "sampling_rate": audio["sampling_rate"],
                },
            )
            count += 1
    
    def _load_from_local(self) -> Iterator[EvalSample]:
        """Load from local directory."""
        import json
        manifest_path = self.data_dir / f"{self.split}.json"
        
        with open(manifest_path) as f:
            data = json.load(f)
        
        for idx, sample in enumerate(data):
            if self.limit and idx >= self.limit:
                break
            
            yield EvalSample(
                id=f"iemocap_{idx}",
                audio_path=sample["audio_path"],
                text_prompt=self.prompt_template,
                reference=sample["emotion"],
                meta={"speaker": sample.get("speaker")},
            )
    
    def get_reference(self, sample: Any) -> str:
        return sample.get("emotion", sample.get("label", "")).lower()
    
    def postprocess_prediction(self, pred_text: str, sample: EvalSample) -> str:
        pred = pred_text.lower().strip()
        # Try to match to known labels
        for label in EMOTION_LABELS:
            if label in pred:
                return label
        return pred
