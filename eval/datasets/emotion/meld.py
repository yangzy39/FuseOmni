"""
MELD Emotion Recognition dataset.
"""

from typing import Iterator, Any
from pathlib import Path

from ..base import BaseDataset
from ...schema import EvalSample
from ...registry import register_dataset


MELD_EMOTIONS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
MELD_SENTIMENTS = ["positive", "negative", "neutral"]


@register_dataset("meld_emotion")
class MELDEmotionDataset(BaseDataset):
    """
    MELD Emotion Recognition dataset.
    
    Multimodal EmotionLines Dataset from Friends TV series.
    
    Source: https://huggingface.co/datasets/declare-lab/MELD
    """
    
    name = "meld_emotion"
    task_type = "emotion"
    default_split = "test"
    metrics = ["accuracy", "weighted_f1"]
    language = "en"
    description = "MELD - Multimodal emotion recognition (Friends TV)"
    
    prompt_template = """Listen to the audio and identify the emotion expressed.

Choose from: anger, disgust, fear, joy, neutral, sadness, surprise

Answer with only the emotion label."""
    
    def load(self) -> Iterator[EvalSample]:
        """Load MELD emotion dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            "declare-lab/MELD",
            split=self.split,
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            # MELD may have audio or video
            if "audio" not in sample:
                continue
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"meld_emotion_{idx}")
            
            yield EvalSample(
                id=f"meld_emotion_{idx}",
                audio_path=audio_path,
                text_prompt=self.build_prompt(sample),
                reference=self.get_reference(sample),
                meta={
                    "utterance": sample.get("Utterance"),
                    "speaker": sample.get("Speaker"),
                    "audio_array": audio["array"],
                    "sampling_rate": audio["sampling_rate"],
                },
            )
            count += 1
    
    def get_reference(self, sample: Any) -> str:
        return sample.get("Emotion", "").lower()
    
    def postprocess_prediction(self, pred_text: str, sample: EvalSample) -> str:
        pred = pred_text.lower().strip()
        for label in MELD_EMOTIONS:
            if label in pred:
                return label
        return pred


@register_dataset("meld_sentiment")
class MELDSentimentDataset(BaseDataset):
    """
    MELD Sentiment Classification dataset.
    """
    
    name = "meld_sentiment"
    task_type = "emotion"
    default_split = "test"
    metrics = ["accuracy", "macro_f1"]
    language = "en"
    description = "MELD - Sentiment classification (positive/negative/neutral)"
    
    prompt_template = """Listen to the audio and identify the sentiment.

Choose from: positive, negative, neutral

Answer with only the sentiment label."""
    
    def load(self) -> Iterator[EvalSample]:
        """Load MELD sentiment dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            "declare-lab/MELD",
            split=self.split,
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            if "audio" not in sample:
                continue
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"meld_sentiment_{idx}")
            
            yield EvalSample(
                id=f"meld_sentiment_{idx}",
                audio_path=audio_path,
                text_prompt=self.build_prompt(sample),
                reference=self.get_reference(sample),
                meta={
                    "utterance": sample.get("Utterance"),
                    "audio_array": audio["array"],
                    "sampling_rate": audio["sampling_rate"],
                },
            )
            count += 1
    
    def get_reference(self, sample: Any) -> str:
        return sample.get("Sentiment", "").lower()
    
    def postprocess_prediction(self, pred_text: str, sample: EvalSample) -> str:
        pred = pred_text.lower().strip()
        for label in MELD_SENTIMENTS:
            if label in pred:
                return label
        return pred
