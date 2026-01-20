"""
Spoken-SQuAD QA dataset.
"""

from typing import Iterator, Any
from pathlib import Path

from ..base import BaseDataset
from ...schema import EvalSample
from ...registry import register_dataset


@register_dataset("spoken_squad")
class SpokenSQuADDataset(BaseDataset):
    """
    Spoken-SQuAD: Spoken Question Answering dataset.
    
    Audio version of the SQuAD reading comprehension dataset.
    
    Source: https://huggingface.co/datasets/speech_squad
    """
    
    name = "spoken_squad"
    task_type = "qa"
    default_split = "test"
    metrics = ["qa_f1", "qa_em"]
    language = "en"
    description = "Spoken-SQuAD - Spoken question answering"
    
    prompt_template = """Listen to the audio which contains a question about a passage.
Answer the question based on the spoken content.

Provide a concise answer."""
    
    def load(self) -> Iterator[EvalSample]:
        """Load Spoken-SQuAD dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            "speech_squad",
            split=self.split,
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"spoken_squad_{idx}")
            
            yield EvalSample(
                id=f"spoken_squad_{idx}",
                audio_path=audio_path,
                text_prompt=self.build_prompt(sample),
                reference=self.get_reference(sample),
                meta={
                    "question": sample.get("question"),
                    "context": sample.get("context"),
                    "audio_array": audio["array"],
                    "sampling_rate": audio["sampling_rate"],
                },
            )
            count += 1
    
    def get_reference(self, sample: Any) -> Any:
        # SQuAD can have multiple answers
        answers = sample.get("answers", {})
        if isinstance(answers, dict):
            return answers.get("text", [])
        return answers
    
    def postprocess_prediction(self, pred_text: str, sample: EvalSample) -> str:
        return pred_text.strip()
