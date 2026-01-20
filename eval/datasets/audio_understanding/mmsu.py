"""
MMSU Speech Understanding dataset.
"""

from typing import Iterator, Any
from pathlib import Path

from ..base import BaseDataset
from ..common_types import parse_mcq_answer
from ...schema import EvalSample
from ...registry import register_dataset


@register_dataset("mmsu")
class MMSUDataset(BaseDataset):
    """
    MMSU (Massive Multi-task Speech Understanding) benchmark.
    
    Speech specialist benchmark focusing on linguistic nuances:
    intonation, emotion, prosody, speaking style.
    
    Source: https://huggingface.co/datasets/MMSU/mmsu
    """
    
    name = "mmsu"
    task_type = "mcq"
    default_split = "test"
    metrics = ["accuracy"]
    language = "en"
    description = "MMSU - Speech understanding (intonation, emotion, prosody)"
    
    prompt_template = """Analyze the speech in the audio and answer the question.

Question: {question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Choose the best answer (A, B, C, or D)."""
    
    def load_from_hf(self) -> Iterator[EvalSample]:
        """Load MMSU dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            "MMSU/mmsu",
            split=self.split,
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"mmsu_{idx}")
            
            prompt = self.prompt_template.format(
                question=sample["question"],
                option_a=sample["option_a"],
                option_b=sample["option_b"],
                option_c=sample["option_c"],
                option_d=sample["option_d"],
            )
            
            yield EvalSample(
                id=f"mmsu_{idx}",
                audio_path=audio_path,
                text_prompt=prompt,
                reference=self.get_reference(sample),
                meta={
                    "task_type": sample.get("task_type"),
                    "audio_array": audio["array"],
                    "sampling_rate": audio["sampling_rate"],
                },
            )
            count += 1
    
    def get_reference(self, sample: Any) -> str:
        return sample["answer"]
    
    def build_prompt_from_local(self, sample: dict) -> str:
        return self.prompt_template.format(
            question=sample.get("text", sample.get("question", "")),
            option_a=sample.get("option_a", ""),
            option_b=sample.get("option_b", ""),
            option_c=sample.get("option_c", ""),
            option_d=sample.get("option_d", ""),
        )
    
    def get_reference_from_local(self, sample: dict) -> str:
        return sample.get("answer", "")
    
    def postprocess_prediction(self, pred_text: str, sample: EvalSample) -> str:
        answer = parse_mcq_answer(pred_text)
        return answer if answer else pred_text.strip()[:1].upper()
