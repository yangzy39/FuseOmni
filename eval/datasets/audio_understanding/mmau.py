"""
MMAU Audio Understanding dataset.
"""

from typing import Iterator, Any
from pathlib import Path

from ..base import BaseDataset
from ..common_types import parse_mcq_answer
from ...schema import EvalSample
from ...registry import register_dataset


@register_dataset("mmau")
class MMAUDataset(BaseDataset):
    """
    MMAU (Massive Multi-task Audio Understanding) benchmark.
    
    Comprehensive audio understanding benchmark covering speech, music, and sound.
    Uses multiple-choice questions.
    
    Source: https://huggingface.co/datasets/MMAU/mmau_mini
    """
    
    name = "mmau"
    task_type = "mcq"
    default_split = "test"
    metrics = ["accuracy"]
    language = "en"
    description = "MMAU - Massive Multi-task Audio Understanding (speech, music, sound)"
    
    prompt_template = """Listen to the audio and answer the following question.

Question: {question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Answer with only the letter (A, B, C, or D)."""
    
    def load_from_hf(self) -> Iterator[EvalSample]:
        """Load MMAU dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            "MMAU/mmau_mini",
            split=self.split,
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"mmau_{idx}")
            
            # Build prompt with options
            prompt = self.prompt_template.format(
                question=sample["question"],
                option_a=sample["option_a"],
                option_b=sample["option_b"],
                option_c=sample["option_c"],
                option_d=sample["option_d"],
            )
            
            yield EvalSample(
                id=f"mmau_{idx}",
                audio_path=audio_path,
                text_prompt=prompt,
                reference=self.get_reference(sample),
                meta={
                    "category": sample.get("category"),
                    "subcategory": sample.get("subcategory"),
                    "question": sample["question"],
                    "audio_array": audio["array"],
                    "sampling_rate": audio["sampling_rate"],
                },
            )
            count += 1
    
    def get_reference(self, sample: Any) -> str:
        """Get the correct answer letter."""
        return sample["answer"]
    
    def build_prompt_from_local(self, sample: dict) -> str:
        """Build prompt from local manifest."""
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
        """Extract answer letter from prediction."""
        answer = parse_mcq_answer(pred_text)
        return answer if answer else pred_text.strip()[:1].upper()


@register_dataset("mmau_pro")
class MMAUProDataset(BaseDataset):
    """
    MMAU-Pro advanced audio understanding benchmark.
    
    Advanced scenarios with long-form, spatial, and overlapping audio.
    
    Source: https://huggingface.co/datasets/MMAU/mmau_pro
    """
    
    name = "mmau_pro"
    task_type = "mcq"
    default_split = "test"
    metrics = ["accuracy"]
    language = "en"
    description = "MMAU-Pro - Advanced audio understanding scenarios"
    
    prompt_template = """Listen carefully to the audio and answer the question.

Question: {question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Respond with only the letter of the correct answer."""
    
    def load_from_hf(self) -> Iterator[EvalSample]:
        """Load MMAU-Pro dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            "MMAU/mmau_pro",
            split=self.split,
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"mmau_pro_{idx}")
            
            prompt = self.prompt_template.format(
                question=sample["question"],
                option_a=sample["option_a"],
                option_b=sample["option_b"],
                option_c=sample["option_c"],
                option_d=sample["option_d"],
            )
            
            yield EvalSample(
                id=f"mmau_pro_{idx}",
                audio_path=audio_path,
                text_prompt=prompt,
                reference=self.get_reference(sample),
                meta={
                    "scenario": sample.get("scenario"),
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
