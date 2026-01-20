"""
VoiceBench Spoken QA dataset.
"""

from typing import Iterator, Any
from pathlib import Path

from ..base import BaseDataset
from ...schema import EvalSample
from ...registry import register_dataset


@register_dataset("voicebench")
class VoiceBenchDataset(BaseDataset):
    """
    VoiceBench: Comprehensive spoken QA benchmark.
    
    Covers instruction-following, general knowledge, safety, and robustness.
    
    Source: https://huggingface.co/datasets/voicebench/VoiceBench
    """
    
    name = "voicebench"
    task_type = "qa"
    default_split = "test"
    metrics = ["accuracy"]
    language = "en"
    description = "VoiceBench - Comprehensive spoken QA benchmark"
    
    prompt_template = """Listen to the audio and follow the instruction or answer the question.

Respond appropriately based on the audio content."""
    
    def load(self) -> Iterator[EvalSample]:
        """Load VoiceBench dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            "voicebench/VoiceBench",
            split=self.split,
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"voicebench_{idx}")
            
            yield EvalSample(
                id=f"voicebench_{idx}",
                audio_path=audio_path,
                text_prompt=self.build_prompt(sample),
                reference=self.get_reference(sample),
                meta={
                    "subset": sample.get("subset"),
                    "category": sample.get("category"),
                    "audio_array": audio["array"],
                    "sampling_rate": audio["sampling_rate"],
                },
            )
            count += 1
    
    def get_reference(self, sample: Any) -> str:
        return sample.get("answer", sample.get("response", ""))
    
    def postprocess_prediction(self, pred_text: str, sample: EvalSample) -> str:
        return pred_text.strip()


@register_dataset("openaudiobench")
class OpenAudioBenchDataset(BaseDataset):
    """
    OpenAudioBench: Open-source audio QA benchmark.
    
    Covers AlpacaEval, LlamaQ, ReasoningQA, TriviaQA, WebQ.
    
    Source: https://huggingface.co/datasets/baichuan-inc/OpenAudioBench
    """
    
    name = "openaudiobench"
    task_type = "qa"
    default_split = "test"
    metrics = ["accuracy"]
    language = "en"
    description = "OpenAudioBench - Spoken question answering benchmark"
    
    prompt_template = """Listen to the audio and answer the question or follow the instruction."""
    
    def load(self) -> Iterator[EvalSample]:
        """Load OpenAudioBench dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            "baichuan-inc/OpenAudioBench",
            split=self.split,
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"openaudiobench_{idx}")
            
            yield EvalSample(
                id=f"openaudiobench_{idx}",
                audio_path=audio_path,
                text_prompt=self.build_prompt(sample),
                reference=self.get_reference(sample),
                meta={
                    "task": sample.get("task"),
                    "audio_array": audio["array"],
                    "sampling_rate": audio["sampling_rate"],
                },
            )
            count += 1
    
    def get_reference(self, sample: Any) -> str:
        return sample.get("answer", "")
    
    def postprocess_prediction(self, pred_text: str, sample: EvalSample) -> str:
        return pred_text.strip()
