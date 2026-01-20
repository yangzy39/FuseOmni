"""
AIR-Bench Audio Understanding dataset.
"""

from typing import Iterator, Any
from pathlib import Path

from ..base import BaseDataset
from ...schema import EvalSample
from ...registry import register_dataset


@register_dataset("airbench")
class AIRBenchDataset(BaseDataset):
    """
    AIR-Bench: Audio Instruction Recognition Benchmark.
    
    Benchmarking large audio-language models across multiple domains.
    
    Source: https://huggingface.co/datasets/AIR-Bench/air_bench
    """
    
    name = "airbench"
    task_type = "qa"
    default_split = "test"
    metrics = ["accuracy"]
    language = "en"
    description = "AIR-Bench - Audio instruction recognition and understanding"
    
    prompt_template = """Listen to the audio and respond to the following instruction.

{instruction}"""
    
    def load(self) -> Iterator[EvalSample]:
        """Load AIR-Bench dataset."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library is required. Install with: pip install datasets")
        
        dataset = load_dataset(
            "AIR-Bench/air_bench",
            split=self.split,
            trust_remote_code=True,
        )
        
        count = 0
        for idx, sample in enumerate(dataset):
            if self.limit and count >= self.limit:
                break
            
            audio = sample["audio"]
            audio_path = audio.get("path", f"airbench_{idx}")
            
            prompt = self.prompt_template.format(
                instruction=sample.get("instruction", sample.get("question", ""))
            )
            
            yield EvalSample(
                id=f"airbench_{idx}",
                audio_path=audio_path,
                text_prompt=prompt,
                reference=self.get_reference(sample),
                meta={
                    "domain": sample.get("domain"),
                    "audio_array": audio["array"],
                    "sampling_rate": audio["sampling_rate"],
                },
            )
            count += 1
    
    def get_reference(self, sample: Any) -> str:
        return sample.get("answer", sample.get("response", ""))
    
    def postprocess_prediction(self, pred_text: str, sample: EvalSample) -> str:
        return pred_text.strip()
