"""
Core schema definitions for the evaluation framework.

Defines the canonical data structures used across all datasets, metrics, and engines.
"""

from dataclasses import dataclass, field
from typing import Any, Optional, Literal
from pathlib import Path
import json


@dataclass
class EvalSample:
    """
    Represents a single evaluation sample.
    
    Attributes:
        id: Unique identifier for the sample
        audio_path: Path to the audio file (or bytes if loaded)
        text_prompt: The text prompt to send to the model (without audio placeholder)
        reference: Ground truth reference (transcript, answer, label, etc.)
        meta: Additional metadata (e.g., language, speaker, duration)
    """
    id: str
    audio_path: str | Path
    text_prompt: str
    reference: Any
    meta: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "audio_path": str(self.audio_path),
            "text_prompt": self.text_prompt,
            "reference": self.reference,
            "meta": self.meta,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EvalSample":
        return cls(
            id=data["id"],
            audio_path=data["audio_path"],
            text_prompt=data["text_prompt"],
            reference=data["reference"],
            meta=data.get("meta", {}),
        )


@dataclass
class EvalPrediction:
    """
    Represents a model prediction for an evaluation sample.
    
    Attributes:
        sample_id: ID of the corresponding EvalSample
        text: The generated text output
        raw: Raw output from the engine (for debugging)
        meta: Additional metadata (e.g., generation time, tokens)
    """
    sample_id: str
    text: str
    raw: Any = None
    meta: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "text": self.text,
            "raw": self.raw if isinstance(self.raw, (dict, list, str, int, float, bool, type(None))) else str(self.raw),
            "meta": self.meta,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EvalPrediction":
        return cls(
            sample_id=data["sample_id"],
            text=data["text"],
            raw=data.get("raw"),
            meta=data.get("meta", {}),
        )


@dataclass
class SamplingConfig:
    """
    Sampling parameters for generation.
    """
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 512
    repetition_penalty: float = 1.0
    stop: list[str] = field(default_factory=list)
    seed: Optional[int] = 42
    
    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "stop": self.stop,
            "seed": self.seed,
        }


@dataclass
class EngineConfig:
    """
    Configuration for the vLLM-Omni inference engine.
    
    See: https://github.com/vllm-project/vllm-omni
    """
    model_path: str
    
    # vLLM-Omni specific
    stage_configs_path: Optional[str] = None  # Path to stage configs YAML
    log_stats: bool = False  # Enable detailed statistics logging
    stage_init_timeout: int = 300  # Timeout for stage initialization (seconds)
    system_prompt: Optional[str] = None  # Custom system prompt
    
    # Legacy vLLM params (for compatibility)
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True
    limit_mm_per_prompt: dict = field(default_factory=lambda: {"audio": 1})
    
    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "stage_configs_path": self.stage_configs_path,
            "log_stats": self.log_stats,
            "stage_init_timeout": self.stage_init_timeout,
            "system_prompt": self.system_prompt,
            "tensor_parallel_size": self.tensor_parallel_size,
            "dtype": self.dtype,
            "max_model_len": self.max_model_len,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
            "limit_mm_per_prompt": self.limit_mm_per_prompt,
        }


@dataclass
class RunConfig:
    """
    Configuration for an evaluation run.
    """
    dataset: str
    model_path: str
    output_dir: str = "outputs"
    split: str = "test"
    limit: Optional[int] = None
    batch_size: int = 1
    num_workers: int = 4
    resume: bool = False
    verbose: bool = False
    
    # Sampling params
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 512
    repetition_penalty: float = 1.0
    seed: int = 42
    
    # Engine params
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    max_model_len: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    trust_remote_code: bool = True
    
    def to_sampling_config(self) -> SamplingConfig:
        return SamplingConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed,
        )
    
    def to_engine_config(self) -> EngineConfig:
        return EngineConfig(
            model_path=self.model_path,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype=self.dtype,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=self.trust_remote_code,
        )
    
    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "split": self.split,
            "limit": self.limit,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "resume": self.resume,
            "verbose": self.verbose,
            "sampling": self.to_sampling_config().to_dict(),
            "engine": self.to_engine_config().to_dict(),
        }


@dataclass
class EvalResult:
    """
    Aggregated evaluation result.
    """
    dataset: str
    model_path: str
    metrics: dict[str, float]
    num_samples: int
    config: dict
    predictions_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "model_path": self.model_path,
            "metrics": self.metrics,
            "num_samples": self.num_samples,
            "config": self.config,
            "predictions_path": self.predictions_path,
        }
    
    def save(self, path: str | Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
