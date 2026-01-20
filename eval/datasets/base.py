"""
Base classes for datasets.

All dataset implementations should inherit from BaseDataset.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Any, Literal, Optional
from pathlib import Path
import os
import json
import logging

from ..schema import EvalSample

logger = logging.getLogger(__name__)

TaskType = Literal["asr", "mcq", "qa", "translation", "emotion", "instruction"]


class BaseDataset(ABC):
    """
    Abstract base class for all evaluation datasets.
    
    Supports two loading modes:
    1. HuggingFace (online): Load from HuggingFace datasets hub
    2. Local (offline): Load from downloaded manifest files
    
    Subclasses must implement:
        - load(): Yield EvalSample instances (for HuggingFace mode)
        - build_prompt(): Create the text prompt for a sample
        - get_reference(): Extract the reference from raw data
        - postprocess_prediction(): Clean/parse model output
    
    Attributes:
        name: Unique identifier for the dataset
        task_type: Type of task (asr, mcq, qa, translation, emotion, instruction)
        default_split: Default data split to use
        metrics: List of metric names to compute
        language: Primary language(s) of the dataset
        description: Human-readable description
    """
    
    name: str = "base"
    task_type: TaskType = "asr"
    default_split: str = "test"
    metrics: list[str] = ["wer"]
    language: str = "en"
    description: str = "Base dataset class"
    
    # Prompt template for this dataset
    prompt_template: str = "{instruction}"
    
    def __init__(
        self,
        split: str | None = None,
        data_dir: str | Path | None = None,
        limit: int | None = None,
        use_local: bool = False,
        **kwargs,
    ):
        """
        Initialize the dataset.
        
        Args:
            split: Data split to use (default: self.default_split)
            data_dir: Data directory for local loading (if use_local=True)
            limit: Maximum number of samples to load
            use_local: If True, load from local manifest files
            **kwargs: Additional dataset-specific arguments
        """
        self.split = split or self.default_split
        self.data_dir = Path(data_dir) if data_dir else None
        self.limit = limit
        self.use_local = use_local
        self.kwargs = kwargs
        
        # If data_dir is set and contains our dataset, use local loading
        if self.data_dir and not use_local:
            manifest_path = self.data_dir / self.name / "manifest.json"
            if manifest_path.exists():
                logger.info(f"Found local manifest for {self.name}, using local loading")
                self.use_local = True
    
    def load(self) -> Iterator[EvalSample]:
        """
        Load the dataset and yield EvalSample instances.
        
        Automatically chooses between local and HuggingFace loading.
        
        Yields:
            EvalSample for each data point
        """
        if self.use_local:
            yield from self.load_from_local()
        else:
            yield from self.load_from_hf()
    
    def load_from_local(self) -> Iterator[EvalSample]:
        """
        Load dataset from local manifest file.
        
        Expects manifest.json in data_dir/dataset_name/ with format:
        {
            "name": "dataset_name",
            "samples": [
                {"id": "...", "audio_path": "...", "text": "...", ...},
                ...
            ]
        }
        
        Yields:
            EvalSample for each data point
        """
        if self.data_dir is None:
            raise ValueError(f"data_dir must be set for local loading of {self.name}")
        
        manifest_path = self.data_dir / self.name / "manifest.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Run: python -m eval.download_datasets --output-dir {self.data_dir} --datasets {self.name}"
            )
        
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        
        samples = manifest.get("samples", [])
        logger.info(f"Loading {self.name} from local: {len(samples)} samples")
        
        count = 0
        for sample in samples:
            if self.limit and count >= self.limit:
                break
            
            # Build audio path relative to data_dir
            audio_path = self.data_dir / sample["audio_path"]
            
            # Build prompt from sample data
            prompt = self.build_prompt_from_local(sample)
            
            # Get reference
            reference = self.get_reference_from_local(sample)
            
            yield EvalSample(
                id=sample["id"],
                audio_path=str(audio_path),
                text_prompt=prompt,
                reference=reference,
                meta=self._extract_local_meta(sample),
            )
            count += 1
    
    def build_prompt_from_local(self, sample: dict) -> str:
        """
        Build prompt from local manifest sample.
        
        Override in subclasses for task-specific prompt building.
        
        Args:
            sample: Sample dict from manifest
            
        Returns:
            The formatted text prompt
        """
        return self.prompt_template
    
    def get_reference_from_local(self, sample: dict) -> Any:
        """
        Get reference from local manifest sample.
        
        Override in subclasses for task-specific reference extraction.
        
        Args:
            sample: Sample dict from manifest
            
        Returns:
            The reference value
        """
        # Default: look for common keys
        for key in ["text", "answer", "translation", "emotion", "label"]:
            if key in sample:
                return sample[key]
        return ""
    
    def _extract_local_meta(self, sample: dict) -> dict:
        """Extract metadata from local sample."""
        meta = {}
        # Common metadata keys
        meta_keys = [
            "speaker_id", "chapter_id", "age", "gender", "accent",
            "category", "subcategory", "task", "domain", "speaker",
            "source_text", "scenario", "task_type", "utterance",
        ]
        for key in meta_keys:
            if key in sample:
                meta[key] = sample[key]
        return meta
    
    @abstractmethod
    def load_from_hf(self) -> Iterator[EvalSample]:
        """
        Load the dataset from HuggingFace.
        
        Subclasses must implement this for online loading.
        
        Yields:
            EvalSample for each data point
        """
        raise NotImplementedError
    
    def build_prompt(self, sample: Any) -> str:
        """
        Build the text prompt for a HuggingFace sample.
        
        This is called during load_from_hf() to create the text_prompt field.
        Override in subclasses for custom prompt formatting.
        
        Args:
            sample: Raw data sample from the source dataset
            
        Returns:
            The formatted text prompt
        """
        return self.prompt_template
    
    @abstractmethod
    def get_reference(self, sample: Any) -> Any:
        """
        Extract the reference (ground truth) from a raw HuggingFace sample.
        
        Args:
            sample: Raw data sample
            
        Returns:
            The reference value (transcript, answer, label, etc.)
        """
        raise NotImplementedError
    
    def postprocess_prediction(self, pred_text: str, sample: EvalSample) -> Any:
        """
        Post-process the model's prediction.
        
        Override in subclasses for task-specific parsing (e.g., MCQ answer extraction).
        
        Args:
            pred_text: Raw model output text
            sample: The corresponding EvalSample
            
        Returns:
            Processed prediction
        """
        return pred_text.strip()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, split={self.split!r}, use_local={self.use_local})"
