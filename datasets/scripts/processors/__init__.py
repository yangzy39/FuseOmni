#!/usr/bin/env python3
"""
Dataset processor registry.

Provides a registry of all available processors and factory functions
for creating processor instances.
"""

from pathlib import Path
from typing import Dict, Type, Optional

from .base import BaseProcessor, ProcessorConfig

# Import all processors
from .librispeech import LibriSpeechProcessor
from .common_voice import CommonVoiceProcessor
from .gigaspeech import GigaSpeechProcessor
from .wavcaps import WavCapsProcessor
from .aishell import AISHELL1Processor


# Registry mapping dataset names to processor classes
PROCESSOR_REGISTRY: Dict[str, Type[BaseProcessor]] = {
    # Audio-only (ASR)
    "librispeech": LibriSpeechProcessor,
    "common_voice": CommonVoiceProcessor,
    "gigaspeech": GigaSpeechProcessor,
    "aishell1": AISHELL1Processor,
    
    # Audio captioning
    "wavcaps": WavCapsProcessor,
}


# Dataset configurations with HuggingFace paths and default settings
DATASET_CONFIGS: Dict[str, Dict] = {
    "librispeech": {
        "hf_id": "openslr/librispeech_asr",
        "subset": "clean",
        "split": "train.clean.100",
        "modality": "audio",
        "task": "asr",
        "language": "en",
        "description": "English audiobook ASR dataset",
    },
    "common_voice": {
        "hf_id": "mozilla-foundation/common_voice_17_0",
        "subset": "en",
        "split": "train",
        "modality": "audio",
        "task": "asr",
        "language": "en",
        "requires_auth": True,
        "description": "Multilingual crowdsourced ASR",
    },
    "common_voice_zh": {
        "hf_id": "mozilla-foundation/common_voice_17_0",
        "subset": "zh-CN",
        "split": "train",
        "modality": "audio",
        "task": "asr",
        "language": "zh",
        "requires_auth": True,
        "description": "Chinese Common Voice ASR",
        "processor": "common_voice",
    },
    "gigaspeech": {
        "hf_id": "speechcolab/gigaspeech",
        "subset": "xs",
        "split": "train",
        "modality": "audio",
        "task": "asr",
        "language": "en",
        "requires_auth": True,
        "description": "Large-scale English ASR",
    },
    "aishell1": {
        "hf_id": "AISHELL/AISHELL-1",
        "subset": None,
        "split": "train",
        "modality": "audio",
        "task": "asr",
        "language": "zh",
        "description": "Chinese Mandarin ASR dataset",
    },
    "wavcaps": {
        "hf_id": "cvssp/WavCaps",
        "subset": None,
        "split": "train",
        "modality": "audio",
        "task": "audio_captioning",
        "language": "en",
        "description": "Audio captioning dataset",
    },
}


def get_processor_class(name: str) -> Optional[Type[BaseProcessor]]:
    """Get processor class by dataset name."""
    # Check if config specifies a different processor
    if name in DATASET_CONFIGS:
        processor_name = DATASET_CONFIGS[name].get("processor", name)
        return PROCESSOR_REGISTRY.get(processor_name)
    return PROCESSOR_REGISTRY.get(name)


def create_processor(
    name: str,
    data_dir: Path,
    output_dir: Path,
    max_samples: int = -1,
    task_type: str = "sft",
    system_prompt: Optional[str] = None,
    **kwargs,
) -> BaseProcessor:
    """
    Factory function to create a processor instance.
    
    Args:
        name: Dataset name (must be in registry)
        data_dir: Directory containing downloaded dataset
        output_dir: Output directory for processed data
        max_samples: Maximum samples to process (-1 for all)
        task_type: "sft" or "grpo"
        system_prompt: Optional system prompt
        **kwargs: Additional config overrides
        
    Returns:
        Configured processor instance
        
    Raises:
        ValueError: If dataset name is not in registry
    """
    processor_class = get_processor_class(name)
    if processor_class is None:
        available = list(PROCESSOR_REGISTRY.keys())
        raise ValueError(f"Unknown dataset: {name}. Available: {available}")
    
    # Get default config
    default_config = DATASET_CONFIGS.get(name, {})
    
    # Build processor config
    config = ProcessorConfig(
        name=name,
        hf_id=kwargs.get("hf_id", default_config.get("hf_id", "")),
        max_samples=max_samples,
        split=kwargs.get("split", default_config.get("split", "train")),
        subset=kwargs.get("subset", default_config.get("subset")),
        output_dir=output_dir,
        task_type=task_type,
        system_prompt=system_prompt,
    )
    
    return processor_class(data_dir, config)


def list_available_datasets() -> Dict[str, Dict]:
    """List all available datasets with their configurations."""
    return DATASET_CONFIGS.copy()


def get_dataset_info(name: str) -> Optional[Dict]:
    """Get configuration info for a specific dataset."""
    return DATASET_CONFIGS.get(name)


__all__ = [
    "BaseProcessor",
    "ProcessorConfig",
    "PROCESSOR_REGISTRY",
    "DATASET_CONFIGS",
    "get_processor_class",
    "create_processor",
    "list_available_datasets",
    "get_dataset_info",
    # Individual processors
    "LibriSpeechProcessor",
    "CommonVoiceProcessor",
    "GigaSpeechProcessor",
    "WavCapsProcessor",
    "AISHELL1Processor",
]
