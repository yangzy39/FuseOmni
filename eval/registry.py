"""
Registry system for datasets and metrics.

Provides decorators and functions for registering and retrieving
datasets and metrics by name.
"""

from typing import Type, TypeVar, Callable, Any

T = TypeVar("T")

# Global registries
DATASET_REGISTRY: dict[str, Type] = {}
METRIC_REGISTRY: dict[str, Type] = {}


def register_dataset(name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a dataset class.
    
    Usage:
        @register_dataset("librispeech")
        class LibriSpeechDataset(BaseDataset):
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        if name in DATASET_REGISTRY:
            raise ValueError(f"Dataset '{name}' is already registered")
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset(name: str) -> Type:
    """
    Get a dataset class by name.
    
    Args:
        name: The registered name of the dataset
        
    Returns:
        The dataset class
        
    Raises:
        KeyError: If the dataset is not registered
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise KeyError(f"Dataset '{name}' not found. Available: {available}")
    return DATASET_REGISTRY[name]


def list_datasets() -> dict[str, Type]:
    """
    List all registered datasets.
    
    Returns:
        Dictionary mapping dataset names to their classes
    """
    return dict(DATASET_REGISTRY)


def register_metric(name: str) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a metric class.
    
    Usage:
        @register_metric("wer")
        class WERMetric(BaseMetric):
            ...
    """
    def decorator(cls: Type[T]) -> Type[T]:
        if name in METRIC_REGISTRY:
            raise ValueError(f"Metric '{name}' is already registered")
        METRIC_REGISTRY[name] = cls
        return cls
    return decorator


def get_metric(name: str) -> Type:
    """
    Get a metric class by name.
    
    Args:
        name: The registered name of the metric
        
    Returns:
        The metric class
        
    Raises:
        KeyError: If the metric is not registered
    """
    if name not in METRIC_REGISTRY:
        available = ", ".join(sorted(METRIC_REGISTRY.keys()))
        raise KeyError(f"Metric '{name}' not found. Available: {available}")
    return METRIC_REGISTRY[name]


def list_metrics() -> dict[str, Type]:
    """
    List all registered metrics.
    
    Returns:
        Dictionary mapping metric names to their classes
    """
    return dict(METRIC_REGISTRY)
