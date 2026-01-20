"""
Base class for metrics.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseMetric(ABC):
    """
    Abstract base class for all evaluation metrics.
    
    Subclasses must implement:
        - compute(): Calculate the metric from references and hypotheses
    
    Attributes:
        name: Unique identifier for the metric
        higher_is_better: Whether higher values are better
        description: Human-readable description
    """
    
    name: str = "base"
    higher_is_better: bool = True
    description: str = "Base metric class"
    
    @abstractmethod
    def compute(
        self,
        references: list[Any],
        hypotheses: list[Any],
        metas: list[dict] | None = None,
    ) -> dict[str, float]:
        """
        Compute the metric from references and hypotheses.
        
        Args:
            references: List of ground truth values
            hypotheses: List of predicted values
            metas: Optional list of metadata dicts for each sample
            
        Returns:
            Dictionary with metric name(s) and value(s)
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
