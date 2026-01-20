"""
Accuracy metric for classification and MCQ tasks.
"""

from typing import Any
from .base import BaseMetric
from ..registry import register_metric


@register_metric("accuracy")
class AccuracyMetric(BaseMetric):
    """
    Simple accuracy metric for classification tasks.
    
    Computes the percentage of correct predictions.
    """
    
    name = "accuracy"
    higher_is_better = True
    description = "Classification accuracy"
    
    def __init__(self, case_sensitive: bool = False):
        """
        Args:
            case_sensitive: Whether comparison should be case-sensitive
        """
        self.case_sensitive = case_sensitive
    
    def compute(
        self,
        references: list[Any],
        hypotheses: list[Any],
        metas: list[dict] | None = None,
    ) -> dict[str, float]:
        """
        Compute accuracy.
        
        Returns:
            Dictionary with 'accuracy' as percentage (0-100)
        """
        if len(references) == 0:
            return {"accuracy": 0.0}
        
        correct = 0
        for ref, hyp in zip(references, hypotheses):
            ref_str = str(ref).strip()
            hyp_str = str(hyp).strip()
            
            if not self.case_sensitive:
                ref_str = ref_str.lower()
                hyp_str = hyp_str.lower()
            
            if ref_str == hyp_str:
                correct += 1
        
        accuracy = (correct / len(references)) * 100
        return {"accuracy": accuracy}


@register_metric("exact_match")
class ExactMatchMetric(BaseMetric):
    """
    Exact match metric (same as accuracy with string normalization).
    """
    
    name = "exact_match"
    higher_is_better = True
    description = "Exact string match"
    
    def compute(
        self,
        references: list[Any],
        hypotheses: list[Any],
        metas: list[dict] | None = None,
    ) -> dict[str, float]:
        """
        Compute exact match rate.
        """
        if len(references) == 0:
            return {"exact_match": 0.0}
        
        correct = 0
        for ref, hyp in zip(references, hypotheses):
            # Normalize both strings
            ref_norm = " ".join(str(ref).lower().split())
            hyp_norm = " ".join(str(hyp).lower().split())
            
            if ref_norm == hyp_norm:
                correct += 1
        
        return {"exact_match": (correct / len(references)) * 100}
