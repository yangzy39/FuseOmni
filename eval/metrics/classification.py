"""
Classification metrics (F1, precision, recall).
"""

from typing import Any
from .base import BaseMetric
from ..registry import register_metric


@register_metric("macro_f1")
class MacroF1Metric(BaseMetric):
    """
    Macro-averaged F1 score for multi-class classification.
    """
    
    name = "macro_f1"
    higher_is_better = True
    description = "Macro-averaged F1 for classification"
    
    def compute(
        self,
        references: list[Any],
        hypotheses: list[Any],
        metas: list[dict] | None = None,
    ) -> dict[str, float]:
        """
        Compute macro F1 score.
        """
        try:
            from sklearn.metrics import f1_score
        except ImportError:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        refs = [str(r).lower().strip() for r in references]
        hyps = [str(h).lower().strip() for h in hypotheses]
        
        # Get unique labels
        labels = sorted(set(refs))
        
        # Compute F1
        f1 = f1_score(refs, hyps, labels=labels, average="macro", zero_division=0)
        
        return {"macro_f1": f1 * 100}


@register_metric("weighted_f1")
class WeightedF1Metric(BaseMetric):
    """
    Weighted F1 score for multi-class classification.
    """
    
    name = "weighted_f1"
    higher_is_better = True
    description = "Weighted F1 for classification"
    
    def compute(
        self,
        references: list[Any],
        hypotheses: list[Any],
        metas: list[dict] | None = None,
    ) -> dict[str, float]:
        """
        Compute weighted F1 score.
        """
        try:
            from sklearn.metrics import f1_score
        except ImportError:
            raise ImportError("scikit-learn is required. Install with: pip install scikit-learn")
        
        refs = [str(r).lower().strip() for r in references]
        hyps = [str(h).lower().strip() for h in hypotheses]
        
        labels = sorted(set(refs))
        f1 = f1_score(refs, hyps, labels=labels, average="weighted", zero_division=0)
        
        return {"weighted_f1": f1 * 100}
