"""
BLEU metric for translation tasks.
"""

from typing import Any
from .base import BaseMetric
from ..registry import register_metric


@register_metric("bleu")
class BLEUMetric(BaseMetric):
    """
    BLEU score for machine translation evaluation.
    
    Uses sacrebleu for standardized BLEU computation.
    """
    
    name = "bleu"
    higher_is_better = True
    description = "BLEU score for translation"
    
    def __init__(self, tokenize: str = "13a"):
        """
        Args:
            tokenize: Tokenization scheme for sacrebleu
        """
        self.tokenize = tokenize
    
    def compute(
        self,
        references: list[Any],
        hypotheses: list[Any],
        metas: list[dict] | None = None,
    ) -> dict[str, float]:
        """
        Compute BLEU score.
        
        Returns:
            Dictionary with 'bleu' score (0-100)
        """
        try:
            import sacrebleu
        except ImportError:
            raise ImportError("sacrebleu is required for BLEU. Install with: pip install sacrebleu")
        
        refs = [[str(r) for r in references]]  # sacrebleu expects list of reference lists
        hyps = [str(h) for h in hypotheses]
        
        bleu = sacrebleu.corpus_bleu(hyps, refs, tokenize=self.tokenize)
        
        return {"bleu": bleu.score}


@register_metric("chrf")
class ChrFMetric(BaseMetric):
    """
    chrF score for translation evaluation.
    
    Character-level metric, useful for morphologically rich languages.
    """
    
    name = "chrf"
    higher_is_better = True
    description = "chrF score for translation"
    
    def compute(
        self,
        references: list[Any],
        hypotheses: list[Any],
        metas: list[dict] | None = None,
    ) -> dict[str, float]:
        """
        Compute chrF score.
        """
        try:
            import sacrebleu
        except ImportError:
            raise ImportError("sacrebleu is required for chrF. Install with: pip install sacrebleu")
        
        refs = [[str(r) for r in references]]
        hyps = [str(h) for h in hypotheses]
        
        chrf = sacrebleu.corpus_chrf(hyps, refs)
        
        return {"chrf": chrf.score}
