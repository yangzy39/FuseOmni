"""
Word Error Rate (WER) metric for ASR evaluation.
"""

from typing import Any
from .base import BaseMetric
from ..registry import register_metric


@register_metric("wer")
class WERMetric(BaseMetric):
    """
    Word Error Rate metric for automatic speech recognition.
    
    WER = (Substitutions + Deletions + Insertions) / Reference Words
    """
    
    name = "wer"
    higher_is_better = False
    description = "Word Error Rate for ASR evaluation"
    
    def __init__(self, normalize: bool = True):
        """
        Args:
            normalize: Whether to normalize text before computing WER
        """
        self.normalize = normalize
    
    def compute(
        self,
        references: list[Any],
        hypotheses: list[Any],
        metas: list[dict] | None = None,
    ) -> dict[str, float]:
        """
        Compute WER between references and hypotheses.
        
        Returns:
            Dictionary with 'wer' as percentage (0-100)
        """
        try:
            import jiwer
        except ImportError:
            raise ImportError("jiwer is required for WER computation. Install with: pip install jiwer")
        
        # Convert to strings
        refs = [str(r) for r in references]
        hyps = [str(h) for h in hypotheses]
        
        if self.normalize:
            # Standard ASR normalization
            transform = jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
            ])
            refs = transform(refs)
            hyps = transform(hyps)
        
        # Handle empty references
        if not refs or all(len(r.strip()) == 0 for r in refs):
            return {"wer": 0.0 if all(len(h.strip()) == 0 for h in hyps) else 100.0}
        
        wer = jiwer.wer(refs, hyps)
        
        return {"wer": wer * 100}  # Return as percentage


@register_metric("cer")
class CERMetric(BaseMetric):
    """
    Character Error Rate metric for ASR evaluation.
    
    Similar to WER but operates on character level.
    Useful for languages without clear word boundaries (e.g., Chinese).
    """
    
    name = "cer"
    higher_is_better = False
    description = "Character Error Rate for ASR evaluation"
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
    
    def compute(
        self,
        references: list[Any],
        hypotheses: list[Any],
        metas: list[dict] | None = None,
    ) -> dict[str, float]:
        """
        Compute CER between references and hypotheses.
        
        Returns:
            Dictionary with 'cer' as percentage (0-100)
        """
        try:
            import jiwer
        except ImportError:
            raise ImportError("jiwer is required for CER computation. Install with: pip install jiwer")
        
        refs = [str(r) for r in references]
        hyps = [str(h) for h in hypotheses]
        
        if self.normalize:
            transform = jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
            ])
            refs = transform(refs)
            hyps = transform(hyps)
        
        # Handle empty references
        if not refs or all(len(r.strip()) == 0 for r in refs):
            return {"cer": 0.0 if all(len(h.strip()) == 0 for h in hyps) else 100.0}
        
        cer = jiwer.cer(refs, hyps)
        
        return {"cer": cer * 100}
