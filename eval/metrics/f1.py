"""
F1 and related metrics for QA tasks.
"""

from typing import Any
import re
import string
from collections import Counter
from .base import BaseMetric
from ..registry import register_metric


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(prediction_tokens) if prediction_tokens else 0
    recall = num_same / len(ground_truth_tokens) if ground_truth_tokens else 0
    
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


@register_metric("qa_f1")
class QAF1Metric(BaseMetric):
    """
    Token-level F1 score for question answering.
    
    Based on SQuAD evaluation script.
    """
    
    name = "qa_f1"
    higher_is_better = True
    description = "Token-level F1 for QA"
    
    def compute(
        self,
        references: list[Any],
        hypotheses: list[Any],
        metas: list[dict] | None = None,
    ) -> dict[str, float]:
        """
        Compute average F1 score.
        
        Returns:
            Dictionary with 'f1' score (0-100)
        """
        if len(references) == 0:
            return {"f1": 0.0}
        
        f1_scores = []
        for ref, hyp in zip(references, hypotheses):
            ref_str = str(ref)
            hyp_str = str(hyp)
            
            # Handle multiple reference answers
            if isinstance(ref, list):
                f1 = max(compute_f1(hyp_str, str(r)) for r in ref)
            else:
                f1 = compute_f1(hyp_str, ref_str)
            
            f1_scores.append(f1)
        
        return {"f1": (sum(f1_scores) / len(f1_scores)) * 100}


@register_metric("qa_em")
class QAExactMatchMetric(BaseMetric):
    """
    Exact match metric for question answering.
    """
    
    name = "qa_em"
    higher_is_better = True
    description = "Exact match for QA"
    
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
            return {"em": 0.0}
        
        em_scores = []
        for ref, hyp in zip(references, hypotheses):
            hyp_str = str(hyp)
            
            if isinstance(ref, list):
                em = max(compute_exact_match(hyp_str, str(r)) for r in ref)
            else:
                em = compute_exact_match(hyp_str, str(ref))
            
            em_scores.append(em)
        
        return {"em": (sum(em_scores) / len(em_scores)) * 100}


@register_metric("qa_f1_em")
class QAF1EMMetric(BaseMetric):
    """
    Combined F1 and EM metrics for question answering.
    """
    
    name = "qa_f1_em"
    higher_is_better = True
    description = "Combined F1 and EM for QA"
    
    def compute(
        self,
        references: list[Any],
        hypotheses: list[Any],
        metas: list[dict] | None = None,
    ) -> dict[str, float]:
        """
        Compute both F1 and EM.
        """
        f1_metric = QAF1Metric()
        em_metric = QAExactMatchMetric()
        
        f1_result = f1_metric.compute(references, hypotheses, metas)
        em_result = em_metric.compute(references, hypotheses, metas)
        
        return {**f1_result, **em_result}
