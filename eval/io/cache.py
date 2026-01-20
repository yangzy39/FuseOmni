"""
Caching utilities for resumable evaluation runs.
"""

from pathlib import Path
from typing import Optional
import json

from ..schema import EvalPrediction
from .jsonl import read_jsonl, append_jsonl


def load_existing_predictions(output_dir: str | Path) -> dict[str, EvalPrediction]:
    """
    Load existing predictions from a previous run.
    
    Args:
        output_dir: Directory containing predictions.jsonl
        
    Returns:
        Dictionary mapping sample_id to EvalPrediction
    """
    output_dir = Path(output_dir)
    predictions_path = output_dir / "predictions.jsonl"
    
    predictions = {}
    
    for record in read_jsonl(predictions_path):
        pred = EvalPrediction.from_dict(record)
        predictions[pred.sample_id] = pred
    
    return predictions


def save_predictions(
    output_dir: str | Path,
    predictions: list[EvalPrediction],
    append: bool = False,
) -> None:
    """
    Save predictions to file.
    
    Args:
        output_dir: Output directory
        predictions: List of predictions to save
        append: Whether to append to existing file
    """
    output_dir = Path(output_dir)
    predictions_path = output_dir / "predictions.jsonl"
    
    records = [pred.to_dict() for pred in predictions]
    
    if append:
        append_jsonl(predictions_path, records)
    else:
        from .jsonl import write_jsonl
        write_jsonl(predictions_path, records)


def get_completed_ids(output_dir: str | Path) -> set[str]:
    """
    Get set of sample IDs that have already been processed.
    
    Args:
        output_dir: Directory containing predictions.jsonl
        
    Returns:
        Set of completed sample IDs
    """
    predictions = load_existing_predictions(output_dir)
    return set(predictions.keys())
