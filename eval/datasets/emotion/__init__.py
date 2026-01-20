"""
Emotion Recognition datasets module.
"""

from .iemocap import IEMOCAPDataset
from .meld import MELDEmotionDataset, MELDSentimentDataset

__all__ = [
    "IEMOCAPDataset",
    "MELDEmotionDataset",
    "MELDSentimentDataset",
]
