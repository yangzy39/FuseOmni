"""
Speech Translation datasets module.
"""

from .covost2 import CoVoST2EnZhDataset, CoVoST2ZhEnDataset
from .fleurs import FLEURSEnZhDataset

__all__ = [
    "CoVoST2EnZhDataset",
    "CoVoST2ZhEnDataset",
    "FLEURSEnZhDataset",
]
