"""
ASR datasets module.
"""

from .librispeech import LibriSpeechCleanDataset, LibriSpeechOtherDataset
from .common_voice import CommonVoiceEnglishDataset, CommonVoiceChineseDataset
from .aishell import AISHELL1Dataset
from .gigaspeech import GigaSpeechDataset
from .wenetspeech import WenetSpeechDataset

__all__ = [
    "LibriSpeechCleanDataset",
    "LibriSpeechOtherDataset",
    "CommonVoiceEnglishDataset",
    "CommonVoiceChineseDataset",
    "AISHELL1Dataset",
    "GigaSpeechDataset",
    "WenetSpeechDataset",
]
