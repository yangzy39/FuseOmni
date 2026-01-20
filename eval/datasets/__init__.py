"""
Dataset module initialization.

Imports all dataset implementations to register them.
"""

# Import base classes
from .base import BaseDataset, TaskType
from .common_types import (
    normalize_text,
    normalize_chinese,
    strip_special_tokens,
    parse_mcq_answer,
    parse_json_answer,
    extract_answer_from_brackets,
)

# Import dataset implementations to register them
from .asr import *
from .audio_understanding import *
from .spoken_qa import *
from .speech_translation import *
from .emotion import *

__all__ = [
    "BaseDataset",
    "TaskType",
    "normalize_text",
    "normalize_chinese",
    "strip_special_tokens",
    "parse_mcq_answer",
    "parse_json_answer",
    "extract_answer_from_brackets",
]
