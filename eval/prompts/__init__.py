"""
Prompts module initialization.
"""

from .templates import (
    ASR_PROMPT_EN,
    ASR_PROMPT_ZH,
    MCQ_PROMPT_TEMPLATE,
    QA_PROMPT,
    EMOTION_PROMPT_TEMPLATE,
    INSTRUCTION_PROMPT,
    get_asr_prompt,
    get_translation_prompt,
    get_mcq_prompt,
    get_emotion_prompt,
)
from .chat import format_chat_prompt

__all__ = [
    "ASR_PROMPT_EN",
    "ASR_PROMPT_ZH",
    "MCQ_PROMPT_TEMPLATE",
    "QA_PROMPT",
    "EMOTION_PROMPT_TEMPLATE",
    "INSTRUCTION_PROMPT",
    "get_asr_prompt",
    "get_translation_prompt",
    "get_mcq_prompt",
    "get_emotion_prompt",
    "format_chat_prompt",
]
