"""
Prompt templates for different task types.
"""

# ASR prompts
ASR_PROMPT_EN = "Transcribe the following audio exactly as spoken."
ASR_PROMPT_ZH = "请将以下语音转录为文字。"

# Translation prompts
TRANSLATION_PROMPT_TEMPLATE = "Translate the following {src_lang} speech to {tgt_lang}."

# MCQ prompts
MCQ_PROMPT_TEMPLATE = """Listen to the audio and answer the following question.

Question: {question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}

Answer with only the letter (A, B, C, or D)."""

# QA prompts
QA_PROMPT = "Listen to the audio and answer the question based on the content."

# Emotion prompts
EMOTION_PROMPT_TEMPLATE = """Listen to the audio and identify the emotion expressed.

Choose from: {emotions}

Answer with only the emotion label."""

# Instruction following prompts
INSTRUCTION_PROMPT = "Listen to the audio and follow the instruction or respond appropriately."


def get_asr_prompt(language: str = "en") -> str:
    """Get ASR prompt for the specified language."""
    if language.startswith("zh"):
        return ASR_PROMPT_ZH
    return ASR_PROMPT_EN


def get_translation_prompt(src_lang: str, tgt_lang: str) -> str:
    """Get translation prompt for the language pair."""
    lang_names = {
        "en": "English",
        "zh": "Chinese",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "ja": "Japanese",
        "ko": "Korean",
    }
    src_name = lang_names.get(src_lang, src_lang)
    tgt_name = lang_names.get(tgt_lang, tgt_lang)
    return TRANSLATION_PROMPT_TEMPLATE.format(src_lang=src_name, tgt_lang=tgt_name)


def get_mcq_prompt(question: str, options: list[str]) -> str:
    """Get MCQ prompt with the question and options."""
    if len(options) < 4:
        options = options + [""] * (4 - len(options))
    return MCQ_PROMPT_TEMPLATE.format(
        question=question,
        option_a=options[0],
        option_b=options[1],
        option_c=options[2],
        option_d=options[3],
    )


def get_emotion_prompt(emotions: list[str]) -> str:
    """Get emotion classification prompt."""
    return EMOTION_PROMPT_TEMPLATE.format(emotions=", ".join(emotions))
