"""
Common types and utilities for dataset processing.
"""

import re
import unicodedata
from typing import Optional


def normalize_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    normalize_unicode: bool = True,
) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Input text
        lowercase: Whether to convert to lowercase
        remove_punctuation: Whether to remove punctuation
        normalize_unicode: Whether to normalize unicode characters
        
    Returns:
        Normalized text
    """
    if normalize_unicode:
        text = unicodedata.normalize("NFKC", text)
    
    if lowercase:
        text = text.lower()
    
    if remove_punctuation:
        # Keep alphanumeric, spaces, and common characters
        text = re.sub(r"[^\w\s]", " ", text)
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    return text.strip()


def normalize_chinese(text: str) -> str:
    """
    Normalize Chinese text by removing spaces between characters.
    """
    # Remove spaces between CJK characters
    result = []
    prev_is_cjk = False
    for char in text:
        is_cjk = "\u4e00" <= char <= "\u9fff"
        if char == " ":
            if not prev_is_cjk:
                result.append(char)
        else:
            result.append(char)
        prev_is_cjk = is_cjk
    return "".join(result)


def strip_special_tokens(text: str) -> str:
    """
    Remove common special tokens from model output.
    """
    # Common special tokens to remove
    special_tokens = [
        "<|endoftext|>", "<|end|>", "</s>", "<s>",
        "<|im_end|>", "<|im_start|>",
        "<|audio_bos|>", "<|audio_eos|>", "<|AUDIO|>",
    ]
    for token in special_tokens:
        text = text.replace(token, "")
    return text.strip()


def parse_mcq_answer(text: str) -> Optional[str]:
    """
    Extract MCQ answer (A/B/C/D) from model output.
    
    Handles various formats:
        - "A"
        - "The answer is A"
        - "A. Some explanation"
        - "(A)"
    
    Returns:
        The answer letter (A/B/C/D) or None if not found
    """
    text = text.strip().upper()
    
    # Direct single letter
    if text in ["A", "B", "C", "D"]:
        return text
    
    # Pattern: starts with letter followed by punctuation or space
    match = re.match(r"^([A-D])[\.\)\:\s]", text)
    if match:
        return match.group(1)
    
    # Pattern: "(A)" or "[A]"
    match = re.search(r"[\(\[]([A-D])[\)\]]", text)
    if match:
        return match.group(1)
    
    # Pattern: "answer is A" or "answer: A"
    match = re.search(r"answer\s*(?:is|:)?\s*([A-D])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern: "option A" or "choice A"
    match = re.search(r"(?:option|choice)\s+([A-D])\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Last resort: find any standalone A/B/C/D
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1)
    
    return None


def parse_json_answer(text: str) -> Optional[dict]:
    """
    Extract JSON from model output.
    
    Returns:
        Parsed JSON dict or None if not found/invalid
    """
    import json
    
    # Try to find JSON in the text
    match = re.search(r"\{[^{}]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_answer_from_brackets(text: str) -> Optional[str]:
    """
    Extract answer from [[answer]] format.
    """
    match = re.search(r"\[\[(.+?)\]\]", text)
    if match:
        return match.group(1).strip()
    return None
