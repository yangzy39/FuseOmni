"""
Chat formatting utilities.
"""

from typing import Optional


def format_chat_prompt(
    system: Optional[str] = None,
    user: str = "",
    audio_placeholder: str = "<|audio_bos|><|AUDIO|><|audio_eos|>",
    model_type: str = "qwen",
) -> str:
    """
    Format a chat prompt for the given model type.
    
    Args:
        system: System message (optional)
        user: User message
        audio_placeholder: Audio placeholder tokens
        model_type: Model type (qwen, llama, etc.)
        
    Returns:
        Formatted prompt string
    """
    if model_type.startswith("qwen"):
        # Qwen chat format
        parts = []
        if system:
            parts.append(f"<|im_start|>system\n{system}<|im_end|>")
        parts.append(f"<|im_start|>user\n{audio_placeholder}\n{user}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
    
    elif model_type.startswith("llama"):
        # Llama chat format
        parts = []
        if system:
            parts.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>")
        parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{audio_placeholder}\n{user}<|eot_id|>")
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)
    
    else:
        # Default simple format
        if system:
            return f"{system}\n\n{audio_placeholder}\n{user}\n"
        return f"{audio_placeholder}\n{user}\n"
