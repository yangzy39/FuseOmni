"""Convert datasets to transformers BatchEncoded or vLLM TokensPrompt formats.

We follow the OpenAI spec for conversational datasets.

ie..,
messages = [
    {"role": "system", "content": "You are AGI"},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "What is my purpose?"},
]

Includes the ability to select from specific categories within the dataset and convert
the dataset into either a language modelling dataset with attention applied to every
token or a prompt-completion dataset for training on completions only with SFTTrainer.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import random
import logging
from functools import partial


import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, BatchEncoding
from vllm import TokensPrompt


logger = logging.getLogger(__name__)


# --- Base Dataset Processors --------------------------------------------------


class BaseDatasetProcessor(ABC):
    category_field: str = "category"
    text_field: str = "text"
    completion_field: str = "completion"
    prompt_field: str = "prompt"
    messages_field: str = "messages"
    all_categories_label: str = "all"

    def __init__(
        self,
        dataset: Dataset | DatasetDict,
        tokenizer: AutoTokenizer,
        pack_samples: bool = True,
        max_input_len: int | None = None,
        split: str | None = None,
        split_by_category: bool = True,
        return_vllm_tokens_prompt: bool = False,
        truncate: bool = False,
        select_only_categories: list[str] | str | None = None,
        processor: Any | None = None,
    ):
        """Defines base functionality for all Dataset Processors.

        Args:
            dataset (Dataset | DatasetDict): _description_
            tokenizer (AutoTokenizer): _description_
            split (str | None, optional): _description_. Defaults to None.
            split_by_category (bool, optional): _description_. Defaults to True.
            return_vllm_tokens_prompt (bool, optional): If True, will return
                TokensPrompt objects instead of BatchEncoding. Defaults to False
            truncate (bool, optional): If True, will truncate the samples from
                the dataset to the max_input_len instead of skipping them.
            processor (Any | None, optional): Multimodal processor for the model.
                Defaults to None.

        """
        if isinstance(dataset, DatasetDict):
            if split is None:
                split = list(dataset.keys())[0]
                logging.warning(
                    f"Using split '{split}' as default for dataset. Available "
                    f"splits: {list(dataset.keys())}",
                )
            dataset = dataset[split]
        if max_input_len is None:
            max_input_len = tokenizer.model_max_length
            logger.warning(
                f"max_input_len is set to {max_input_len} as per tokenizer's "
                f"model_max_length. This will be used for truncation.",
            )
        self.pack_samples = pack_samples
        self.max_input_len = max_input_len
        self.dataset = dataset
        self._mapped_dataset = None
        self.tokenizer = tokenizer
        self.processor = processor
        self.split_by_category = split_by_category
        self.return_vllm_tokens_prompt = return_vllm_tokens_prompt
        self.truncate = truncate
        self.categories = self.get_categories()
        if isinstance(select_only_categories, str):
            select_only_categories = [select_only_categories]
        self.select_only_categories = select_only_categories
        if self.select_only_categories:
            logger.warning(
                "select_only_categories is not None but split_by_category "
                "was False. Setting split_by_category to True and processing "
                f"categories: {self.select_only_categories}"
            )
            self.split_by_category = True
            if self.category_field not in self.dataset.column_names:
                raise RuntimeError(
                    f"Category field '{self.category_field}' not found in dataset. "
                    "Please ensure the dataset has a category field.",
                )
            for category in self.select_only_categories:
                if category not in self.categories:
                    raise RuntimeError(
                        f"Category '{category}' found in dataset. "
                        "Please ensure the dataset has the specified categories.",
                    )
        
        if self.processor is not None and self.pack_samples:
            logger.warning(
                "Processor is provided, which usually implies multimodal data. "
                "Disabling pack_samples as packing multimodal BatchEncoding is not supported."
            )
            self.pack_samples = False

    @staticmethod
    @abstractmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        """Map a row of the dataset to the desired output format.

        EG., map "prompts" and "completions" to "messages" for chat datasets.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses.",
        )

    @abstractmethod
    def _encode_sample(self, sample: str) -> torch.Tensor:
        """Encode a str sample from the desired category of the dataset into
        tokens.
        """
        raise NotImplementedError(
            "This method should be implemented by subclasses.",
        )

    def get_processed_dataset(
        self, samples_per_category: int
    ) -> dict[str, list[TokensPrompt]] | dict[str, list[BatchEncoding]]:
        """Get requests for each category in the dataset."""
        if self._mapped_dataset is None:
            self._mapped_dataset = self.dataset.map(self._map_fn)
        if self.split_by_category:
            categories = (
                self.categories
                if self.select_only_categories is None
                else self.select_only_categories
            )
            return {
                c: self._process_samples_for_category(c, samples_per_category)
                for c in categories
            }
        else:
            return {
                self.all_categories_label: self._process_samples_for_category(
                    self.all_categories_label, samples_per_category
                ),
            }

    def get_categories(self) -> list[str]:
        """Get the unique categories in the dataset."""
        if self.category_field is None:
            logger.warning(
                "No category field specified for dataset, returning 'all' category."
            )
            return ["all"]
        return self.dataset.unique(self.category_field)

    def _process_samples_for_category(
        self,
        category: str,
        samples_per_category: int,
    ) -> list[TokensPrompt] | list[BatchEncoding]:
        if category != self.all_categories_label:
            category_dataset = self._mapped_dataset.filter(
                lambda sample: sample[self.category_field] == category,
            )
        else:
            category_dataset = self._mapped_dataset
            category = self.all_categories_label

        if self.pack_samples:
            return self._process_samples_for_category_packed(
                category, samples_per_category, category_dataset
            )
        else:
            return self._process_samples_for_category_unpacked(
                category, samples_per_category, category_dataset
            )

    def _process_samples_for_category_unpacked(
        self,
        category: str,
        samples_per_category: int,
        category_dataset: Dataset,
    ) -> list[TokensPrompt] | list[BatchEncoding]:
        processed_samples = []
        sampled = []  # sample without replacement
        while len(processed_samples) < samples_per_category:
            if len(sampled) >= len(category_dataset):
                logger.warning(
                    f"Not enough samples in category '{category}' to reach "
                    f"{samples_per_category} samples. Only {len(sampled)} "
                    "samples were processed.",
                )
                break
            sample_idx = random.randint(0, len(category_dataset) - 1)
            if sample_idx in sampled:
                continue
            sampled.append(sample_idx)
            sample = category_dataset[sample_idx]
            encoded_sample = self._encode_sample(sample)
            
            # Handle BatchEncoding by looking at input_ids shape
            if hasattr(encoded_sample, "input_ids"):
                sample_len = encoded_sample.input_ids.shape[-1]
            else:
                sample_len = encoded_sample.shape[-1]

            if sample_len > self.max_input_len:
                if self.truncate:
                    if hasattr(encoded_sample, "input_ids"):
                        # This is a bit complex for BatchEncoding as we'd need to truncate all fields
                        # For now, we'll just skip or do a simple truncate if it's just a tensor
                        logger.warning("Truncation for BatchEncoding is not fully supported. Skipping sample.")
                        continue
                    encoded_sample = encoded_sample[:, : self.max_input_len]
                else:
                    continue

            if self.return_vllm_tokens_prompt:
                encoded_sample = TokensPrompt(
                    prompt_token_ids=encoded_sample[0, :-1].tolist()
                )
            processed_samples.append(encoded_sample)
        return processed_samples

    def _process_samples_for_category_packed(
        self,
        category: str,
        samples_per_category: int,
        category_dataset: Dataset,
    ) -> list[TokensPrompt] | list[BatchEncoding]:
        processed_samples = []
        sampled = []
        while len(processed_samples) < samples_per_category:
            if len(sampled) >= len(category_dataset):
                logger.warning(
                    f"Not enough samples in category '{category}' to reach "
                    f"{samples_per_category} samples. Only {len(sampled)} "
                    "samples were processed.",
                )
                break
            seq = torch.zeros((1, self.max_input_len), dtype=torch.long)
            seq_idx = 0
            while seq_idx < self.max_input_len:
                if len(sampled) >= len(category_dataset):
                    logger.warning(
                        f"Not enough samples to pack last sequence to max_input_len."
                    )
                    break
                sample_idx = random.randint(0, len(category_dataset) - 1)
                if sample_idx in sampled:
                    continue
                sampled.append(sample_idx)
                sample = category_dataset[sample_idx]
                encoded_sample = self._encode_sample(sample)  # shape (batch, seq)
                end_seq = seq_idx + encoded_sample.shape[-1]
                if end_seq > self.max_input_len:
                    encoded_sample = encoded_sample[:, : (self.max_input_len - seq_idx)]
                    end_seq = self.max_input_len
                seq[:, seq_idx:end_seq] = encoded_sample
                seq_idx = end_seq + 1
            if self.return_vllm_tokens_prompt:
                encoded_sample = TokensPrompt(
                    prompt_token_ids=seq[0, :-1].tolist()  # -1 for vLLM.LLM.generate
                )
            else:
                encoded_sample = seq
            processed_samples.append(encoded_sample)
        return processed_samples


class ChatDatasetProcessor(BaseDatasetProcessor):
    def _encode_sample(self, sample: str) -> torch.Tensor:
        if self.processor is not None:
            chat_sample = self.processor.apply_chat_template(
                sample[self.messages_field],
                add_generation_prompt=False,
                tokenize=False,
            )
        else:
            chat_sample = self.tokenizer.apply_chat_template(
                sample[self.messages_field],
                add_generation_prompt=False,
                tokenize=False,
            )
        return self.tokenizer(
            chat_sample,
            truncation=self.truncate,
            max_length=self.tokenizer.model_max_length if self.truncate else None,
            return_tensors="pt",
        )["input_ids"]

    def get_llmcompressor_dataset(self) -> Dataset:
        """Get the mapped dataset without tokenization applied."""

        def chat_template_fn(sample: dict[str, any]) -> dict[str, any]:
            """Apply chat template to the sample."""
            if self.processor is not None:
                chat_sample = self.processor.apply_chat_template(
                    sample[self.messages_field],
                    add_generation_prompt=False,
                    tokenize=False,
                )
            else:
                chat_sample = self.tokenizer.apply_chat_template(
                    sample[self.messages_field],
                    add_generation_prompt=False,
                    tokenize=False,
                )
            return {"text": chat_sample}

        if self._mapped_dataset is None:
            self._mapped_dataset = self.dataset.map(self._map_fn)

        return self._mapped_dataset.map(chat_template_fn)


class LMDatasetProcessor(BaseDatasetProcessor):
    def _encode_sample(self, sample: str) -> torch.Tensor:
        return self.tokenizer(
            sample[self.text_field],
            truncation=self.truncate,
            max_length=self.tokenizer.model_max_length if self.truncate else None,
            return_tensors="pt",
        )["input_ids"]

    def get_llmcompressor_dataset(self) -> Dataset:
        """Get the mapped dataset without tokenization applied."""

        if self._mapped_dataset is None:
            self._mapped_dataset = self.dataset.map(self._map_fn)

        return self._mapped_dataset


### --- Concrete Implementations -----------------------------------------------


class CodeFeedbackChatDataset(ChatDatasetProcessor):
    category_field: str = "lang"
    messages_field: str = "text_fieldmessages"

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class TuluSFTMixtureChatDataset(ChatDatasetProcessor):
    category_field: str = "source"

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample

class PersonasMathChatDataset(ChatDatasetProcessor):
    """Dataset for Tulu-3 SFT Personas Math."""
    
    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample

class WildChatSFTMixtureChatDataset(ChatDatasetProcessor):
    category_field: str = "langauge"

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class MmluChatDataset(ChatDatasetProcessor):
    category_field: str = "subject"

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{sample['question']} "
                        f"Choose from the following options: {sample['choices']}"
                    ),
                },
            ],
        }


class MagicoderEvolInstructChatDataset(ChatDatasetProcessor):
    """Dataset for Magicoder-Evol-Instruct-110K."""

    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return {
            "messages": [
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["response"]},
            ],
        }


class C4LMDataset(LMDatasetProcessor):
    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return sample


class CodeAlpacaChatDataset(ChatDatasetProcessor):
    """Dataset for evol-codealpaca-v1."""

    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return {
            "messages": [
                {"role": "user", "content": sample["instruction"]},
                {"role": "assistant", "content": sample["output"]},
            ],
        }

class WritingPromptsChatDataset(ChatDatasetProcessor):
    """Dataset for WritingPrompts_curated."""
    
    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        return {
            "messages": [
                {"role": "user", "content": f"Please write a creative story using the following writing prompt:\n\n {sample['prompt']}"},
                {"role": "assistant", "content": sample["body"]},
            ],
        }

class FuseOmniChatDataset(ChatDatasetProcessor):
    """Dataset for FuseOmni train.jsonl data with multimodal messages."""
    
    category_field: str = None

    @staticmethod
    def _map_fn(sample: dict[str, any]) -> dict[str, any]:
        messages = sample["messages"]
        transformed = []
        for msg in messages:
            new_content = []
            for c in msg["content"]:
                if c.get("audio_path") is not None:
                    # Qwen-Omni expects 'audio' key for audio content in messages
                    # But process_mm_info might expect 'audio_path' to extract it
                    new_content.append({
                        "type": "audio", 
                        "audio": c["audio_path"],
                        "audio_path": c["audio_path"]
                    })
                elif c.get("image_path") is not None:
                    new_content.append({
                        "type": "image", 
                        "image": c["image_path"],
                        "image_path": c["image_path"]
                    })
                elif c.get("video_path") is not None:
                    new_content.append({
                        "type": "video", 
                        "video": c["video_path"],
                        "video_path": c["video_path"]
                    })
                elif c.get("text") is not None:
                    new_content.append({"type": "text", "text": str(c["text"])})
                elif c.get("type") in ["text", "audio", "image", "video"]:
                    # Already in a recognized format, ensure path keys are present for safety
                    item = c.copy()
                    if item["type"] == "audio" and "audio_path" not in item:
                        item["audio_path"] = item.get("audio")
                    elif item["type"] == "image" and "image_path" not in item:
                        item["image_path"] = item.get("image")
                    elif item["type"] == "video" and "video_path" not in item:
                        item["video_path"] = item.get("video")
                    new_content.append(item)
            transformed.append({"role": msg["role"], "content": new_content})
        
        return {"messages": transformed}
    
    def _fix_text_mismatch(self, text, audios, images, videos):
        """Ensure the number of modality tokens in text matches the number of media items."""
        # Qwen3-Omni/Qwen2-VL tokens usually available on the processor
        audio_token = getattr(self.processor, "audio_token", None)
        # For Qwen2-VL/Qwen3-Omni, image and video tokens might be on the processor or image_processor
        image_token = getattr(self.processor, "image_token", None)
        video_token = getattr(self.processor, "video_token", None)
        
        # Fallbacks if not found on processor (common for some versions/models)
        if audio_token is None: audio_token = "<|audio_pad|>"
        if image_token is None: image_token = "<|vision_pad|>"
        if video_token is None: video_token = "<|video_pad|>"

        def fix_modality(text, token, items, label):
            if token is None or token not in text:
                return text
            
            num_tokens = text.count(token)
            num_items = len(items) if items is not None else 0
            
            if num_tokens > num_items:
                logger.warning(
                    f"Mismatch for {label}: {num_tokens} tokens found, but only {num_items} items provided. "
                    f"Stripping extra {num_tokens - num_items} tokens to avoid StopIteration."
                )
                # Keep only the first num_items tokens
                parts = text.split(token)
                # Re-join the first num_items+1 parts with the token
                new_text = token.join(parts[:num_items+1])
                # Append the remaining parts without the token
                new_text += "".join(parts[num_items+1:])
                return new_text
            return text

        text = fix_modality(text, audio_token, audios, "audio")
        text = fix_modality(text, image_token, images, "image")
        text = fix_modality(text, video_token, videos, "video")
        return text

    def _encode_sample(self, sample: dict[str, any]) -> torch.Tensor | BatchEncoding:
        if self.processor is None:
            return super()._encode_sample(sample)
        
        # Defensive check: if _map_fn wasn't applied or returned unformatted data
        messages = sample.get(self.messages_field)
        if messages is None:
            # Fallback if messages_field is missing
            messages = sample.get("messages", [])
        
        # Check if we need to apply mapping manually (e.g. if map() was cached or skipped)
        needs_mapping = False
        if isinstance(messages, list) and len(messages) > 0:
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    content = msg["content"]
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and any(k in item for k in ["audio_path", "image_path", "video_path"]):
                                needs_mapping = True
                                break
                if needs_mapping: break
        
        if needs_mapping:
            sample = self._map_fn(sample)
            messages = sample["messages"]
        
        from qwen_omni_utils import process_mm_info

        # Qwen3-Omni uses the processor for both chat template and multimodal processing
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        
        # Fix potential mismatch between tokens and data items
        text = self._fix_text_mismatch(text, audios, images, videos)
        
        # Safety check: if the text contains placeholders but no media was extracted,
        # we might hit StopIteration in the processor.
        if "<|audio|>" in text and not audios:
            logger.warning("Found <|audio|> token but no audio data extracted. Processor might fail.")
        if "<|image|>" in text and not images:
            logger.warning("Found <|image|> token but no image data extracted. Processor might fail.")
        if "<|video|>" in text and not videos:
            logger.warning("Found <|video|> token but no video data extracted. Processor might fail.")

        try:
            inputs = self.processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=True
            )
        except StopIteration as e:
            logger.error(f"StopIteration in processor: Likely mismatch between tokens and media data. Text: {text[:100]}...")
            raise e

        # We need to return the full inputs for record_activations
        return inputs



DATASET_REGISTRY: dict[str, BaseDatasetProcessor] = {
    "m-a-p/CodeFeedback-Filtered-Instruction": CodeFeedbackChatDataset,
    "allenai/tulu-3-sft-mixture": TuluSFTMixtureChatDataset,
    "cais/mmlu": MmluChatDataset,
    "ise-uiuc/Magicoder-Evol-Instruct-110K": MagicoderEvolInstructChatDataset,
    "allenai/c4": C4LMDataset,
    "theblackcat102/evol-codealpaca-v1": CodeAlpacaChatDataset,
    "euclaise/WritingPrompts_curated": WritingPromptsChatDataset,
    "allenai/tulu-3-sft-personas-math": PersonasMathChatDataset,
    "fuse_omni_jsonl": FuseOmniChatDataset,
}
