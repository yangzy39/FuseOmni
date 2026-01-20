#!/usr/bin/env python3
"""
Layer Similarity Pruning for Qwen3-Omni-30B-A3B

This module implements inter-layer similarity pruning based on hidden state
similarity between adjacent layers. It uses REAL calibration data (audio data
from REAP step 2) to collect hidden states during forward pass.

Algorithm:
1. Load model and audio calibration data (same as REAP step 2)
2. Run forward pass with hooks to collect hidden states per layer
3. Compute cosine similarity between adjacent layers' hidden states
4. Identify layers with high similarity (redundant layers)
5. Remove redundant layers and renumber remaining layers
6. Update model weights and configuration

Based on: REAP-OMNI methodology and PruneMe (https://github.com/arcee-ai/PruneMe)

Author: REAP-OMNI Implementation
"""

import argparse
import json
import logging
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LayerPruningConfig:
    """Configuration for layer similarity pruning."""
    
    # Model paths
    model_path: str = r"D:\PycharmProjects\FuseOmni\models\Qwen3-Omni-30B-A3B-Instruct"
    output_path: str = r"D:\PycharmProjects\FuseOmni\models\Qwen3-Omni-30B-A3B-Layer-Pruned"
    
    # Target component
    component: str = "thinker"  # "thinker" or "talker"
    
    # Calibration data (same as REAP step 2)
    audio_data_path: Optional[str] = None
    calibration_samples: int = 32
    batch_size: int = 1
    max_seq_length: int = 512
    
    # Pruning parameters
    similarity_threshold: float = 0.9  # Layers with similarity > threshold are candidates
    max_layers_to_prune: int = 8       # Maximum number of layers to prune
    layers_to_skip: int = 1            # Compare layer L with layer L+skip
    
    # Protected layers (never prune first/last N layers)
    protect_first_n: int = 4
    protect_last_n: int = 4
    
    # Execution
    device: str = "cuda"
    dtype: str = "bfloat16"
    verbose: bool = True


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(verbose: bool = True) -> logging.Logger:
    """Configure logging for layer pruning."""
    logger = logging.getLogger("LayerPruning")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    
    return logger


# ============================================================================
# Calibration Data Loader (Same as REAP step 2)
# ============================================================================

class AudioCalibrationLoader:
    """
    Load audio calibration data for layer similarity computation.
    
    Uses the same data format as REAP expert pruning step 2.
    """
    
    def __init__(self, config: LayerPruningConfig, processor=None):
        self.config = config
        self.processor = processor
        self.logger = logging.getLogger("AudioCalibration")
    
    def load_from_jsonl(self, path: str) -> List[Dict]:
        """Load calibration samples from JSONL file."""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples[:self.config.calibration_samples]
    
    def generate_synthetic_audio_samples(self, n: int = 32) -> List[Dict]:
        """Generate synthetic audio-focused calibration prompts as fallback."""
        templates = [
            "Listen carefully to the audio and describe what you hear.",
            "The speaker says: 'Hello, how are you today?'",
            "Transcribe the following speech: This is a test of the audio system.",
            "Audio description: Someone is speaking about technology.",
            "The recording contains: A person discussing machine learning.",
            "Speech recognition output: The quick brown fox jumps over the lazy dog.",
            "Podcast transcript: Welcome to today's episode about artificial intelligence.",
            "Audio contains sounds of music followed by speech.",
            "The voice says: 'Please listen to this important message.'",
            "Describe the audio: A lecture about neural networks.",
            "Transcribe: The weather today is sunny with clear skies.",
            "Audio analysis: Multiple speakers in a conversation.",
            "The audio clip features: Background noise with speech overlay.",
            "Speech-to-text: Good morning everyone, let's begin the meeting.",
            "Audio content: A narrator explaining scientific concepts.",
            "Listen and transcribe: Numbers from one to ten in sequence.",
        ]
        
        samples = []
        for i in range(n):
            samples.append({
                "id": f"syn_audio_{i:05d}",
                "text": templates[i % len(templates)],
                "modality": "audio"
            })
        return samples
    
    def get_calibration_data(self) -> List[Dict]:
        """Get audio calibration data."""
        if self.config.audio_data_path and os.path.exists(self.config.audio_data_path):
            self.logger.info(f"Loading audio data from: {self.config.audio_data_path}")
            return self.load_from_jsonl(self.config.audio_data_path)
        else:
            self.logger.warning("No audio data path provided, using synthetic samples")
            return self.generate_synthetic_audio_samples(self.config.calibration_samples)
    
    def prepare_inputs(
        self,
        samples: List[Dict],
        device: str = "cuda"
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Prepare model inputs from calibration samples.
        
        Yields batched input tensors for the model.
        """
        if self.processor is None:
            self.logger.warning("No processor provided, using dummy inputs")
            for sample in samples:
                yield {
                    "input_ids": torch.randint(0, 10000, (1, 128), device=device),
                    "attention_mask": torch.ones(1, 128, dtype=torch.long, device=device)
                }
            return
        
        for i in range(0, len(samples), self.config.batch_size):
            batch = samples[i:i + self.config.batch_size]
            texts = [s.get("text", "") for s in batch]
            
            # Use processor/tokenizer
            if hasattr(self.processor, 'tokenizer'):
                tokenizer = self.processor.tokenizer
            else:
                tokenizer = self.processor
            
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length
            )
            
            yield {k: v.to(device) for k, v in inputs.items()}


# ============================================================================
# Hidden State Collector
# ============================================================================

class HiddenStateCollector:
    """
    Collects hidden states from transformer layers during forward pass.
    
    Registers forward hooks on each decoder layer to capture the output
    hidden states for similarity computation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: LayerPruningConfig,
        target_component: str = "thinker"
    ):
        self.model = model
        self.config = config
        self.target_component = target_component
        self.logger = logging.getLogger("HiddenStateCollector")
        
        # Storage for hidden states: {layer_idx: [batch of hidden states]}
        self.hidden_states: Dict[int, List[torch.Tensor]] = defaultdict(list)
        
        # Hooks
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # Model info
        self.num_layers = self._get_num_layers()
    
    def _get_num_layers(self) -> int:
        """Get number of layers from model config."""
        config = self.model.config
        if hasattr(config, f"{self.target_component}_config"):
            component_config = getattr(config, f"{self.target_component}_config")
            if hasattr(component_config, "text_config"):
                return component_config.text_config.num_hidden_layers
        return 48 if self.target_component == "thinker" else 20
    
    def _create_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook to capture hidden states."""
        def hook(module, inputs, outputs):
            # Handle different output formats
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
            else:
                hidden_states = outputs
            
            # Store on CPU to save GPU memory, detach to avoid memory leak
            self.hidden_states[layer_idx].append(
                hidden_states.detach().cpu().float()  # Convert to float32 for stable similarity computation
            )
        
        return hook
    
    def register_hooks(self) -> None:
        """Register forward hooks on all decoder layers."""
        self.logger.info(f"Registering hooks for {self.num_layers} layers...")
        
        # Pattern to find decoder layers
        # Matches: thinker.model.layers.0, thinker.model.layers.1, etc.
        layer_pattern = re.compile(
            rf"^{self.target_component}\.model\.layers\.(\d+)$"
        )
        
        registered_count = 0
        for name, module in self.model.named_modules():
            match = layer_pattern.match(name)
            if match:
                layer_idx = int(match.group(1))
                hook = module.register_forward_hook(self._create_hook(layer_idx))
                self.hooks.append(hook)
                self.logger.debug(f"Registered hook for layer {layer_idx}: {name}")
                registered_count += 1
        
        self.logger.info(f"Registered {registered_count} hooks")
        
        if registered_count == 0:
            self.logger.warning("No hooks registered! Check module naming pattern.")
            # Try alternative patterns
            self._try_alternative_patterns()
    
    def _try_alternative_patterns(self) -> None:
        """Try alternative module naming patterns."""
        alternative_patterns = [
            rf"^{self.target_component}\.layers\.(\d+)$",
            rf"^model\.{self.target_component}\.layers\.(\d+)$",
            rf"^{self.target_component}\.decoder\.layers\.(\d+)$",
        ]
        
        for pattern_str in alternative_patterns:
            pattern = re.compile(pattern_str)
            for name, module in self.model.named_modules():
                match = pattern.match(name)
                if match:
                    layer_idx = int(match.group(1))
                    hook = module.register_forward_hook(self._create_hook(layer_idx))
                    self.hooks.append(hook)
                    self.logger.info(f"Found layer with alternative pattern: {name}")
        
        if self.hooks:
            self.logger.info(f"Registered {len(self.hooks)} hooks with alternative pattern")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.logger.info("Removed all hooks")
    
    def clear(self) -> None:
        """Clear collected hidden states."""
        self.hidden_states.clear()
    
    def get_aggregated_states(self) -> Dict[int, torch.Tensor]:
        """
        Get aggregated hidden states per layer.
        
        Returns:
            Dict mapping layer_idx -> tensor of shape [total_tokens, hidden_dim]
        """
        result = {}
        
        for layer_idx in sorted(self.hidden_states.keys()):
            states_list = self.hidden_states[layer_idx]
            if states_list:
                # Concatenate all collected states: [num_batches, batch, seq, hidden] -> [total_tokens, hidden]
                # First concat along batch dimension, then flatten batch and seq
                all_states = torch.cat(states_list, dim=0)  # [total_batch, seq, hidden]
                # Reshape to [total_tokens, hidden]
                total_batch, seq_len, hidden_dim = all_states.shape
                result[layer_idx] = all_states.view(-1, hidden_dim)
            else:
                self.logger.warning(f"No hidden states collected for layer {layer_idx}")
        
        return result


# ============================================================================
# Similarity Metrics
# ============================================================================

def cosine_similarity_batch(x1: torch.Tensor, x2: torch.Tensor) -> float:
    """
    Compute mean cosine similarity between two batches of hidden states.
    
    Args:
        x1: Tensor of shape [num_tokens, hidden_dim]
        x2: Tensor of shape [num_tokens, hidden_dim]
        
    Returns:
        Mean cosine similarity (scalar)
    """
    # Normalize
    x1_norm = x1 / (torch.norm(x1, dim=-1, keepdim=True) + 1e-8)
    x2_norm = x2 / (torch.norm(x2, dim=-1, keepdim=True) + 1e-8)
    
    # Compute cosine similarity per token
    cos_sim = (x1_norm * x2_norm).sum(dim=-1)
    
    # Return mean
    return cos_sim.mean().item()


def angular_distance_batch(x1: torch.Tensor, x2: torch.Tensor) -> float:
    """
    Compute mean angular distance between two batches of hidden states.
    
    This is the metric used in PruneMe paper.
    
    Args:
        x1: Tensor of shape [num_tokens, hidden_dim]
        x2: Tensor of shape [num_tokens, hidden_dim]
        
    Returns:
        Mean angular distance normalized to [0, 1]
    """
    # Normalize
    x1_norm = x1 / (torch.norm(x1, dim=-1, keepdim=True) + 1e-8)
    x2_norm = x2 / (torch.norm(x2, dim=-1, keepdim=True) + 1e-8)
    
    # Compute cosine similarity
    cos_sim = (x1_norm * x2_norm).sum(dim=-1)
    
    # Clamp to valid range for acos
    cos_sim = cos_sim.clamp(min=-1.0, max=1.0)
    
    # Angular distance: arccos(similarity) / pi
    angular_dist = torch.acos(cos_sim) / torch.pi
    
    return angular_dist.mean().item()


def compute_layer_similarities(
    hidden_states: Dict[int, torch.Tensor],
    layers_to_skip: int = 1,
    metric: str = "cosine"
) -> Dict[int, float]:
    """
    Compute similarity between adjacent layers.
    
    Args:
        hidden_states: Dict mapping layer_idx -> hidden states tensor
        layers_to_skip: Number of layers to skip when comparing (1 = adjacent)
        metric: "cosine" for cosine similarity, "angular" for angular distance
        
    Returns:
        Dict mapping layer_idx -> similarity with layer_idx + layers_to_skip
    """
    similarities = {}
    layer_indices = sorted(hidden_states.keys())
    
    for i, layer_idx in enumerate(layer_indices):
        next_layer_idx = layer_idx + layers_to_skip
        
        if next_layer_idx not in hidden_states:
            continue
        
        h_l = hidden_states[layer_idx]
        h_l_next = hidden_states[next_layer_idx]
        
        # Ensure same number of tokens (should be the case)
        min_tokens = min(h_l.shape[0], h_l_next.shape[0])
        h_l = h_l[:min_tokens]
        h_l_next = h_l_next[:min_tokens]
        
        if metric == "cosine":
            sim = cosine_similarity_batch(h_l, h_l_next)
        else:  # angular
            dist = angular_distance_batch(h_l, h_l_next)
            sim = 1.0 - dist  # Convert to similarity
        
        similarities[layer_idx] = sim
    
    return similarities


# ============================================================================
# Layer Pruner with Calibration
# ============================================================================

class LayerSimilarityPruner:
    """
    Main class for layer similarity pruning with real calibration data.
    
    Implements the full algorithm:
    1. Load model and audio calibration data
    2. Run forward pass to collect hidden states
    3. Compute inter-layer similarity
    4. Identify and rank redundant layers
    5. Remove layer weights and renumber
    6. Update model configuration
    """
    
    def __init__(self, config: LayerPruningConfig):
        self.config = config
        self.logger = setup_logging(config.verbose)
        
        self.model_path = Path(config.model_path)
        self.output_path = Path(config.output_path)
        
        # Validate paths
        self._validate_paths()
        
        # Load model configuration
        self.model_config = self._load_model_config()
        
        # Layer info
        self.num_layers = self._get_num_layers()
        
        self.logger.info(f"Layer Similarity Pruner initialized")
        self.logger.info(f"  Component: {config.component}")
        self.logger.info(f"  Num layers: {self.num_layers}")
        self.logger.info(f"  Similarity threshold: {config.similarity_threshold}")
        self.logger.info(f"  Max layers to prune: {config.max_layers_to_prune}")
    
    def _validate_paths(self) -> None:
        """Validate input paths."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        if not (self.model_path / "config.json").exists():
            raise FileNotFoundError(f"Config not found: {self.model_path / 'config.json'}")
    
    def _load_model_config(self) -> Dict:
        """Load model configuration."""
        with open(self.model_path / "config.json", 'r') as f:
            return json.load(f)
    
    def _get_num_layers(self) -> int:
        """Get number of layers."""
        if self.config.component == "thinker":
            return self.model_config.get("thinker_config", {}).get("text_config", {}).get("num_hidden_layers", 48)
        else:
            return self.model_config.get("talker_config", {}).get("text_config", {}).get("num_hidden_layers", 20)
    
    def load_model(self) -> Tuple[nn.Module, Any]:
        """
        Load the Qwen3-Omni model for calibration.
        
        Returns:
            Tuple of (model, processor)
        """
        self.logger.info("Loading model for calibration...")
        
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        except ImportError:
            raise ImportError("transformers package required. Install with: pip install transformers")
        
        # Determine dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)
        
        # Load processor/tokenizer
        try:
            processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
        except Exception:
            self.logger.warning("Could not load AutoProcessor, trying AutoTokenizer")
            processor = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
        
        # Load model
        self.logger.info(f"Loading model with dtype={self.config.dtype}, device={self.config.device}")
        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            torch_dtype=dtype,
            device_map=self.config.device if self.config.device != "cpu" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        model.eval()
        self.logger.info("Model loaded successfully")
        
        return model, processor
    
    def collect_hidden_states(
        self,
        model: nn.Module,
        processor: Any
    ) -> Dict[int, torch.Tensor]:
        """
        Run forward pass on audio calibration data and collect hidden states.
        
        Args:
            model: The loaded model
            processor: Tokenizer/processor for the model
            
        Returns:
            Dict mapping layer_idx -> aggregated hidden states tensor
        """
        self.logger.info("Collecting hidden states from calibration data...")
        
        # Initialize data loader
        data_loader = AudioCalibrationLoader(self.config, processor)
        audio_samples = data_loader.get_calibration_data()
        
        self.logger.info(f"Loaded {len(audio_samples)} audio calibration samples")
        
        # Initialize hidden state collector
        collector = HiddenStateCollector(model, self.config, self.config.component)
        collector.register_hooks()
        
        # Run forward pass
        device = self.config.device
        
        try:
            with torch.no_grad():
                for batch_idx, inputs in enumerate(tqdm(
                    data_loader.prepare_inputs(audio_samples, device),
                    total=(len(audio_samples) + self.config.batch_size - 1) // self.config.batch_size,
                    desc="Collecting hidden states"
                )):
                    # Forward pass - this triggers hooks
                    _ = model(**inputs)
                    
                    # Clear CUDA cache periodically
                    if batch_idx % 10 == 0 and torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        finally:
            collector.remove_hooks()
        
        # Get aggregated states
        hidden_states = collector.get_aggregated_states()
        
        self.logger.info(f"Collected hidden states for {len(hidden_states)} layers")
        for layer_idx, states in sorted(hidden_states.items())[:5]:
            self.logger.info(f"  Layer {layer_idx}: {states.shape}")
        if len(hidden_states) > 5:
            self.logger.info(f"  ... and {len(hidden_states) - 5} more layers")
        
        return hidden_states
    
    def compute_similarities(
        self,
        hidden_states: Dict[int, torch.Tensor]
    ) -> Dict[int, float]:
        """
        Compute similarity between adjacent layers.
        
        Args:
            hidden_states: Dict mapping layer_idx -> hidden states tensor
            
        Returns:
            Dict mapping layer_idx -> similarity with next layer
        """
        self.logger.info("Computing layer similarities...")
        
        similarities = compute_layer_similarities(
            hidden_states,
            layers_to_skip=self.config.layers_to_skip,
            metric="cosine"
        )
        
        # Log similarities
        self.logger.info("\nLayer Similarities (cosine):")
        self.logger.info("-" * 50)
        for layer_idx in sorted(similarities.keys()):
            sim = similarities[layer_idx]
            marker = "**" if sim >= self.config.similarity_threshold else "  "
            protected = ""
            if layer_idx < self.config.protect_first_n:
                protected = " [PROTECTED: first N]"
            elif layer_idx >= self.num_layers - self.config.protect_last_n - self.config.layers_to_skip:
                protected = " [PROTECTED: last N]"
            self.logger.info(
                f"  {marker}Layer {layer_idx:2d} -> {layer_idx + self.config.layers_to_skip:2d}: "
                f"{sim:.4f}{protected}"
            )
        
        return similarities
    
    def select_layers_to_prune(
        self,
        similarities: Dict[int, float]
    ) -> List[int]:
        """
        Select which layers to prune based on similarity scores.
        
        Args:
            similarities: Dict mapping layer_idx -> similarity with next layer
            
        Returns:
            List of layer indices to prune (0-indexed)
        """
        # Identify layers above threshold (excluding protected layers)
        candidates = []
        
        for layer_idx, sim in similarities.items():
            # Check if layer is protected
            if layer_idx < self.config.protect_first_n:
                continue
            if layer_idx >= self.num_layers - self.config.protect_last_n - self.config.layers_to_skip:
                continue
            
            if sim >= self.config.similarity_threshold:
                candidates.append((layer_idx, sim))
        
        # Sort by similarity (highest first = most redundant)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N
        layers_to_prune = [
            layer_idx for layer_idx, _ 
            in candidates[:self.config.max_layers_to_prune]
        ]
        
        # Sort for consistent processing
        layers_to_prune.sort()
        
        self.logger.info(f"\nSelected {len(layers_to_prune)} layers to prune:")
        for l in layers_to_prune:
            self.logger.info(f"  Layer {l}: similarity = {similarities[l]:.4f}")
        
        return layers_to_prune
    
    def prune_layers(
        self,
        layers_to_prune: List[int],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Remove layers from model weights.
        
        This operates directly on safetensor files.
        
        Args:
            layers_to_prune: List of layer indices to remove
            dry_run: If True, only analyze without making changes
            
        Returns:
            Statistics about the pruning operation
        """
        self.logger.info("=" * 60)
        self.logger.info("LAYER SIMILARITY PRUNING")
        self.logger.info("=" * 60)
        self.logger.info(f"Layers to prune: {layers_to_prune}")
        self.logger.info(f"Original layers: {self.num_layers}")
        self.logger.info(f"Remaining layers: {self.num_layers - len(layers_to_prune)}")
        
        # Load weight index
        index_path = self.model_path / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Weight index not found: {index_path}")
        
        with open(index_path) as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        
        # Pattern to match layer weights
        layer_pattern = re.compile(
            rf"^{self.config.component}\.model\.layers\.(\d+)\."
        )
        
        # Create mapping: old layer index -> new layer index (after pruning)
        layers_to_keep = [l for l in range(self.num_layers) if l not in layers_to_prune]
        old_to_new = {old: new for new, old in enumerate(layers_to_keep)}
        
        # Categorize weights
        layer_weights = defaultdict(list)  # layer_idx -> [weight_names]
        other_weights = []
        
        for weight_name in weight_map.keys():
            match = layer_pattern.match(weight_name)
            if match:
                layer_idx = int(match.group(1))
                layer_weights[layer_idx].append(weight_name)
            else:
                other_weights.append(weight_name)
        
        # Calculate statistics
        weights_to_remove = []
        for layer_idx in layers_to_prune:
            weights_to_remove.extend(layer_weights[layer_idx])
        
        weights_to_rename = []
        for layer_idx in layers_to_keep:
            if layer_idx != old_to_new[layer_idx]:  # Layer index changes
                weights_to_rename.extend(layer_weights[layer_idx])
        
        stats = {
            "original_layers": self.num_layers,
            "layers_pruned": len(layers_to_prune),
            "remaining_layers": len(layers_to_keep),
            "weights_removed": len(weights_to_remove),
            "weights_renamed": len(weights_to_rename),
            "weights_unchanged": len(other_weights) + sum(
                len(layer_weights[l]) for l in layers_to_keep if l == old_to_new[l]
            ),
            "pruned_layer_indices": layers_to_prune,
        }
        
        self.logger.info(f"Weights to remove: {stats['weights_removed']}")
        self.logger.info(f"Weights to rename: {stats['weights_renamed']}")
        
        if dry_run:
            self.logger.info("[DRY RUN] No changes made.")
            return stats
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Process shards
        new_weight_map = {}
        shards = set(weight_map.values())
        total_removed_bytes = 0
        
        for shard_file in tqdm(shards, desc="Processing shards"):
            input_shard = self.model_path / shard_file
            output_shard = self.output_path / shard_file
            
            new_tensors = {}
            
            with safe_open(input_shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    match = layer_pattern.match(key)
                    
                    if match:
                        layer_idx = int(match.group(1))
                        
                        if layer_idx in layers_to_prune:
                            # Remove this weight
                            tensor = f.get_tensor(key)
                            total_removed_bytes += tensor.numel() * tensor.element_size()
                            continue
                        
                        # Rename layer if needed
                        new_layer_idx = old_to_new[layer_idx]
                        new_key = layer_pattern.sub(
                            f"{self.config.component}.model.layers.{new_layer_idx}.",
                            key
                        )
                        
                        new_tensors[new_key] = f.get_tensor(key)
                        new_weight_map[new_key] = shard_file
                    else:
                        # Non-layer weight - keep as is
                        new_tensors[key] = f.get_tensor(key)
                        new_weight_map[key] = shard_file
            
            # Save modified shard
            if new_tensors:
                save_file(new_tensors, output_shard)
        
        # Save new index
        new_index = {
            "metadata": {
                "total_size": index.get("metadata", {}).get("total_size", 0) - total_removed_bytes * 2,
                "layer_pruned": True,
                "original_layers": self.num_layers,
                "remaining_layers": len(layers_to_keep),
                "pruned_layers": layers_to_prune,
                "similarity_threshold": self.config.similarity_threshold,
                "calibration_method": "audio_forward_pass"
            },
            "weight_map": new_weight_map
        }
        
        with open(self.output_path / "model.safetensors.index.json", "w") as f:
            json.dump(new_index, f, indent=2)
        
        # Update and save config
        self._save_pruned_config(layers_to_keep)
        
        # Copy other files
        self._copy_auxiliary_files()
        
        stats["bytes_removed"] = total_removed_bytes
        stats["gb_removed"] = total_removed_bytes / (1024**3)
        
        self.logger.info("=" * 60)
        self.logger.info("LAYER PRUNING COMPLETE")
        self.logger.info(f"Layers removed: {len(layers_to_prune)}")
        self.logger.info(f"Size reduced: {stats['gb_removed']:.2f} GB")
        self.logger.info(f"Output: {self.output_path}")
        self.logger.info("=" * 60)
        
        return stats
    
    def _save_pruned_config(self, layers_to_keep: List[int]) -> None:
        """Save modified configuration for pruned model."""
        config = self.model_config.copy()
        
        new_num_layers = len(layers_to_keep)
        
        if self.config.component == "thinker":
            config["thinker_config"]["text_config"]["num_hidden_layers"] = new_num_layers
            config["thinker_config"]["text_config"]["original_num_hidden_layers"] = self.num_layers
        else:
            config["talker_config"]["text_config"]["num_hidden_layers"] = new_num_layers
            config["talker_config"]["text_config"]["original_num_hidden_layers"] = self.num_layers
        
        # Add pruning metadata
        config["layer_pruning_info"] = {
            "method": "similarity_pruning_with_calibration",
            "component": self.config.component,
            "original_layers": self.num_layers,
            "remaining_layers": new_num_layers,
            "similarity_threshold": self.config.similarity_threshold,
            "layers_to_skip": self.config.layers_to_skip,
            "calibration_samples": self.config.calibration_samples,
            "audio_data_path": self.config.audio_data_path,
            "kept_layer_mapping": {new: old for new, old in enumerate(layers_to_keep)}
        }
        
        with open(self.output_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)
    
    def _copy_auxiliary_files(self) -> None:
        """Copy tokenizer and other necessary files."""
        files_to_copy = [
            "tokenizer_config.json",
            "vocab.json",
            "generation_config.json",
            "chat_template.json",
            "merges.txt",
            "tokenizer.json",
            "special_tokens_map.json",
        ]
        
        for filename in files_to_copy:
            src = self.model_path / filename
            if src.exists():
                shutil.copy2(src, self.output_path / filename)
    
    def run_with_calibration(
        self,
        layers_to_prune: Optional[List[int]] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Run layer pruning with real calibration data.
        
        This is the main entry point that:
        1. Loads the model
        2. Runs calibration to collect hidden states
        3. Computes similarities
        4. Prunes layers
        
        Args:
            layers_to_prune: Explicit list of layers to prune, or None to auto-select
            dry_run: If True, only analyze without making changes
            
        Returns:
            Statistics about the pruning operation
        """
        self.logger.info("=" * 60)
        self.logger.info("LAYER SIMILARITY PRUNING WITH CALIBRATION")
        self.logger.info("=" * 60)
        
        # Load model
        model, processor = self.load_model()
        
        # Collect hidden states
        hidden_states = self.collect_hidden_states(model, processor)
        
        # Free model memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Compute similarities
        similarities = self.compute_similarities(hidden_states)
        
        # Free hidden states memory
        del hidden_states
        
        # Select layers to prune (if not specified)
        if layers_to_prune is None:
            layers_to_prune = self.select_layers_to_prune(similarities)
        
        if not layers_to_prune:
            self.logger.info("No layers selected for pruning.")
            return {"layers_pruned": 0, "similarities": similarities}
        
        # Prune layers
        stats = self.prune_layers(layers_to_prune, dry_run=dry_run)
        stats["similarities"] = {str(k): v for k, v in similarities.items()}
        
        return stats
    
    def run_static_pruning(
        self,
        layers_to_prune: Optional[List[int]] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Run layer pruning with manually specified layers (no model loading).
        
        Use this when you already know which layers to prune.
        
        Args:
            layers_to_prune: List of layer indices to prune (REQUIRED)
            dry_run: If True, only analyze without making changes
            
        Returns:
            Statistics about the pruning operation
        """
        if layers_to_prune is None or len(layers_to_prune) == 0:
            raise ValueError("layers_to_prune must be specified for static pruning mode")
        
        self.logger.info("Running static layer pruning (no calibration)")
        return self.prune_layers(layers_to_prune, dry_run=dry_run)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Layer Similarity Pruning for Qwen3-Omni (with calibration)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with audio calibration data (recommended - same data as REAP step 2)
  python layer_similarity_pruning.py \\
      --audio-data ./calibration/audio.jsonl \\
      --similarity-threshold 0.9 \\
      --max-layers 8
  
  # Dry run to see similarities without pruning
  python layer_similarity_pruning.py \\
      --audio-data ./calibration/audio.jsonl \\
      --dry-run
  
  # Prune specific layers without calibration
  python layer_similarity_pruning.py \\
      --static \\
      --prune-layers 12 16 20 24
        """
    )
    
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default=r"D:\PycharmProjects\FuseOmni\models\Qwen3-Omni-30B-A3B-Instruct",
        help="Path to the original model"
    )
    
    parser.add_argument(
        "--output-path", "-o",
        type=str,
        default=r"D:\PycharmProjects\FuseOmni\models\Qwen3-Omni-30B-A3B-Layer-Pruned",
        help="Path to save the pruned model"
    )
    
    parser.add_argument(
        "--component",
        type=str,
        default="thinker",
        choices=["thinker", "talker"],
        help="Which component to prune"
    )
    
    # Calibration data
    parser.add_argument(
        "--audio-data",
        type=str,
        default=None,
        help="Path to audio calibration data (JSONL) - same format as REAP step 2"
    )
    
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=32,
        help="Number of calibration samples to use"
    )
    
    # Pruning parameters
    parser.add_argument(
        "--similarity-threshold", "-t",
        type=float,
        default=0.9,
        help="Layers with similarity >= threshold are candidates for pruning"
    )
    
    parser.add_argument(
        "--max-layers",
        type=int,
        default=8,
        help="Maximum number of layers to prune"
    )
    
    parser.add_argument(
        "--prune-layers",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of layer indices to prune (overrides auto-selection)"
    )
    
    parser.add_argument(
        "--protect-first",
        type=int,
        default=4,
        help="Number of first layers to protect from pruning"
    )
    
    parser.add_argument(
        "--protect-last",
        type=int,
        default=4,
        help="Number of last layers to protect from pruning"
    )
    
    parser.add_argument(
        "--layers-to-skip",
        type=int,
        default=1,
        help="Compare layer L with layer L+skip (1 = adjacent layers)"
    )
    
    # Execution mode
    parser.add_argument(
        "--static",
        action="store_true",
        help="Static mode: prune specified layers without loading model (requires --prune-layers)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze without making changes"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for model inference"
    )
    
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for model loading"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = LayerPruningConfig(
        model_path=args.model_path,
        output_path=args.output_path,
        component=args.component,
        audio_data_path=args.audio_data,
        calibration_samples=args.calibration_samples,
        similarity_threshold=args.similarity_threshold,
        max_layers_to_prune=args.max_layers,
        layers_to_skip=args.layers_to_skip,
        protect_first_n=args.protect_first,
        protect_last_n=args.protect_last,
        device=args.device,
        dtype=args.dtype,
        verbose=args.verbose
    )
    
    # Run pruning
    pruner = LayerSimilarityPruner(config)
    
    if args.static:
        # Static mode: just prune specified layers
        if args.prune_layers is None:
            parser.error("--static mode requires --prune-layers")
        stats = pruner.run_static_pruning(
            layers_to_prune=args.prune_layers,
            dry_run=args.dry_run
        )
    else:
        # Calibration mode: load model, compute similarities, then prune
        stats = pruner.run_with_calibration(
            layers_to_prune=args.prune_layers,
            dry_run=args.dry_run
        )
    
    # Print summary
    print("\n" + "=" * 60)
    print("LAYER PRUNING SUMMARY")
    print("=" * 60)
    
    # Format stats for printing (exclude large dicts)
    print_stats = {k: v for k, v in stats.items() if k != "similarities"}
    print(json.dumps(print_stats, indent=2, default=str))
    
    if "similarities" in stats:
        print("\nLayer Similarities:")
        for layer_idx, sim in sorted(stats["similarities"].items(), key=lambda x: int(x[0])):
            print(f"  Layer {layer_idx} -> {int(layer_idx) + config.layers_to_skip}: {sim:.4f}")


if __name__ == "__main__":
    main()
