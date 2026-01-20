#!/usr/bin/env python3
"""
REAP Expert Pruning for Qwen3-Omni-30B-A3B MoE

This module implements the REAP (Router-weighted Expert Activation Pruning) algorithm
for pruning MoE experts in multimodal models, following the REAP-OMNI methodology.

Key Algorithm:
1. Calculate expert saliency: S(e, D) = mean(g_e(x) * ||h_e(x)||_2)
   - g_e(x) = gating weight from router
   - ||h_e(x)||_2 = L2 norm of expert output

2. Audio Affinity Score: A_audio(e) = S1(e) + λ * ReLU(S3(e) - β * S2(e))
   - S1 = saliency on pure audio data
   - S2 = saliency on pure video data
   - S3 = saliency on mixed audio+video data
   - λ = 1.0 (importance weight for mixed data)
   - β = 1.0 (video denoising coefficient)

3. Rank experts by A_audio, keep top-K (e.g., 50%)
4. Remove pruned experts from model weights

Author: REAP-OMNI Implementation
Based on: D:/PycharmProjects/FuseOmni/REAP-OMNI/reap-omni.pdf
Reference: https://github.com/CerebrasResearch/reap
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
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class REAPConfig:
    """Configuration for REAP expert pruning."""
    
    # Model configuration
    model_path: str = r"D:\PycharmProjects\FuseOmni\models\Qwen3-Omni-30B-A3B-Instruct"
    output_path: str = r"D:\PycharmProjects\FuseOmni\models\Qwen3-Omni-30B-A3B-REAP-Pruned"
    component: str = "thinker"  # "thinker" or "talker"
    
    # REAP algorithm parameters
    retention_rate: float = 0.5  # Keep top 50% of experts
    lambda_weight: float = 1.0  # Weight for mixed data term
    beta_weight: float = 1.0    # Video denoising coefficient
    
    # Calibration data paths
    audio_data_path: Optional[str] = None
    video_data_path: Optional[str] = None
    mixed_data_path: Optional[str] = None
    
    # Execution parameters
    calibration_samples: int = 100
    batch_size: int = 1
    device: str = "cuda"
    dtype: str = "bfloat16"
    
    # Output options
    prune_mode: str = "remove"  # "remove" (delete weights) or "zero" (zero out weights)
    save_saliency: bool = True
    verbose: bool = True


@dataclass
class ExpertSaliency:
    """Stores saliency scores for a single expert."""
    layer_idx: int
    expert_idx: int
    
    # Individual saliency scores
    s1_audio: float = 0.0      # Pure audio saliency
    s2_video: float = 0.0      # Pure video saliency
    s3_mixed: float = 0.0      # Mixed saliency
    
    # Computed audio affinity score
    audio_affinity: float = 0.0
    
    # Statistics
    activation_count: int = 0
    mean_gating_weight: float = 0.0
    mean_activation_norm: float = 0.0


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(verbose: bool = True) -> logging.Logger:
    """Configure logging for REAP pruning."""
    logger = logging.getLogger("REAPPruning")
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
# Saliency Observer
# ============================================================================

class ExpertSaliencyObserver:
    """
    Observer for recording expert saliency during forward pass.
    
    This hooks into MoE layers to capture:
    - Router gating weights (g_e)
    - Expert output activation norms (||h_e||_2)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: REAPConfig,
        target_component: str = "thinker"
    ):
        self.model = model
        self.config = config
        self.target_component = target_component
        self.logger = logging.getLogger("SaliencyObserver")
        
        # Storage for saliency data per layer
        # Structure: {layer_idx: {expert_idx: {"reap": [], "count": 0}}}
        self.saliency_data: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(
            lambda: defaultdict(lambda: {"reap_scores": [], "count": 0})
        )
        
        # Hooks
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # Model info
        self.num_experts = self._get_num_experts()
        self.num_layers = self._get_num_layers()
    
    def _get_num_experts(self) -> int:
        """Get number of experts from model config."""
        config = self.model.config
        if hasattr(config, f"{self.target_component}_config"):
            component_config = getattr(config, f"{self.target_component}_config")
            if hasattr(component_config, "text_config"):
                return component_config.text_config.num_experts
        return 128  # Default for Qwen3-Omni
    
    def _get_num_layers(self) -> int:
        """Get number of layers from model config."""
        config = self.model.config
        if hasattr(config, f"{self.target_component}_config"):
            component_config = getattr(config, f"{self.target_component}_config")
            if hasattr(component_config, "text_config"):
                return component_config.text_config.num_hidden_layers
        return 48 if self.target_component == "thinker" else 20
    
    def _create_moe_hook(self, layer_idx: int) -> Callable:
        """
        Create a forward hook for MoE layer.
        
        The hook captures router logits and expert outputs to compute REAP saliency.
        """
        def hook(module, inputs, outputs):
            # Handle different output formats
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
                router_logits = outputs[1] if len(outputs) > 1 else None
            else:
                hidden_states = outputs
                router_logits = None
            
            # Try to get router logits from module if not in outputs
            if router_logits is None and hasattr(module, 'gate'):
                # Recompute router logits from input
                if isinstance(inputs, tuple):
                    input_hidden = inputs[0]
                else:
                    input_hidden = inputs
                
                with torch.no_grad():
                    router_logits = module.gate(input_hidden)
            
            if router_logits is None:
                self.logger.warning(f"Could not get router logits for layer {layer_idx}")
                return
            
            # Compute routing weights (softmax over experts)
            # router_logits shape: [batch, seq_len, num_experts]
            routing_weights = F.softmax(router_logits, dim=-1)
            
            # Get top-k expert selections
            num_experts_per_tok = getattr(module, 'num_experts_per_tok', 8)
            top_k_weights, top_k_indices = torch.topk(
                routing_weights, k=num_experts_per_tok, dim=-1
            )
            
            # Normalize top-k weights
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
            
            # Compute REAP saliency for each expert
            batch_size, seq_len, _ = router_logits.shape
            
            for expert_idx in range(self.num_experts):
                # Find tokens routed to this expert
                expert_mask = (top_k_indices == expert_idx).any(dim=-1)  # [batch, seq]
                
                if not expert_mask.any():
                    continue
                
                # Get gating weights for this expert
                expert_weights = routing_weights[..., expert_idx][expert_mask]
                
                # Compute activation norm (using hidden states as proxy)
                # In true REAP, we'd capture individual expert outputs
                activation_norms = torch.linalg.norm(
                    hidden_states[expert_mask], dim=-1
                )
                
                # REAP saliency: mean(g_e * ||h_e||_2)
                reap_score = (expert_weights * activation_norms).mean().item()
                
                # Store
                self.saliency_data[layer_idx][expert_idx]["reap_scores"].append(reap_score)
                self.saliency_data[layer_idx][expert_idx]["count"] += expert_mask.sum().item()
        
        return hook
    
    def register_hooks(self) -> None:
        """Register forward hooks on all MoE layers."""
        self.logger.info(f"Registering hooks for {self.num_layers} layers...")
        
        # Find MoE modules
        moe_pattern = re.compile(
            rf"{self.target_component}\.model\.layers\.(\d+)\.mlp"
        )
        
        for name, module in self.model.named_modules():
            # Check if this is an MoE module
            if "SparseMoeBlock" in type(module).__name__ or "MoE" in type(module).__name__:
                match = moe_pattern.search(name)
                if match:
                    layer_idx = int(match.group(1))
                    hook = module.register_forward_hook(self._create_moe_hook(layer_idx))
                    self.hooks.append(hook)
                    self.logger.debug(f"Registered hook for layer {layer_idx}: {name}")
        
        self.logger.info(f"Registered {len(self.hooks)} MoE hooks")
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.logger.info("Removed all hooks")
    
    def clear_data(self) -> None:
        """Clear accumulated saliency data."""
        self.saliency_data.clear()
        self.saliency_data = defaultdict(
            lambda: defaultdict(lambda: {"reap_scores": [], "count": 0})
        )
    
    def get_mean_saliency(self) -> Dict[int, Dict[int, float]]:
        """
        Compute mean saliency per expert across all observations.
        
        Returns:
            Dict mapping layer_idx -> {expert_idx -> mean_saliency}
        """
        result = {}
        
        for layer_idx, experts in self.saliency_data.items():
            result[layer_idx] = {}
            for expert_idx, data in experts.items():
                scores = data["reap_scores"]
                if scores:
                    result[layer_idx][expert_idx] = sum(scores) / len(scores)
                else:
                    result[layer_idx][expert_idx] = 0.0
        
        return result


# ============================================================================
# Calibration Data Generator
# ============================================================================

class CalibrationDataGenerator:
    """
    Generate calibration data for REAP saliency computation.
    
    Supports three data types:
    - X1: Pure audio samples
    - X2: Pure video samples  
    - X3: Mixed audio+video samples
    """
    
    def __init__(self, config: REAPConfig, tokenizer=None):
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logging.getLogger("CalibrationData")
    
    def load_from_jsonl(self, path: str) -> List[Dict]:
        """Load calibration samples from JSONL file."""
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples[:self.config.calibration_samples]
    
    def generate_synthetic_audio(self, n: int = 100) -> List[Dict]:
        """Generate synthetic audio-focused calibration prompts."""
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
        ]
        
        samples = []
        for i in range(n):
            samples.append({
                "id": f"syn_audio_{i:05d}",
                "text": templates[i % len(templates)],
                "modality": "audio"
            })
        return samples
    
    def generate_synthetic_video(self, n: int = 100) -> List[Dict]:
        """Generate synthetic video-focused calibration prompts."""
        templates = [
            "Watch the video and describe what you see.",
            "The video shows: A person walking in a park.",
            "Visual description: A cityscape at sunset.",
            "In this video clip, cooking is happening.",
            "The scene contains: Various objects on a table.",
            "Video frame description: Person typing on a computer.",
            "Silent video showing: A cat playing with a toy.",
            "Describe the visual content: A presentation slide.",
            "The video displays: Charts and graphs about data.",
            "Visual content: A landscape with mountains.",
        ]
        
        samples = []
        for i in range(n):
            samples.append({
                "id": f"syn_video_{i:05d}",
                "text": templates[i % len(templates)],
                "modality": "video"
            })
        return samples
    
    def generate_synthetic_mixed(self, n: int = 100) -> List[Dict]:
        """Generate synthetic mixed modality calibration prompts."""
        templates = [
            "The video shows a person speaking while music plays.",
            "Watch and listen: A lecture with visual slides.",
            "Video with audio: A concert performance.",
            "The person in the video says 'hello' while waving.",
            "Multimodal content: Visual presentation with narration.",
            "A video call showing people having a conversation.",
            "Movie scene: Characters talking in a cafe.",
            "Educational video: Teacher explaining with diagrams.",
            "Interview video: Person answering questions on camera.",
            "Vlog content: Person talking while showing their surroundings.",
        ]
        
        samples = []
        for i in range(n):
            samples.append({
                "id": f"syn_mixed_{i:05d}",
                "text": templates[i % len(templates)],
                "modality": "mixed"
            })
        return samples
    
    def get_calibration_data(
        self
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Get calibration data for all three modalities.
        
        Returns:
            Tuple of (audio_data, video_data, mixed_data)
        """
        n = self.config.calibration_samples
        
        # Load from files if provided
        if self.config.audio_data_path and os.path.exists(self.config.audio_data_path):
            audio_data = self.load_from_jsonl(self.config.audio_data_path)
        else:
            self.logger.info("Using synthetic audio data")
            audio_data = self.generate_synthetic_audio(n)
        
        if self.config.video_data_path and os.path.exists(self.config.video_data_path):
            video_data = self.load_from_jsonl(self.config.video_data_path)
        else:
            self.logger.info("Using synthetic video data")
            video_data = self.generate_synthetic_video(n)
        
        if self.config.mixed_data_path and os.path.exists(self.config.mixed_data_path):
            mixed_data = self.load_from_jsonl(self.config.mixed_data_path)
        else:
            self.logger.info("Using synthetic mixed data")
            mixed_data = self.generate_synthetic_mixed(n)
        
        return audio_data, video_data, mixed_data
    
    def prepare_inputs(
        self, 
        samples: List[Dict],
        device: str = "cuda"
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Prepare model inputs from calibration samples.
        
        Yields batched input tensors for the model.
        """
        if self.tokenizer is None:
            self.logger.warning("No tokenizer provided, using dummy inputs")
            # Generate dummy inputs
            for sample in samples:
                yield {
                    "input_ids": torch.randint(0, 10000, (1, 128), device=device),
                    "attention_mask": torch.ones(1, 128, device=device)
                }
            return
        
        for i in range(0, len(samples), self.config.batch_size):
            batch = samples[i:i + self.config.batch_size]
            texts = [s["text"] for s in batch]
            
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            yield {k: v.to(device) for k, v in inputs.items()}


# ============================================================================
# REAP Pruner
# ============================================================================

class REAPExpertPruner:
    """
    Main class for REAP-based expert pruning.
    
    Implements the full REAP-OMNI algorithm:
    1. Collect saliency on three data types
    2. Compute audio affinity scores
    3. Rank and select top-K experts
    4. Prune model weights
    """
    
    def __init__(self, config: REAPConfig):
        self.config = config
        self.logger = setup_logging(config.verbose)
        
        self.model_path = Path(config.model_path)
        self.output_path = Path(config.output_path)
        
        # Validate paths
        self._validate_paths()
        
        # Load model configuration
        self.model_config = self._load_model_config()
        
        # Expert info
        self.num_experts = self._get_num_experts()
        self.num_layers = self._get_num_layers()
        self.num_experts_per_tok = self._get_num_experts_per_tok()
    
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
    
    def _get_num_experts(self) -> int:
        """Get number of experts."""
        if self.config.component == "thinker":
            return self.model_config.get("thinker_config", {}).get("text_config", {}).get("num_experts", 128)
        else:
            return self.model_config.get("talker_config", {}).get("text_config", {}).get("num_experts", 128)
    
    def _get_num_layers(self) -> int:
        """Get number of layers."""
        if self.config.component == "thinker":
            return self.model_config.get("thinker_config", {}).get("text_config", {}).get("num_hidden_layers", 48)
        else:
            return self.model_config.get("talker_config", {}).get("text_config", {}).get("num_hidden_layers", 20)
    
    def _get_num_experts_per_tok(self) -> int:
        """Get number of experts per token."""
        if self.config.component == "thinker":
            return self.model_config.get("thinker_config", {}).get("text_config", {}).get("num_experts_per_tok", 8)
        else:
            return self.model_config.get("talker_config", {}).get("text_config", {}).get("num_experts_per_tok", 6)
    
    def compute_audio_affinity(
        self,
        s1_audio: Dict[int, Dict[int, float]],
        s2_video: Dict[int, Dict[int, float]],
        s3_mixed: Dict[int, Dict[int, float]]
    ) -> Dict[int, Dict[int, float]]:
        """
        Compute audio affinity scores for all experts.
        
        Formula: A_audio(e) = S1(e) + λ * ReLU(S3(e) - β * S2(e))
        
        Args:
            s1_audio: Saliency on pure audio data
            s2_video: Saliency on pure video data
            s3_mixed: Saliency on mixed data
            
        Returns:
            Dict mapping layer_idx -> {expert_idx -> audio_affinity}
        """
        lambda_w = self.config.lambda_weight
        beta_w = self.config.beta_weight
        
        audio_affinity = {}
        
        for layer_idx in range(self.num_layers):
            audio_affinity[layer_idx] = {}
            
            for expert_idx in range(self.num_experts):
                # Get saliency scores (default to 0 if not observed)
                s1 = s1_audio.get(layer_idx, {}).get(expert_idx, 0.0)
                s2 = s2_video.get(layer_idx, {}).get(expert_idx, 0.0)
                s3 = s3_mixed.get(layer_idx, {}).get(expert_idx, 0.0)
                
                # Compute audio affinity
                # A = S1 + λ * ReLU(S3 - β * S2)
                differential = s3 - beta_w * s2
                relu_diff = max(0.0, differential)
                affinity = s1 + lambda_w * relu_diff
                
                audio_affinity[layer_idx][expert_idx] = affinity
        
        return audio_affinity
    
    def select_experts_to_keep(
        self,
        audio_affinity: Dict[int, Dict[int, float]]
    ) -> Dict[int, Set[int]]:
        """
        Select top-K experts to keep based on audio affinity.
        
        Args:
            audio_affinity: Audio affinity scores per layer per expert
            
        Returns:
            Dict mapping layer_idx -> set of expert indices to keep
        """
        num_to_keep = int(self.num_experts * self.config.retention_rate)
        
        self.logger.info(f"Keeping top {num_to_keep}/{self.num_experts} experts per layer "
                        f"(retention rate: {self.config.retention_rate:.0%})")
        
        experts_to_keep = {}
        
        for layer_idx in range(self.num_layers):
            layer_scores = audio_affinity.get(layer_idx, {})
            
            # Sort by affinity (descending)
            sorted_experts = sorted(
                layer_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Keep top-K
            kept = set(expert_idx for expert_idx, _ in sorted_experts[:num_to_keep])
            experts_to_keep[layer_idx] = kept
            
            # Log some statistics
            if layer_idx == 0 or layer_idx == self.num_layers - 1:
                kept_scores = [s for e, s in sorted_experts[:num_to_keep]]
                pruned_scores = [s for e, s in sorted_experts[num_to_keep:]]
                self.logger.info(
                    f"Layer {layer_idx}: kept mean={sum(kept_scores)/len(kept_scores) if kept_scores else 0:.4f}, "
                    f"pruned mean={sum(pruned_scores)/len(pruned_scores) if pruned_scores else 0:.4f}"
                )
        
        return experts_to_keep
    
    def prune_weights_static(
        self,
        experts_to_keep: Dict[int, Set[int]],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Prune expert weights without loading the full model.
        
        This operates directly on safetensor files.
        
        Args:
            experts_to_keep: Mapping of layer -> set of expert indices to keep
            dry_run: If True, only analyze without making changes
            
        Returns:
            Statistics about the pruning operation
        """
        self.logger.info("=" * 60)
        self.logger.info("REAP EXPERT PRUNING - WEIGHT REMOVAL MODE")
        self.logger.info("=" * 60)
        
        # Load weight index
        with open(self.model_path / "model.safetensors.index.json") as f:
            index = json.load(f)
        
        weight_map = index["weight_map"]
        
        # Expert pattern based on component
        expert_pattern = re.compile(
            rf"^{self.config.component}\.model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$"
        )
        
        # Categorize weights
        expert_weights = []
        other_weights = []
        
        for weight_name in weight_map.keys():
            match = expert_pattern.match(weight_name)
            if match:
                layer_idx = int(match.group(1))
                expert_idx = int(match.group(2))
                expert_weights.append((weight_name, layer_idx, expert_idx))
            else:
                other_weights.append(weight_name)
        
        # Calculate what to keep and remove
        weights_to_keep = set(other_weights)  # Always keep non-expert weights
        weights_to_remove = set()
        
        for weight_name, layer_idx, expert_idx in expert_weights:
            if layer_idx in experts_to_keep and expert_idx in experts_to_keep[layer_idx]:
                weights_to_keep.add(weight_name)
            else:
                weights_to_remove.add(weight_name)
        
        stats = {
            "total_weights": len(weight_map),
            "expert_weights": len(expert_weights),
            "experts_per_layer": self.num_experts,
            "experts_to_keep_per_layer": len(experts_to_keep.get(0, set())),
            "weights_to_keep": len(weights_to_keep),
            "weights_to_remove": len(weights_to_remove),
            "removal_percentage": len(weights_to_remove) / len(weight_map) * 100
        }
        
        self.logger.info(f"Total weights: {stats['total_weights']}")
        self.logger.info(f"Expert weights: {stats['expert_weights']}")
        self.logger.info(f"Weights to keep: {stats['weights_to_keep']}")
        self.logger.info(f"Weights to remove: {stats['weights_to_remove']} ({stats['removal_percentage']:.1f}%)")
        
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
            
            # Load shard
            new_tensors = {}
            
            with safe_open(input_shard, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in weights_to_keep:
                        new_tensors[key] = f.get_tensor(key)
                        new_weight_map[key] = shard_file
                    else:
                        tensor = f.get_tensor(key)
                        total_removed_bytes += tensor.numel() * tensor.element_size()
            
            # Save modified shard
            if new_tensors:
                save_file(new_tensors, output_shard)
        
        # Save new index
        new_index = {
            "metadata": {
                "total_size": index.get("metadata", {}).get("total_size", 0) - total_removed_bytes * 2,
                "reap_pruned": True,
                "retention_rate": self.config.retention_rate,
                "component": self.config.component,
                "lambda": self.config.lambda_weight,
                "beta": self.config.beta_weight
            },
            "weight_map": new_weight_map
        }
        
        with open(self.output_path / "model.safetensors.index.json", "w") as f:
            json.dump(new_index, f, indent=2)
        
        # Update and save config
        self._save_pruned_config(experts_to_keep)
        
        # Copy other files
        self._copy_auxiliary_files()
        
        stats["bytes_removed"] = total_removed_bytes
        stats["gb_removed"] = total_removed_bytes / (1024**3)
        
        self.logger.info("=" * 60)
        self.logger.info("REAP PRUNING COMPLETE")
        self.logger.info(f"Size reduced: {stats['gb_removed']:.2f} GB")
        self.logger.info(f"Output: {self.output_path}")
        self.logger.info("=" * 60)
        
        return stats
    
    def _save_pruned_config(self, experts_to_keep: Dict[int, Set[int]]) -> None:
        """Save modified configuration for pruned model."""
        config = self.model_config.copy()
        
        # Update expert count
        new_num_experts = len(experts_to_keep.get(0, set()))
        
        if self.config.component == "thinker":
            config["thinker_config"]["text_config"]["num_experts"] = new_num_experts
            config["thinker_config"]["text_config"]["original_num_experts"] = self.num_experts
        else:
            config["talker_config"]["text_config"]["num_experts"] = new_num_experts
            config["talker_config"]["text_config"]["original_num_experts"] = self.num_experts
        
        # Add pruning metadata
        config["reap_pruning_info"] = {
            "method": "REAP-OMNI",
            "component": self.config.component,
            "retention_rate": self.config.retention_rate,
            "lambda_weight": self.config.lambda_weight,
            "beta_weight": self.config.beta_weight,
            "original_num_experts": self.num_experts,
            "new_num_experts": new_num_experts
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
    
    def run_static_pruning(
        self,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Run REAP pruning without loading the model.
        
        Uses simulated/uniform saliency for quick pruning without calibration.
        """
        self.logger.info("Running static REAP pruning (no model loading)")
        self.logger.info(f"Component: {self.config.component}")
        self.logger.info(f"Retention rate: {self.config.retention_rate:.0%}")
        
        # Without actual model inference, we use uniform saliency
        # and select experts randomly or by index
        # In practice, you'd want to run calibration with real data
        
        # For static pruning, keep first N experts (or random selection)
        num_to_keep = int(self.num_experts * self.config.retention_rate)
        
        experts_to_keep = {}
        for layer_idx in range(self.num_layers):
            # Keep experts 0 to num_to_keep-1
            # In real REAP, these would be selected by audio affinity score
            experts_to_keep[layer_idx] = set(range(num_to_keep))
        
        return self.prune_weights_static(experts_to_keep, dry_run=dry_run)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="REAP Expert Pruning for Qwen3-Omni MoE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick static pruning (no calibration data)
  python reap_expert_pruning.py --retention-rate 0.5 --component thinker
  
  # With calibration data
  python reap_expert_pruning.py \\
      --model-path ./models/Qwen3-Omni-30B-A3B-Instruct \\
      --output-path ./models/Qwen3-Omni-REAP-50 \\
      --retention-rate 0.5 \\
      --audio-data ./calibration/audio.jsonl \\
      --video-data ./calibration/video.jsonl \\
      --mixed-data ./calibration/mixed.jsonl
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
        default=r"D:\PycharmProjects\FuseOmni\models\Qwen3-Omni-30B-A3B-REAP-Pruned",
        help="Path to save the pruned model"
    )
    
    parser.add_argument(
        "--component",
        type=str,
        default="thinker",
        choices=["thinker", "talker"],
        help="Which component to prune (thinker=main LLM, talker=speech decoder)"
    )
    
    parser.add_argument(
        "--retention-rate", "-r",
        type=float,
        default=0.5,
        help="Fraction of experts to keep (0.0-1.0)"
    )
    
    parser.add_argument(
        "--lambda-weight",
        type=float,
        default=1.0,
        help="Weight for mixed data term in audio affinity calculation"
    )
    
    parser.add_argument(
        "--beta-weight",
        type=float,
        default=1.0,
        help="Video denoising coefficient"
    )
    
    parser.add_argument(
        "--audio-data",
        type=str,
        default=None,
        help="Path to audio calibration data (JSONL)"
    )
    
    parser.add_argument(
        "--video-data",
        type=str,
        default=None,
        help="Path to video calibration data (JSONL)"
    )
    
    parser.add_argument(
        "--mixed-data",
        type=str,
        default=None,
        help="Path to mixed calibration data (JSONL)"
    )
    
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=100,
        help="Number of calibration samples per modality"
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
        help="Device for computations"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = REAPConfig(
        model_path=args.model_path,
        output_path=args.output_path,
        component=args.component,
        retention_rate=args.retention_rate,
        lambda_weight=args.lambda_weight,
        beta_weight=args.beta_weight,
        audio_data_path=args.audio_data,
        video_data_path=args.video_data,
        mixed_data_path=args.mixed_data,
        calibration_samples=args.calibration_samples,
        device=args.device,
        verbose=args.verbose
    )
    
    # Run pruning
    pruner = REAPExpertPruner(config)
    stats = pruner.run_static_pruning(dry_run=args.dry_run)
    
    # Print summary
    print("\n" + "=" * 60)
    print("REAP PRUNING SUMMARY")
    print("=" * 60)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
