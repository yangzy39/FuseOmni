#!/usr/bin/env python3
"""
Vision Modality Stripping for Qwen3-Omni-30B-A3B

This module implements complete vision modality removal from the Qwen3-Omni model,
following the REAP-OMNI methodology:
1. Remove Vision Encoder weights
2. Remove Vision Projector (visual.merger) weights
3. Update config.json to remove vision configuration
4. Regenerate model.safetensors.index.json

Author: REAP-OMNI Implementation
Based on: D:/PycharmProjects/FuseOmni/REAP-OMNI/reap-omni.pdf
"""

import argparse
import json
import logging
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

# Vision-related weight patterns for Qwen3-Omni
VISION_WEIGHT_PATTERNS = [
    # Vision encoder weights
    r"^thinker\.visual\..*",
    # Vision embeddings
    r"^thinker\.model\.embed_tokens\..*image.*",
    r"^thinker\.model\.embed_tokens\..*vision.*",
    r"^thinker\.model\.embed_tokens\..*visual.*",
    # Any visual/vision prefix in thinker
    r"^visual\..*",
]

# Vision-related token IDs to potentially clean up
VISION_TOKEN_IDS = {
    "vision_start_token_id": 151652,
    "vision_end_token_id": 151653,
    "image_token_id": 151655,
    "video_token_id": 151656,
}


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(verbose: bool = True) -> logging.Logger:
    """Configure logging for the vision stripping process."""
    logger = logging.getLogger("VisionStrip")
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
# Vision Weight Identifier
# ============================================================================

class VisionWeightIdentifier:
    """Identifies vision-related weights in the model."""
    
    def __init__(self, patterns: List[str] = None):
        self.patterns = [re.compile(p) for p in (patterns or VISION_WEIGHT_PATTERNS)]
        self.logger = logging.getLogger("VisionWeightIdentifier")
    
    def is_vision_weight(self, weight_name: str) -> bool:
        """Check if a weight name belongs to vision components."""
        for pattern in self.patterns:
            if pattern.match(weight_name):
                return True
        return False
    
    def categorize_weights(
        self, 
        weight_names: List[str]
    ) -> Tuple[Set[str], Set[str]]:
        """
        Categorize weights into vision and non-vision sets.
        
        Returns:
            Tuple of (vision_weights, other_weights)
        """
        vision_weights = set()
        other_weights = set()
        
        for name in weight_names:
            if self.is_vision_weight(name):
                vision_weights.add(name)
            else:
                other_weights.add(name)
        
        return vision_weights, other_weights


# ============================================================================
# Config Modifier
# ============================================================================

class ConfigModifier:
    """Modifies model configuration to remove vision components."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.logger = logging.getLogger("ConfigModifier")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.original_config = json.dumps(self.config, indent=2)
    
    def remove_vision_config(self) -> Dict:
        """
        Remove vision-related configuration from the model config.
        
        This includes:
        - thinker_config.vision_config
        - Vision token IDs
        - Vision-related fields in various configs
        """
        modified = False
        
        # Remove vision_config from thinker_config
        if "thinker_config" in self.config:
            thinker = self.config["thinker_config"]
            
            if "vision_config" in thinker:
                self.logger.info("Removing thinker_config.vision_config")
                del thinker["vision_config"]
                modified = True
            
            # Remove vision token IDs
            vision_tokens_to_remove = [
                "vision_start_token_id",
                "vision_end_token_id", 
                "image_token_id",
                "video_token_id",
            ]
            
            for token_key in vision_tokens_to_remove:
                if token_key in thinker:
                    self.logger.info(f"Removing thinker_config.{token_key}")
                    del thinker[token_key]
                    modified = True
        
        # Remove from talker_config if present
        if "talker_config" in self.config:
            talker = self.config["talker_config"]
            
            vision_tokens_to_remove = [
                "vision_start_token_id",
                "image_token_id",
                "video_token_id",
                "spatial_merge_size",
            ]
            
            for token_key in vision_tokens_to_remove:
                if token_key in talker:
                    self.logger.info(f"Removing talker_config.{token_key}")
                    del talker[token_key]
                    modified = True
        
        # Add metadata about vision stripping
        self.config["vision_stripped"] = True
        self.config["vision_strip_info"] = {
            "method": "REAP-OMNI vision modality stripping",
            "components_removed": [
                "vision_encoder",
                "vision_projector (visual.merger)",
                "vision_config",
                "vision_token_ids"
            ]
        }
        
        return self.config
    
    def save(self, output_path: Path) -> None:
        """Save the modified configuration."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved modified config to {output_path}")


# ============================================================================
# Weight Stripper
# ============================================================================

class VisionWeightStripper:
    """
    Main class for stripping vision weights from Qwen3-Omni model.
    
    This class handles:
    1. Loading the model weight index
    2. Identifying vision-related weights
    3. Creating new safetensors files without vision weights
    4. Updating the weight index
    """
    
    def __init__(
        self,
        model_path: str,
        output_path: str,
        device: str = "cuda",
        verbose: bool = True
    ):
        self.model_path = Path(model_path)
        self.output_path = Path(output_path)
        self.device = device
        self.logger = setup_logging(verbose)
        
        self.identifier = VisionWeightIdentifier()
        
        # Validate paths
        self._validate_paths()
        
        # Load weight index
        self.weight_index = self._load_weight_index()
    
    def _validate_paths(self) -> None:
        """Validate input and output paths."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {self.model_path}")
        
        index_path = self.model_path / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Weight index not found: {index_path}")
        
        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
    
    def _load_weight_index(self) -> Dict:
        """Load the safetensors weight index."""
        index_path = self.model_path / "model.safetensors.index.json"
        with open(index_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def analyze_weights(self) -> Dict[str, any]:
        """
        Analyze the model weights and identify vision components.
        
        Returns statistics about vision vs non-vision weights.
        """
        weight_map = self.weight_index.get("weight_map", {})
        
        vision_weights, other_weights = self.identifier.categorize_weights(
            weight_map.keys()
        )
        
        # Group vision weights by component
        vision_components = defaultdict(list)
        for weight_name in vision_weights:
            # Extract component name (e.g., "thinker.visual.patch_embed")
            parts = weight_name.split(".")
            if len(parts) >= 3:
                component = ".".join(parts[:3])
            else:
                component = weight_name
            vision_components[component].append(weight_name)
        
        # Calculate sizes per shard
        shard_vision_weights = defaultdict(list)
        shard_other_weights = defaultdict(list)
        
        for weight_name, shard_file in weight_map.items():
            if weight_name in vision_weights:
                shard_vision_weights[shard_file].append(weight_name)
            else:
                shard_other_weights[shard_file].append(weight_name)
        
        stats = {
            "total_weights": len(weight_map),
            "vision_weights": len(vision_weights),
            "other_weights": len(other_weights),
            "vision_percentage": len(vision_weights) / len(weight_map) * 100,
            "vision_components": {k: len(v) for k, v in vision_components.items()},
            "shards_with_vision": len(shard_vision_weights),
            "shards_affected": list(shard_vision_weights.keys()),
        }
        
        return stats, vision_weights, other_weights
    
    def strip_vision_weights(
        self,
        dry_run: bool = False,
        copy_unaffected: bool = True
    ) -> Dict[str, any]:
        """
        Remove vision weights from the model.
        
        Args:
            dry_run: If True, only analyze without making changes
            copy_unaffected: If True, copy shards that don't contain vision weights
            
        Returns:
            Dictionary with operation statistics
        """
        self.logger.info("=" * 60)
        self.logger.info("VISION MODALITY STRIPPING")
        self.logger.info("=" * 60)
        self.logger.info(f"Input:  {self.model_path}")
        self.logger.info(f"Output: {self.output_path}")
        self.logger.info("=" * 60)
        
        # Analyze weights
        stats, vision_weights, other_weights = self.analyze_weights()
        
        self.logger.info(f"Total weights: {stats['total_weights']}")
        self.logger.info(f"Vision weights to remove: {stats['vision_weights']} ({stats['vision_percentage']:.2f}%)")
        self.logger.info(f"Weights to keep: {stats['other_weights']}")
        
        if stats['vision_weights'] == 0:
            self.logger.warning("No vision weights found! Model may already be stripped.")
            return stats
        
        # Log vision components
        self.logger.info("\nVision components to remove:")
        for component, count in stats['vision_components'].items():
            self.logger.info(f"  - {component}: {count} weights")
        
        if dry_run:
            self.logger.info("\n[DRY RUN] No changes made.")
            return stats
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each shard
        weight_map = self.weight_index.get("weight_map", {})
        shards_to_process = set(weight_map.values())
        
        new_weight_map = {}
        total_removed_bytes = 0
        processed_shards = set()
        
        for shard_file in tqdm(shards_to_process, desc="Processing shards"):
            input_shard = self.model_path / shard_file
            output_shard = self.output_path / shard_file
            
            # Get weights in this shard
            shard_weights = [k for k, v in weight_map.items() if v == shard_file]
            
            # Separate vision and non-vision weights
            shard_vision = [w for w in shard_weights if w in vision_weights]
            shard_keep = [w for w in shard_weights if w in other_weights]
            
            if not shard_vision:
                # No vision weights in this shard - copy directly
                if copy_unaffected:
                    shutil.copy2(input_shard, output_shard)
                    for weight in shard_keep:
                        new_weight_map[weight] = shard_file
                continue
            
            # Load shard and filter out vision weights
            with safe_open(input_shard, framework="pt", device="cpu") as f:
                new_tensors = {}
                
                for key in f.keys():
                    if key in other_weights:
                        new_tensors[key] = f.get_tensor(key)
                        new_weight_map[key] = shard_file
                    else:
                        # Track removed size
                        tensor = f.get_tensor(key)
                        total_removed_bytes += tensor.numel() * tensor.element_size()
            
            # Save new shard (if it has any tensors left)
            if new_tensors:
                save_file(new_tensors, output_shard)
                processed_shards.add(shard_file)
            else:
                self.logger.info(f"Shard {shard_file} is now empty (all vision weights)")
        
        # Save new weight index
        new_metadata = self.weight_index.get("metadata", {}).copy()
        new_metadata["vision_stripped"] = True
        new_metadata["vision_weights_removed"] = stats['vision_weights']
        new_metadata["bytes_removed"] = total_removed_bytes
        
        new_index = {
            "metadata": new_metadata,
            "weight_map": new_weight_map
        }
        
        index_output = self.output_path / "model.safetensors.index.json"
        with open(index_output, 'w', encoding='utf-8') as f:
            json.dump(new_index, f, indent=2)
        
        # Modify and save config
        config_modifier = ConfigModifier(self.model_path / "config.json")
        config_modifier.remove_vision_config()
        config_modifier.save(self.output_path / "config.json")
        
        # Copy other necessary files
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
        
        # Final statistics
        stats["bytes_removed"] = total_removed_bytes
        stats["mb_removed"] = total_removed_bytes / (1024 * 1024)
        stats["gb_removed"] = total_removed_bytes / (1024 * 1024 * 1024)
        stats["shards_processed"] = len(processed_shards)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("VISION STRIPPING COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Weights removed: {stats['vision_weights']}")
        self.logger.info(f"Size reduced: {stats['gb_removed']:.2f} GB")
        self.logger.info(f"Output saved to: {self.output_path}")
        self.logger.info("=" * 60)
        
        return stats


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Strip vision modality from Qwen3-Omni model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze vision weights without making changes
  python vision_strip.py --model-path ./models/Qwen3-Omni-30B-A3B-Instruct --dry-run
  
  # Strip vision weights and save to new directory
  python vision_strip.py \\
      --model-path ./models/Qwen3-Omni-30B-A3B-Instruct \\
      --output-path ./models/Qwen3-Omni-30B-A3B-Audio-Only
        """
    )
    
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        default=r"D:\PycharmProjects\FuseOmni\models\Qwen3-Omni-30B-A3B-Instruct",
        help="Path to the original Qwen3-Omni model"
    )
    
    parser.add_argument(
        "--output-path", "-o",
        type=str,
        default=r"D:\PycharmProjects\FuseOmni\models\Qwen3-Omni-30B-A3B-Vision-Stripped",
        help="Path to save the vision-stripped model"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze weights without making changes"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for tensor operations"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=True,
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--no-copy-unaffected",
        action="store_true",
        help="Don't copy shards that have no vision weights (saves time but incomplete model)"
    )
    
    args = parser.parse_args()
    
    # Run vision stripping
    stripper = VisionWeightStripper(
        model_path=args.model_path,
        output_path=args.output_path,
        device=args.device,
        verbose=args.verbose
    )
    
    stats = stripper.strip_vision_weights(
        dry_run=args.dry_run,
        copy_unaffected=not args.no_copy_unaffected
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    main()
