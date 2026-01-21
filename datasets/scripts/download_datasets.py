#!/usr/bin/env python3
"""
Audio Dataset Download Script for Speech SFT.

Downloads audio datasets from HuggingFace for SFT training.
Parses DATASETS_CATALOG.md to extract HuggingFace dataset IDs.

Features:
- Parse catalog markdown to extract HF dataset IDs
- Download via huggingface_hub or datasets library
- Support HF_ENDPOINT mirror and HF_TOKEN authentication
- Cross-platform path handling (Windows/Linux)

Usage:
    # Set environment variables first
    export HF_ENDPOINT=https://hf-mirror.com
    export HF_TOKEN="your_token_here"
    
    # Download all audio datasets
    python download_datasets.py --output ./data --modality audio
    
    # Download specific datasets
    python download_datasets.py --output ./data --datasets librispeech common_voice
    
    # List available datasets from catalog
    python download_datasets.py --list
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a dataset from the catalog."""
    name: str
    hf_id: str
    modality: str  # audio, video, mixed
    description: str = ""
    license: str = ""
    data_scale: str = ""
    
    
# Pre-defined dataset registry with HF paths and processing info
# Extracted from DATASETS_CATALOG.md
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    # === Audio-only (S1) ===
    "librispeech": {
        "hf_path": "openslr/librispeech_asr",
        "hf_config": "clean",
        "split": "train.clean.100",
        "audio_key": "audio",
        "text_key": "text",
        "modality": "audio",
        "description": "English audiobook ASR dataset",
        "extra_keys": ["speaker_id", "chapter_id"],
    },
    "common_voice": {
        "hf_path": "mozilla-foundation/common_voice_15_0",
        "hf_config": "en",
        "split": "train",
        "audio_key": "audio",
        "text_key": "sentence",
        "modality": "audio",
        "description": "Multilingual crowdsourced ASR",
        "requires_auth": True,
        "extra_keys": ["age", "gender", "accent"],
    },
    "gigaspeech": {
        "hf_path": "speechcolab/gigaspeech",
        "hf_config": "xs",
        "split": "train",
        "audio_key": "audio",
        "text_key": "text",
        "modality": "audio",
        "description": "Large-scale English ASR",
        "requires_auth": True,
    },
    "voxpopuli": {
        "hf_path": "facebook/voxpopuli",
        "hf_config": "en",
        "split": "train",
        "audio_key": "audio",
        "text_key": "normalized_text",
        "modality": "audio",
        "description": "European Parliament multilingual corpus",
    },
    "wenetspeech": {
        "hf_path": "wenet-e2e/wenetspeech",
        "hf_config": None,
        "split": "train_s",
        "audio_key": "audio",
        "text_key": "text",
        "modality": "audio",
        "description": "Large-scale Chinese ASR",
    },
    "aishell1": {
        "hf_path": "AISHELL/AISHELL-1",
        "hf_config": None,
        "split": "train",
        "audio_key": "audio",
        "text_key": "text",
        "modality": "audio",
        "description": "Chinese Mandarin ASR",
    },
    "covost2": {
        "hf_path": "facebook/covost2",
        "hf_config": "en_zh-CN",
        "split": "train",
        "audio_key": "audio",
        "text_key": "sentence",
        "modality": "audio",
        "description": "Multilingual speech translation",
        "requires_auth": True,
        "extra_keys": ["translation"],
    },
    "libritts": {
        "hf_path": "openslr/libritts",
        "hf_config": "clean",
        "split": "train.clean.100",
        "audio_key": "audio",
        "text_key": "text_normalized",
        "modality": "audio",
        "description": "English TTS dataset",
    },
    "wavcaps": {
        "hf_path": "cvssp/WavCaps",
        "hf_config": None,
        "split": "train",
        "audio_key": "audio",
        "text_key": "caption",
        "modality": "audio",
        "description": "Audio captioning dataset",
    },
    
    # === Video-only (S2) ===
    "youcook2": {
        "hf_path": "merve/YouCook2",
        "hf_config": None,
        "split": "train",
        "video_key": "video",
        "text_key": "caption",
        "modality": "video",
        "description": "Cooking instruction videos",
    },
    "longvideobench": {
        "hf_path": "longvideobench/LongVideoBench",
        "hf_config": None,
        "split": "test",
        "video_key": "video",
        "text_key": "question",
        "modality": "video",
        "description": "Long video understanding",
    },
    "videochat2_it": {
        "hf_path": "OpenGVLab/VideoChat2-IT",
        "hf_config": None,
        "split": "train",
        "video_key": "video",
        "text_key": "conversations",
        "modality": "video",
        "description": "Video chat instruction tuning",
    },
    
    # === Mixed (S3) ===
    "ugc_videocap": {
        "hf_path": "openinterx/UGC-VideoCap",
        "hf_config": None,
        "split": "train",
        "audio_key": "audio",
        "video_key": "video",
        "text_key": "caption",
        "modality": "mixed",
        "description": "Short video multimodal captioning",
    },
}


def normalize_path(path: str | Path) -> Path:
    """Normalize path for cross-platform compatibility."""
    path = Path(path)
    # Resolve to absolute path
    path = path.resolve()
    return path


def parse_catalog_markdown(catalog_path: Path) -> List[DatasetInfo]:
    """
    Parse DATASETS_CATALOG.md to extract dataset information.
    
    Args:
        catalog_path: Path to the catalog markdown file
        
    Returns:
        List of DatasetInfo objects
    """
    if not catalog_path.exists():
        logger.warning(f"Catalog file not found: {catalog_path}")
        return []
    
    datasets = []
    current_modality = "audio"
    
    with open(catalog_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Detect modality sections
    lines = content.split("\n")
    
    for line in lines:
        # Detect section headers
        if "Audio-only" in line or "纯音频" in line:
            current_modality = "audio"
        elif "Video-only" in line or "纯视频" in line:
            current_modality = "video"
        elif "Mixed" in line or "音视频混合" in line:
            current_modality = "mixed"
        
        # Parse table rows with HuggingFace links
        # Format: | Name | Source | Description | ... | [hf_id](url) |
        if "|" in line and "huggingface.co/datasets/" in line:
            # Extract HuggingFace dataset ID from URL
            hf_match = re.search(r'huggingface\.co/datasets/([^\s\)]+)', line)
            if hf_match:
                hf_id = hf_match.group(1).rstrip(")")
                
                # Extract dataset name from first column
                parts = [p.strip() for p in line.split("|") if p.strip()]
                if parts:
                    name = re.sub(r'\*+', '', parts[0]).strip()
                    description = parts[2] if len(parts) > 2 else ""
                    data_scale = parts[5] if len(parts) > 5 else ""
                    license_info = parts[7] if len(parts) > 7 else ""
                    
                    datasets.append(DatasetInfo(
                        name=name,
                        hf_id=hf_id,
                        modality=current_modality,
                        description=description,
                        license=license_info,
                        data_scale=data_scale,
                    ))
    
    return datasets


def save_audio(audio_array: np.ndarray, sample_rate: int, output_path: Path) -> bool:
    """Save audio array to WAV file."""
    try:
        import soundfile as sf
        # Ensure float32 format and normalize if needed
        audio = np.asarray(audio_array, dtype=np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))
        sf.write(str(output_path), audio, sample_rate, format="WAV")
        return True
    except ImportError:
        try:
            from scipy.io import wavfile
            audio = np.asarray(audio_array, dtype=np.float32)
            if audio.max() <= 1.0 and audio.min() >= -1.0:
                audio = (audio * 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)
            wavfile.write(str(output_path), sample_rate, audio)
            return True
        except ImportError:
            logger.error("Neither soundfile nor scipy is available. Install with: pip install soundfile")
            return False


def download_with_datasets_library(
    name: str,
    config: Dict[str, Any],
    output_dir: Path,
    samples: int = 100,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Download dataset using the datasets library.
    
    Args:
        name: Dataset name
        config: Dataset configuration from registry
        output_dir: Output directory
        samples: Number of samples to download
        token: HuggingFace token
        
    Returns:
        Dictionary with download statistics
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return {"name": name, "status": "error", "error": "datasets library not installed"}
    
    dataset_dir = output_dir / name
    audio_dir = dataset_dir / "audio"
    manifest_path = dataset_dir / "manifest.json"
    
    # Check if already exists
    if manifest_path.exists():
        logger.info(f"Dataset {name} already exists. Skipping.")
        return {"name": name, "status": "skipped", "reason": "Already exists"}
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    
    # Load dataset
    logger.info(f"Downloading {name} from {config['hf_path']}...")
    
    try:
        load_kwargs = {
            "path": config["hf_path"],
            "split": config.get("split", "train"),
            "trust_remote_code": True,
            "streaming": True,  # Use streaming to avoid downloading entire dataset
        }
        if config.get("hf_config"):
            load_kwargs["name"] = config["hf_config"]
        if token:
            load_kwargs["token"] = token
            
        dataset = load_dataset(**load_kwargs)
    except Exception as e:
        logger.error(f"Failed to load {name}: {e}")
        return {"name": name, "status": "error", "error": str(e)}
    
    # Process samples
    manifest = []
    count = 0
    audio_key = config.get("audio_key", "audio")
    text_key = config.get("text_key", "text")
    
    for idx, sample in enumerate(dataset):
        if count >= samples:
            break
            
        try:
            # Handle audio data
            if audio_key in sample:
                audio_data = sample[audio_key]
                
                # Handle different audio formats
                if isinstance(audio_data, dict):
                    audio_array = audio_data.get("array")
                    sample_rate = audio_data.get("sampling_rate", 16000)
                else:
                    continue
                
                if audio_array is None:
                    continue
                
                # Save audio file
                audio_filename = f"{name}_{idx:06d}.wav"
                audio_path = audio_dir / audio_filename
                
                if not save_audio(audio_array, sample_rate, audio_path):
                    continue
                
                # Build manifest entry
                entry = {
                    "id": f"{name}_{idx}",
                    "audio_path": str(audio_path.relative_to(output_dir)),
                    "audio_abs_path": str(normalize_path(audio_path)),
                    "text": sample.get(text_key, ""),
                    "modality": config.get("modality", "audio"),
                }
                
                # Add extra keys
                for key in config.get("extra_keys", []):
                    if key in sample:
                        entry[key] = sample[key]
                
                manifest.append(entry)
                count += 1
                
                if count % 50 == 0:
                    logger.info(f"  Processed {count}/{samples} samples...")
                    
        except Exception as e:
            logger.warning(f"Failed to process sample {idx}: {e}")
            continue
    
    # Save manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "name": name,
            "hf_path": config["hf_path"],
            "description": config.get("description", ""),
            "modality": config.get("modality", "audio"),
            "num_samples": len(manifest),
            "samples": manifest,
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Downloaded {name}: {len(manifest)} samples saved to {dataset_dir}")
    
    return {
        "name": name,
        "status": "success",
        "num_samples": len(manifest),
        "output_dir": str(dataset_dir),
    }


def download_with_hf_cli(
    hf_id: str,
    output_dir: Path,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Download dataset using huggingface-cli.
    
    Args:
        hf_id: HuggingFace dataset ID
        output_dir: Output directory
        token: HuggingFace token
        
    Returns:
        Dictionary with download statistics
    """
    cmd = [
        "huggingface-cli", "download",
        "--repo-type", "dataset",
        hf_id,
        "--local-dir", str(output_dir / hf_id.replace("/", "_")),
    ]
    
    if token:
        cmd.extend(["--token", token])
    
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            return {"hf_id": hf_id, "status": "success"}
        else:
            return {"hf_id": hf_id, "status": "error", "error": result.stderr}
    except subprocess.TimeoutExpired:
        return {"hf_id": hf_id, "status": "error", "error": "Download timed out"}
    except Exception as e:
        return {"hf_id": hf_id, "status": "error", "error": str(e)}


def list_datasets():
    """Print available datasets."""
    print("\n" + "=" * 80)
    print("Available Datasets for Download")
    print("=" * 80)
    
    for modality in ["audio", "video", "mixed"]:
        datasets = {k: v for k, v in DATASET_REGISTRY.items() if v.get("modality") == modality}
        if datasets:
            print(f"\n[{modality.upper()}]")
            for name, config in datasets.items():
                auth = " [AUTH]" if config.get("requires_auth") else ""
                print(f"  {name:20} - {config.get('description', '')}{auth}")
    
    print("\n" + "-" * 80)
    print("[AUTH] = Requires HuggingFace authentication")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download audio datasets for Speech SFT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./data",
        help="Output directory (default: ./data)"
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        type=str,
        default=None,
        help="Specific datasets to download (default: all)"
    )
    parser.add_argument(
        "--modality", "-m",
        type=str,
        choices=["audio", "video", "mixed", "all"],
        default="audio",
        help="Modality to download (default: audio)"
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=100,
        help="Number of samples per dataset (default: 100)"
    )
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=None,
        help="HuggingFace token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Path to DATASETS_CATALOG.md (optional)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit"
    )
    parser.add_argument(
        "--use-cli",
        action="store_true",
        help="Use huggingface-cli for downloading instead of datasets library"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    # Get token from args or environment
    token = args.token or os.environ.get("HF_TOKEN")
    
    # Setup output directory
    output_dir = normalize_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine datasets to download
    if args.datasets:
        datasets_to_download = {
            k: v for k, v in DATASET_REGISTRY.items()
            if k in args.datasets
        }
        # Validate
        invalid = [d for d in args.datasets if d not in DATASET_REGISTRY]
        if invalid:
            logger.warning(f"Unknown datasets: {invalid}")
    elif args.modality == "all":
        datasets_to_download = DATASET_REGISTRY
    else:
        datasets_to_download = {
            k: v for k, v in DATASET_REGISTRY.items()
            if v.get("modality") == args.modality
        }
    
    logger.info(f"Downloading {len(datasets_to_download)} datasets to {output_dir}")
    logger.info(f"Samples per dataset: {args.samples}")
    
    # Check environment
    hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    logger.info(f"HF_ENDPOINT: {hf_endpoint}")
    
    # Download each dataset
    results = []
    for name, config in datasets_to_download.items():
        if config.get("requires_auth") and not token:
            logger.warning(f"Skipping {name}: requires authentication (set HF_TOKEN)")
            results.append({"name": name, "status": "skipped", "reason": "Requires auth"})
            continue
        
        if args.use_cli:
            result = download_with_hf_cli(config["hf_path"], output_dir, token)
            result["name"] = name
        else:
            result = download_with_datasets_library(
                name, config, output_dir, args.samples, token
            )
        results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("Download Summary")
    print("=" * 80)
    
    success = [r for r in results if r["status"] == "success"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]
    
    print(f"\nSuccessful: {len(success)}")
    for r in success:
        samples = r.get("num_samples", "N/A")
        print(f"  [OK] {r['name']}: {samples} samples")
    
    if skipped:
        print(f"\nSkipped: {len(skipped)}")
        for r in skipped:
            print(f"  [-] {r['name']}: {r.get('reason', 'Unknown')}")
    
    if errors:
        print(f"\nErrors: {len(errors)}")
        for r in errors:
            print(f"  [X] {r['name']}: {r.get('error', 'Unknown error')}")
    
    # Save summary
    summary_path = output_dir / "download_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary saved to: {summary_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
