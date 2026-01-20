#!/usr/bin/env python3
"""
Unified Dataset Download Script for Speech Model Evaluation.

Downloads all evaluation datasets to local storage for offline use.
Audio files are extracted and saved in a consistent format.

Usage:
    # Download all datasets
    python download_datasets.py --output-dir ./data
    
    # Download specific datasets
    python download_datasets.py --output-dir ./data --datasets librispeech_clean mmau
    
    # List available datasets
    python download_datasets.py --list
    
    # Download with specific split
    python download_datasets.py --output-dir ./data --split test --limit 100
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# Dataset registry: name -> (hf_path, hf_config, split, audio_key, text_key, extra_keys)
DATASET_REGISTRY = {
    # ASR Datasets
    "librispeech_clean": {
        "hf_path": "openslr/librispeech_asr",
        "hf_config": "clean",
        "split": "test",
        "audio_key": "audio",
        "text_key": "text",
        "extra_keys": ["speaker_id", "chapter_id"],
        "description": "LibriSpeech test-clean (clean English speech)",
    },
    "librispeech_other": {
        "hf_path": "openslr/librispeech_asr",
        "hf_config": "other",
        "split": "test",
        "audio_key": "audio",
        "text_key": "text",
        "extra_keys": ["speaker_id", "chapter_id"],
        "description": "LibriSpeech test-other (challenging English speech)",
    },
    "common_voice_en": {
        "hf_path": "mozilla-foundation/common_voice_17_0",
        "hf_config": "en",
        "split": "test",
        "audio_key": "audio",
        "text_key": "sentence",
        "extra_keys": ["age", "gender", "accent"],
        "description": "Common Voice English test set",
        "requires_auth": True,
    },
    "common_voice_zh": {
        "hf_path": "mozilla-foundation/common_voice_17_0",
        "hf_config": "zh-CN",
        "split": "test",
        "audio_key": "audio",
        "text_key": "sentence",
        "extra_keys": [],
        "description": "Common Voice Chinese test set",
        "requires_auth": True,
    },
    "aishell1": {
        "hf_path": "speechcolab/aishell1",
        "hf_config": None,
        "split": "test",
        "audio_key": "audio",
        "text_key": "text",
        "extra_keys": [],
        "description": "AISHELL-1 Mandarin Chinese ASR",
    },
    "gigaspeech": {
        "hf_path": "speechcolab/gigaspeech",
        "hf_config": "xs",
        "split": "test",
        "audio_key": "audio",
        "text_key": "text",
        "extra_keys": [],
        "description": "GigaSpeech English ASR (xs subset)",
        "requires_auth": True,
    },
    "wenetspeech": {
        "hf_path": "wenet/wenetspeech",
        "hf_config": None,
        "split": "test_net",
        "audio_key": "audio",
        "text_key": "text",
        "extra_keys": [],
        "description": "WenetSpeech Mandarin Chinese ASR",
    },
    
    # Audio Understanding Datasets
    "mmau": {
        "hf_path": "MMAU/mmau_mini",
        "hf_config": None,
        "split": "test",
        "audio_key": "audio",
        "text_key": "question",
        "extra_keys": ["option_a", "option_b", "option_c", "option_d", "answer", "category", "subcategory"],
        "description": "MMAU - Massive Multi-task Audio Understanding",
    },
    "mmau_pro": {
        "hf_path": "MMAU/mmau_pro",
        "hf_config": None,
        "split": "test",
        "audio_key": "audio",
        "text_key": "question",
        "extra_keys": ["option_a", "option_b", "option_c", "option_d", "answer", "scenario"],
        "description": "MMAU-Pro - Advanced audio understanding",
    },
    "mmsu": {
        "hf_path": "MMSU/mmsu",
        "hf_config": None,
        "split": "test",
        "audio_key": "audio",
        "text_key": "question",
        "extra_keys": ["option_a", "option_b", "option_c", "option_d", "answer", "task_type"],
        "description": "MMSU - Speech understanding (intonation, emotion, prosody)",
    },
    "airbench": {
        "hf_path": "AIR-Bench/air_bench",
        "hf_config": None,
        "split": "test",
        "audio_key": "audio",
        "text_key": "instruction",
        "extra_keys": ["answer", "domain"],
        "description": "AIR-Bench - Audio instruction recognition",
    },
    
    # Spoken QA Datasets
    "voicebench": {
        "hf_path": "voicebench/VoiceBench",
        "hf_config": None,
        "split": "test",
        "audio_key": "audio",
        "text_key": "question",
        "extra_keys": ["answer", "subset", "category"],
        "description": "VoiceBench - Comprehensive spoken QA",
    },
    "openaudiobench": {
        "hf_path": "baichuan-inc/OpenAudioBench",
        "hf_config": None,
        "split": "test",
        "audio_key": "audio",
        "text_key": "question",
        "extra_keys": ["answer", "task"],
        "description": "OpenAudioBench - Spoken question answering",
    },
    
    # Speech Translation Datasets
    "covost2_en_zh": {
        "hf_path": "facebook/covost2",
        "hf_config": "en_zh-CN",
        "split": "test",
        "audio_key": "audio",
        "text_key": "sentence",
        "extra_keys": ["translation"],
        "description": "CoVoST2 English to Chinese speech translation",
        "requires_auth": True,
    },
    "covost2_zh_en": {
        "hf_path": "facebook/covost2",
        "hf_config": "zh-CN_en",
        "split": "test",
        "audio_key": "audio",
        "text_key": "sentence",
        "extra_keys": ["translation"],
        "description": "CoVoST2 Chinese to English speech translation",
        "requires_auth": True,
    },
    "fleurs_en": {
        "hf_path": "google/fleurs",
        "hf_config": "en_us",
        "split": "test",
        "audio_key": "audio",
        "text_key": "transcription",
        "extra_keys": ["id"],
        "description": "FLEURS English (for translation to Chinese)",
    },
    "fleurs_zh": {
        "hf_path": "google/fleurs",
        "hf_config": "cmn_hans_cn",
        "split": "test",
        "audio_key": "audio",
        "text_key": "transcription",
        "extra_keys": ["id"],
        "description": "FLEURS Chinese (translation target)",
    },
    
    # Emotion Recognition Datasets
    "iemocap": {
        "hf_path": "Zahra99/IEMOCAP",
        "hf_config": None,
        "split": "test",
        "audio_key": "audio",
        "text_key": "text",
        "extra_keys": ["emotion", "speaker"],
        "description": "IEMOCAP - Interactive emotional speech",
        "note": "May require manual download due to licensing",
    },
    "meld": {
        "hf_path": "declare-lab/MELD",
        "hf_config": None,
        "split": "test",
        "audio_key": "audio",
        "text_key": "Utterance",
        "extra_keys": ["Emotion", "Sentiment", "Speaker"],
        "description": "MELD - Multimodal emotion (Friends TV)",
    },
}


def save_audio(audio_array: np.ndarray, sample_rate: int, output_path: Path):
    """Save audio array to WAV file."""
    try:
        import soundfile as sf
        # Ensure float32 format and normalize if needed
        audio = np.asarray(audio_array, dtype=np.float32)
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))
        sf.write(str(output_path), audio, sample_rate, format="WAV")
    except ImportError:
        try:
            from scipy.io import wavfile
            # Convert to int16 for scipy
            audio = np.asarray(audio_array, dtype=np.float32)
            if audio.max() <= 1.0 and audio.min() >= -1.0:
                audio = (audio * 32767).astype(np.int16)
            else:
                audio = audio.astype(np.int16)
            wavfile.write(str(output_path), sample_rate, audio)
        except ImportError:
            raise ImportError("Either soundfile or scipy is required. Install with: pip install soundfile")


def download_dataset(
    name: str,
    output_dir: Path,
    split: Optional[str] = None,
    limit: Optional[int] = None,
    force: bool = False,
) -> dict:
    """
    Download a single dataset and save to local storage.
    
    Args:
        name: Dataset name from registry
        output_dir: Base output directory
        split: Optional split override
        limit: Optional limit on number of samples
        force: Force re-download even if exists
        
    Returns:
        Dictionary with download statistics
    """
    if name not in DATASET_REGISTRY:
        logger.error(f"Unknown dataset: {name}")
        return {"name": name, "status": "error", "error": "Unknown dataset"}
    
    config = DATASET_REGISTRY[name]
    dataset_dir = output_dir / name
    manifest_path = dataset_dir / "manifest.json"
    audio_dir = dataset_dir / "audio"
    
    # Check if already downloaded
    if manifest_path.exists() and not force:
        logger.info(f"Dataset {name} already exists. Use --force to re-download.")
        return {"name": name, "status": "skipped", "reason": "Already exists"}
    
    # Create directories
    dataset_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(exist_ok=True)
    
    try:
        from datasets import load_dataset
    except ImportError:
        return {"name": name, "status": "error", "error": "datasets library not installed"}
    
    # Check for authentication requirement
    if config.get("requires_auth"):
        logger.warning(f"Dataset {name} may require HuggingFace authentication.")
        logger.warning("Run: huggingface-cli login")
    
    # Load dataset
    logger.info(f"Downloading {name} from {config['hf_path']}...")
    
    try:
        load_kwargs = {
            "path": config["hf_path"],
            "split": split or config["split"],
            "trust_remote_code": True,
        }
        if config.get("hf_config"):
            load_kwargs["name"] = config["hf_config"]
        
        dataset = load_dataset(**load_kwargs)
    except Exception as e:
        logger.error(f"Failed to load {name}: {e}")
        return {"name": name, "status": "error", "error": str(e)}
    
    # Process samples
    manifest = []
    count = 0
    
    for idx, sample in enumerate(dataset):
        if limit and count >= limit:
            break
        
        try:
            # Extract audio
            audio_data = sample[config["audio_key"]]
            audio_array = audio_data["array"]
            sample_rate = audio_data["sampling_rate"]
            
            # Save audio file
            audio_filename = f"{name}_{idx:06d}.wav"
            audio_path = audio_dir / audio_filename
            save_audio(audio_array, sample_rate, audio_path)
            
            # Build manifest entry
            entry = {
                "id": f"{name}_{idx}",
                "audio_path": str(audio_path.relative_to(output_dir)),
                "audio_filename": audio_filename,
                "sample_rate": sample_rate,
                "text": sample.get(config["text_key"], ""),
            }
            
            # Add extra keys
            for key in config.get("extra_keys", []):
                if key in sample:
                    entry[key] = sample[key]
            
            manifest.append(entry)
            count += 1
            
            if count % 100 == 0:
                logger.info(f"  Processed {count} samples...")
                
        except Exception as e:
            logger.warning(f"Failed to process sample {idx}: {e}")
            continue
    
    # Save manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "name": name,
            "description": config["description"],
            "hf_path": config["hf_path"],
            "split": split or config["split"],
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


def list_datasets():
    """Print available datasets."""
    print("\n" + "=" * 80)
    print("Available Datasets for Download")
    print("=" * 80)
    
    categories = {
        "ASR": ["librispeech_clean", "librispeech_other", "common_voice_en", "common_voice_zh", 
                "aishell1", "gigaspeech", "wenetspeech"],
        "Audio Understanding": ["mmau", "mmau_pro", "mmsu", "airbench"],
        "Spoken QA": ["voicebench", "openaudiobench"],
        "Speech Translation": ["covost2_en_zh", "covost2_zh_en", "fleurs_en", "fleurs_zh"],
        "Emotion Recognition": ["iemocap", "meld"],
    }
    
    for category, datasets in categories.items():
        print(f"\n[{category}]")
        for name in datasets:
            if name in DATASET_REGISTRY:
                config = DATASET_REGISTRY[name]
                auth_marker = " [AUTH]" if config.get("requires_auth") else ""
                print(f"  {name:25} - {config['description']}{auth_marker}")
    
    print("\n" + "-" * 80)
    print("[AUTH] = Requires HuggingFace authentication (huggingface-cli login)")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download evaluation datasets for speech model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download all datasets
    python download_datasets.py --output-dir ./data
    
    # Download specific datasets
    python download_datasets.py --output-dir ./data --datasets librispeech_clean mmau
    
    # Download with sample limit (for testing)
    python download_datasets.py --output-dir ./data --datasets librispeech_clean --limit 10
    
    # List available datasets
    python download_datasets.py --list
"""
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./data",
        help="Output directory for downloaded datasets (default: ./data)"
    )
    parser.add_argument(
        "--datasets", "-d",
        nargs="+",
        type=str,
        default=None,
        help="Specific datasets to download (default: all)"
    )
    parser.add_argument(
        "--split", "-s",
        type=str,
        default=None,
        help="Split to download (default: dataset-specific)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of samples per dataset (for testing)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if dataset exists"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets and exit"
    )
    parser.add_argument(
        "--parallel", "-p",
        type=int,
        default=1,
        help="Number of parallel downloads (default: 1)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_datasets()
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine datasets to download
    if args.datasets:
        datasets = args.datasets
        # Validate dataset names
        invalid = [d for d in datasets if d not in DATASET_REGISTRY]
        if invalid:
            logger.error(f"Unknown datasets: {invalid}")
            logger.info("Use --list to see available datasets")
            sys.exit(1)
    else:
        datasets = list(DATASET_REGISTRY.keys())
    
    logger.info(f"Downloading {len(datasets)} datasets to {output_dir}")
    
    # Download datasets
    results = []
    
    if args.parallel > 1:
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            futures = {
                executor.submit(
                    download_dataset, name, output_dir, args.split, args.limit, args.force
                ): name for name in datasets
            }
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for name in datasets:
            result = download_dataset(name, output_dir, args.split, args.limit, args.force)
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
        print(f"  ✓ {r['name']}: {r['num_samples']} samples")
    
    if skipped:
        print(f"\nSkipped: {len(skipped)}")
        for r in skipped:
            print(f"  - {r['name']}: {r.get('reason', 'Already exists')}")
    
    if errors:
        print(f"\nErrors: {len(errors)}")
        for r in errors:
            print(f"  ✗ {r['name']}: {r.get('error', 'Unknown error')}")
    
    # Save overall manifest
    overall_manifest = {
        "datasets": {r["name"]: r for r in results if r["status"] == "success"},
        "output_dir": str(output_dir),
    }
    with open(output_dir / "datasets_manifest.json", "w") as f:
        json.dump(overall_manifest, f, indent=2)
    
    print(f"\nManifest saved to: {output_dir / 'datasets_manifest.json'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
