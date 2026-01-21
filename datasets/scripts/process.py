#!/usr/bin/env python3
"""
Dataset Processing Script.

Processes downloaded datasets and converts them to MS-SWIFT SFT/GRPO format.
Each dataset has its own processor that understands the specific data structure.

Usage:
    # Process all datasets in a directory
    python process.py --input ./data --output ./output
    
    # Process a specific dataset
    python process.py --input ./data/librispeech --dataset librispeech --output ./output
    
    # Process with sample limit
    python process.py --input ./data --output ./output --max-samples 1000
    
    # List available processors
    python process.py --list
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from processors import (
    DATASET_CONFIGS,
    create_processor,
    list_available_datasets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def detect_dataset(data_dir: Path) -> Optional[str]:
    """
    Try to detect which dataset is in the given directory.
    
    Args:
        data_dir: Directory to check
        
    Returns:
        Dataset name if detected, None otherwise
    """
    # Check directory name
    dir_name = data_dir.name.lower()
    
    # Direct name match
    for name in DATASET_CONFIGS:
        if name in dir_name:
            return name
    
    # Check for specific markers
    markers = {
        "librispeech": ["train.clean", "train.other", "librispeech"],
        "common_voice": ["common_voice", "cv-corpus"],
        "gigaspeech": ["gigaspeech", "GigaSpeech"],
        "aishell1": ["aishell", "AISHELL"],
        "wavcaps": ["wavcaps", "WavCaps"],
        "wenetspeech": ["wenetspeech", "WenetSpeech"],
    }
    
    # Check files and subdirectories
    contents = [f.name.lower() for f in data_dir.iterdir()] if data_dir.exists() else []
    
    for dataset_name, keywords in markers.items():
        for keyword in keywords:
            if any(keyword.lower() in c for c in contents):
                return dataset_name
    
    return None


def process_single_dataset(
    data_dir: Path,
    output_dir: Path,
    dataset_name: str,
    max_samples: int = -1,
    task_type: str = "sft",
    system_prompt: Optional[str] = None,
    split: Optional[str] = None,
    subset: Optional[str] = None,
) -> Dict:
    """
    Process a single dataset.
    
    Args:
        data_dir: Directory containing the dataset
        output_dir: Output directory
        dataset_name: Name of the dataset
        max_samples: Maximum samples to process
        task_type: "sft" or "grpo"
        system_prompt: Optional system prompt
        split: Override default split
        subset: Override default subset
        
    Returns:
        Processing statistics
    """
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"  Input: {data_dir}")
    logger.info(f"  Output: {output_dir}")
    
    # Build kwargs for processor
    kwargs = {}
    if split:
        kwargs["split"] = split
    if subset:
        kwargs["subset"] = subset
    
    try:
        processor = create_processor(
            name=dataset_name,
            data_dir=data_dir,
            output_dir=output_dir,
            max_samples=max_samples,
            task_type=task_type,
            system_prompt=system_prompt,
            **kwargs,
        )
        
        stats = processor.process()
        return {"name": dataset_name, "status": "success", "stats": stats}
        
    except Exception as e:
        logger.error(f"Failed to process {dataset_name}: {e}")
        return {"name": dataset_name, "status": "error", "error": str(e)}


def process_all_datasets(
    input_dir: Path,
    output_dir: Path,
    max_samples: int = -1,
    task_type: str = "sft",
    system_prompt: Optional[str] = None,
) -> List[Dict]:
    """
    Process all datasets found in input directory.
    
    Args:
        input_dir: Directory containing dataset subdirectories
        output_dir: Output directory
        max_samples: Maximum samples per dataset
        task_type: "sft" or "grpo"
        system_prompt: Optional system prompt
        
    Returns:
        List of processing results
    """
    results = []
    
    # Find all subdirectories
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        logger.warning(f"No subdirectories found in {input_dir}")
        return results
    
    logger.info(f"Found {len(subdirs)} subdirectories in {input_dir}")
    
    for subdir in subdirs:
        # Try to detect dataset type
        dataset_name = detect_dataset(subdir)
        
        if dataset_name:
            logger.info(f"Detected {dataset_name} in {subdir.name}")
            result = process_single_dataset(
                data_dir=subdir,
                output_dir=output_dir,
                dataset_name=dataset_name,
                max_samples=max_samples,
                task_type=task_type,
                system_prompt=system_prompt,
            )
            results.append(result)
        else:
            logger.warning(f"Could not detect dataset type for {subdir.name}, skipping")
            results.append({
                "name": subdir.name,
                "status": "skipped",
                "reason": "Unknown dataset type",
            })
    
    return results


def merge_outputs(output_dir: Path, task_type: str = "sft") -> Path:
    """
    Merge all processed dataset outputs into a single file.
    
    Args:
        output_dir: Output directory containing processed datasets
        task_type: Task type to look for
        
    Returns:
        Path to merged file
    """
    merged_file = output_dir / f"all_{task_type}.jsonl"
    
    # Find all output files
    output_files = list(output_dir.rglob(f"{task_type}.jsonl"))
    
    if not output_files:
        logger.warning("No output files to merge")
        return merged_file
    
    logger.info(f"Merging {len(output_files)} files...")
    
    total_samples = 0
    with open(merged_file, "w", encoding="utf-8") as out_f:
        for output_file in output_files:
            with open(output_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
                    total_samples += 1
    
    logger.info(f"Merged {total_samples} samples into {merged_file}")
    return merged_file


def list_datasets_info():
    """Print available datasets and their information."""
    print("\n" + "=" * 80)
    print("Available Dataset Processors")
    print("=" * 80 + "\n")
    
    datasets = list_available_datasets()
    
    for name, config in sorted(datasets.items()):
        auth = " [AUTH]" if config.get("requires_auth") else ""
        print(f"  {name:20} - {config.get('description', '')}{auth}")
        print(f"                       HF: {config.get('hf_id', 'N/A')}")
        print(f"                       Lang: {config.get('language', 'N/A')}, Task: {config.get('task', 'N/A')}")
        print()
    
    print("-" * 80)
    print("[AUTH] = Requires HuggingFace authentication")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Process datasets and convert to MS-SWIFT format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=False,
        help="Input directory containing downloaded dataset(s)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./output",
        help="Output directory for processed data (default: ./output)",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="Specific dataset to process (if input is a single dataset directory)",
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=-1,
        help="Maximum samples per dataset (-1 for all)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["sft", "grpo"],
        default="sft",
        help="Task type: sft (with response) or grpo (prompt only)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System prompt to add to all samples",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Override default dataset split",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Override default dataset subset",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all outputs into a single file",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available dataset processors",
    )
    
    args = parser.parse_args()
    
    # List datasets if requested
    if args.list:
        list_datasets_info()
        return
    
    # Validate input
    if not args.input:
        parser.error("--input is required (use --list to see available datasets)")
    
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process
    if args.dataset:
        # Process single dataset
        results = [process_single_dataset(
            data_dir=input_dir,
            output_dir=output_dir,
            dataset_name=args.dataset,
            max_samples=args.max_samples,
            task_type=args.task,
            system_prompt=args.system,
            split=args.split,
            subset=args.subset,
        )]
    else:
        # Process all datasets in directory
        results = process_all_datasets(
            input_dir=input_dir,
            output_dir=output_dir,
            max_samples=args.max_samples,
            task_type=args.task,
            system_prompt=args.system,
        )
    
    # Merge if requested
    if args.merge:
        merge_outputs(output_dir, args.task)
    
    # Summary
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    
    success = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]
    skipped = [r for r in results if r["status"] == "skipped"]
    
    print(f"\nSuccessful: {len(success)}")
    for r in success:
        stats = r.get("stats", {})
        print(f"  [OK] {r['name']}: {stats.get('processed', 0)} samples")
    
    if skipped:
        print(f"\nSkipped: {len(skipped)}")
        for r in skipped:
            print(f"  [-] {r['name']}: {r.get('reason', 'Unknown')}")
    
    if errors:
        print(f"\nErrors: {len(errors)}")
        for r in errors:
            print(f"  [X] {r['name']}: {r.get('error', 'Unknown error')}")
    
    # Save summary
    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)
    
    print(f"\nSummary saved to: {summary_file}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
