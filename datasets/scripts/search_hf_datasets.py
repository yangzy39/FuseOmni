#!/usr/bin/env python3
"""
Search HuggingFace Hub for Audio Datasets.

Searches and lists high-quality audio datasets from HuggingFace suitable for SFT.

Features:
- Search by task category (asr, audio-classification, etc.)
- Filter by language, size, and license
- Display dataset metadata and statistics
- Export results to JSON/CSV

Usage:
    # List popular audio ASR datasets
    python search_hf_datasets.py --task asr --limit 20
    
    # Search for Chinese audio datasets
    python search_hf_datasets.py --task asr --language zh --limit 10
    
    # Export results to JSON
    python search_hf_datasets.py --task asr --output datasets.json
    
    # Show detailed info for a specific dataset
    python search_hf_datasets.py --info openslr/librispeech_asr
"""

import argparse
import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any, Iterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetResult:
    """Dataset search result."""
    id: str
    author: str
    name: str
    downloads: int
    likes: int
    tags: List[str]
    description: str = ""
    license: str = ""
    task_categories: List[str] = None
    languages: List[str] = None
    size_categories: List[str] = None
    
    def __post_init__(self):
        if self.task_categories is None:
            self.task_categories = []
        if self.languages is None:
            self.languages = []
        if self.size_categories is None:
            self.size_categories = []


# Audio-related task categories on HuggingFace
AUDIO_TASK_CATEGORIES = [
    "automatic-speech-recognition",
    "audio-classification",
    "audio-to-audio",
    "text-to-audio",
    "text-to-speech",
    "voice-activity-detection",
    "audio-xvector",
]

# Common audio dataset tags
AUDIO_TAGS = [
    "audio",
    "speech",
    "asr",
    "tts",
    "speech-recognition",
    "speech-synthesis",
]

# Recommended high-quality audio datasets for SFT
RECOMMENDED_DATASETS = {
    "asr": [
        "openslr/librispeech_asr",
        "mozilla-foundation/common_voice_15_0",
        "speechcolab/gigaspeech",
        "facebook/voxpopuli",
        "wenet-e2e/wenetspeech",
        "AISHELL/AISHELL-1",
        "openslr/libritts",
    ],
    "audio_understanding": [
        "MMAU/mmau_mini",
        "MMSU/mmsu",
        "cvssp/WavCaps",
    ],
    "speech_translation": [
        "facebook/covost2",
        "google/fleurs",
    ],
}


def search_datasets(
    task_category: Optional[str] = None,
    search_query: Optional[str] = None,
    language: Optional[str] = None,
    author: Optional[str] = None,
    limit: int = 20,
    sort: str = "downloads",
) -> Iterator[DatasetResult]:
    """
    Search HuggingFace Hub for datasets.
    
    Args:
        task_category: Filter by task (e.g., "automatic-speech-recognition")
        search_query: Text search query
        language: Filter by language code (e.g., "en", "zh")
        author: Filter by author/organization
        limit: Maximum number of results
        sort: Sort field ("downloads", "likes", "lastModified")
        
    Yields:
        DatasetResult objects
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        return
    
    api = HfApi()
    
    # Build filter arguments
    filter_kwargs = {}
    
    if task_category:
        # Map common names to HF task categories
        task_map = {
            "asr": "automatic-speech-recognition",
            "tts": "text-to-speech",
            "audio": "audio-classification",
            "translation": "translation",
        }
        filter_kwargs["task_categories"] = task_map.get(task_category, task_category)
    
    if language:
        filter_kwargs["language"] = language
    
    if author:
        filter_kwargs["author"] = author
    
    try:
        datasets = api.list_datasets(
            search=search_query,
            sort=sort,
            direction=-1,  # Descending
            limit=limit,
            **filter_kwargs
        )
        
        count = 0
        for ds in datasets:
            if count >= limit:
                break
            
            # Parse tags to extract metadata
            tags = ds.tags or []
            task_cats = [t.split(":")[1] for t in tags if t.startswith("task_categories:")]
            languages = [t.split(":")[1] for t in tags if t.startswith("language:")]
            size_cats = [t.split(":")[1] for t in tags if t.startswith("size_categories:")]
            licenses = [t.split(":")[1] for t in tags if t.startswith("license:")]
            
            # Split id into author and name
            parts = ds.id.split("/", 1)
            author_name = parts[0] if len(parts) > 1 else ""
            dataset_name = parts[1] if len(parts) > 1 else ds.id
            
            yield DatasetResult(
                id=ds.id,
                author=author_name,
                name=dataset_name,
                downloads=ds.downloads or 0,
                likes=ds.likes or 0,
                tags=tags,
                description=getattr(ds, 'description', '') or '',
                license=licenses[0] if licenses else "",
                task_categories=task_cats,
                languages=languages,
                size_categories=size_cats,
            )
            count += 1
            
    except Exception as e:
        logger.error(f"Search failed: {e}")


def get_dataset_info(dataset_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific dataset.
    
    Args:
        dataset_id: HuggingFace dataset ID (e.g., "openslr/librispeech_asr")
        
    Returns:
        Dictionary with dataset information
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error("huggingface_hub not installed")
        return None
    
    api = HfApi()
    
    try:
        info = api.dataset_info(dataset_id)
        
        # Parse tags
        tags = info.tags or []
        
        return {
            "id": info.id,
            "author": info.author,
            "downloads": info.downloads,
            "likes": info.likes,
            "created_at": str(info.created_at) if info.created_at else None,
            "last_modified": str(info.last_modified) if info.last_modified else None,
            "private": info.private,
            "gated": info.gated,
            "tags": tags,
            "card_data": info.card_data.__dict__ if info.card_data else {},
            "siblings": [
                {"rfilename": s.rfilename, "size": s.size}
                for s in (info.siblings or [])[:20]  # Limit to first 20 files
            ],
        }
    except Exception as e:
        logger.error(f"Failed to get info for {dataset_id}: {e}")
        return None


def list_recommended_datasets(category: str = "asr") -> List[Dict[str, Any]]:
    """
    List recommended datasets for a category.
    
    Args:
        category: Dataset category (asr, audio_understanding, speech_translation)
        
    Returns:
        List of dataset info dictionaries
    """
    if category not in RECOMMENDED_DATASETS:
        logger.warning(f"Unknown category: {category}")
        return []
    
    results = []
    for dataset_id in RECOMMENDED_DATASETS[category]:
        info = get_dataset_info(dataset_id)
        if info:
            results.append(info)
        else:
            # Fallback with just the ID
            results.append({"id": dataset_id, "status": "info_unavailable"})
    
    return results


def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f}PB"


def print_dataset_table(datasets: List[DatasetResult]):
    """Print datasets in a formatted table."""
    print("\n" + "=" * 100)
    print(f"{'Dataset ID':<45} {'Downloads':>12} {'Likes':>8} {'Languages':<15} {'License':<15}")
    print("=" * 100)
    
    for ds in datasets:
        langs = ",".join(ds.languages[:3]) if ds.languages else "-"
        if len(ds.languages) > 3:
            langs += "..."
        license_str = ds.license[:12] + "..." if len(ds.license) > 15 else ds.license or "-"
        
        print(f"{ds.id:<45} {ds.downloads:>12,} {ds.likes:>8} {langs:<15} {license_str:<15}")
    
    print("=" * 100)
    print(f"Total: {len(datasets)} datasets")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Search HuggingFace Hub for audio datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--task", "-t",
        type=str,
        choices=["asr", "tts", "audio", "translation", "all"],
        default=None,
        help="Filter by task category"
    )
    parser.add_argument(
        "--search", "-s",
        type=str,
        default=None,
        help="Text search query"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default=None,
        help="Filter by language code (e.g., en, zh)"
    )
    parser.add_argument(
        "--author", "-a",
        type=str,
        default=None,
        help="Filter by author/organization"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=20,
        help="Maximum number of results (default: 20)"
    )
    parser.add_argument(
        "--sort",
        type=str,
        choices=["downloads", "likes", "lastModified"],
        default="downloads",
        help="Sort field (default: downloads)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (JSON)"
    )
    parser.add_argument(
        "--info", "-i",
        type=str,
        default=None,
        help="Get detailed info for a specific dataset ID"
    )
    parser.add_argument(
        "--recommended", "-r",
        type=str,
        choices=["asr", "audio_understanding", "speech_translation", "all"],
        default=None,
        help="List recommended datasets for a category"
    )
    
    args = parser.parse_args()
    
    # Get detailed info for a specific dataset
    if args.info:
        info = get_dataset_info(args.info)
        if info:
            print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
        return
    
    # List recommended datasets
    if args.recommended:
        if args.recommended == "all":
            categories = ["asr", "audio_understanding", "speech_translation"]
        else:
            categories = [args.recommended]
        
        all_results = {}
        for cat in categories:
            print(f"\n[{cat.upper()}]")
            results = list_recommended_datasets(cat)
            for r in results:
                downloads = r.get("downloads", "N/A")
                status = r.get("status", "")
                if status:
                    print(f"  {r['id']}: {status}")
                else:
                    print(f"  {r['id']}: {downloads:,} downloads")
            all_results[cat] = results
        
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\nSaved to {args.output}")
        return
    
    # Search datasets
    results = list(search_datasets(
        task_category=args.task,
        search_query=args.search,
        language=args.language,
        author=args.author,
        limit=args.limit,
        sort=args.sort,
    ))
    
    if not results:
        print("No datasets found matching your criteria.")
        return
    
    # Print results
    print_dataset_table(results)
    
    # Save to file
    if args.output:
        output_data = [asdict(r) for r in results]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
