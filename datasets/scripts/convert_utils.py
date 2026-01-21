#!/usr/bin/env python3
"""
Data Format Conversion Utilities for Speech SFT.

Converts audio datasets to MS-SWIFT compatible JSON format for training.

Target Format (SFT):
{
    "messages": [
        {"role": "user", "content": "<audio>What did the audio say?"},
        {"role": "assistant", "content": "The transcription content."}
    ],
    "audios": ["/absolute/path/to/audio.wav"]
}

Usage:
    # Convert manifest to MS-SWIFT SFT format
    python convert_utils.py msswift input.jsonl output.jsonl --task sft
    
    # Convert with custom prompts
    python convert_utils.py msswift input.jsonl output.jsonl --task sft \\
        --user-template "<audio>Please transcribe this audio."
    
    # Validate MS-SWIFT format
    python convert_utils.py validate output.jsonl
    
    # Merge multiple JSONL files
    python convert_utils.py merge file1.jsonl file2.jsonl -o merged.jsonl
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Literal
from dataclasses import dataclass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# === Cross-Platform Path Utilities ===

def normalize_path(path: str | Path) -> Path:
    """Normalize path for cross-platform compatibility."""
    path = Path(path)
    return path.resolve()


def to_posix_path(path: str | Path) -> str:
    """Convert path to POSIX format (forward slashes)."""
    return str(Path(path)).replace("\\", "/")


def get_absolute_path(path: str | Path, base_dir: Optional[Path] = None) -> str:
    """Get absolute path, optionally relative to a base directory."""
    path = Path(path)
    if path.is_absolute():
        return str(normalize_path(path))
    if base_dir:
        return str(normalize_path(base_dir / path))
    return str(normalize_path(path))


# === JSONL I/O Utilities ===

def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    """Read records from a JSONL file."""
    path = Path(path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        return
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")


def write_jsonl(path: str | Path, records: List[Dict[str, Any]]) -> None:
    """Write records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    logger.info(f"Wrote {len(records)} records to {path}")


def append_jsonl(path: str | Path, records: List[Dict[str, Any]]) -> None:
    """Append records to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Load all records from a JSONL file into memory."""
    return list(read_jsonl(path))


# === MS-SWIFT Format Creation ===

@dataclass
class MSSwiftSample:
    """MS-SWIFT format sample."""
    messages: List[Dict[str, str]]
    audios: Optional[List[str]] = None
    videos: Optional[List[str]] = None
    images: Optional[List[str]] = None


def create_msswift_sample(
    user_content: str,
    assistant_content: Optional[str] = None,
    system_prompt: Optional[str] = None,
    audios: Optional[List[str]] = None,
    videos: Optional[List[str]] = None,
    images: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create a single MS-SWIFT format sample.
    
    Args:
        user_content: User message content (should include <audio>/<video>/<image> tags)
        assistant_content: Assistant response (None for GRPO format)
        system_prompt: Optional system prompt
        audios: List of absolute audio file paths
        videos: List of absolute video file paths
        images: List of absolute image file paths
        
    Returns:
        Dictionary in MS-SWIFT format
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": user_content})
    
    if assistant_content is not None:
        messages.append({"role": "assistant", "content": assistant_content})
    
    result = {"messages": messages}
    
    if audios:
        result["audios"] = audios
    if videos:
        result["videos"] = videos
    if images:
        result["images"] = images
    
    return result


def get_modality_tag(modality: str) -> str:
    """Get the modality placeholder tag."""
    tags = {
        "audio": "<audio>",
        "video": "<video>",
        "image": "<image>",
    }
    return tags.get(modality, "<audio>")


def get_default_user_prompt(modality: str, task: str = "asr") -> str:
    """Get default user prompt based on modality and task."""
    prompts = {
        ("audio", "asr"): "<audio>What did the audio say?",
        ("audio", "transcribe"): "<audio>Transcribe the following audio exactly as spoken.",
        ("audio", "caption"): "<audio>Describe what you hear in this audio.",
        ("audio", "qa"): "<audio>Listen to the audio and answer the question.",
        ("video", "caption"): "<video>Describe what happens in this video.",
        ("video", "qa"): "<video>Watch the video and answer the question.",
        ("mixed", "caption"): "<audio><video>Describe what you see and hear.",
    }
    return prompts.get((modality, task), f"{get_modality_tag(modality)}What did the audio say?")


# === Format Conversion Functions ===

def convert_unified_to_msswift(
    input_path: str | Path,
    output_path: str | Path,
    task_type: Literal["sft", "grpo"] = "sft",
    system_prompt: Optional[str] = None,
    user_template: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """
    Convert unified format to MS-SWIFT format.
    
    Unified format:
    {
        "id": "sample_001",
        "text": "transcription or description",
        "audio": "path/to/audio.wav",  # or audio_path, audio_abs_path
        "video": "path/to/video.mp4",  # optional
        "modality": "audio"
    }
    
    Args:
        input_path: Input JSONL file
        output_path: Output JSONL file
        task_type: "sft" (with response) or "grpo" (prompt only)
        system_prompt: Optional system prompt to add
        user_template: Custom user message template (use {text}, {modality_tag})
        base_dir: Base directory for resolving relative paths
        
    Returns:
        Statistics dictionary
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if base_dir is None:
        base_dir = input_path.parent
    
    output_records = []
    stats = {"total": 0, "converted": 0, "errors": 0}
    
    for record in read_jsonl(input_path):
        stats["total"] += 1
        
        try:
            # Extract modality
            modality = record.get("modality", "audio")
            modality_tag = get_modality_tag(modality)
            
            # Extract media paths
            audio_path = record.get("audio_abs_path") or record.get("audio_path") or record.get("audio")
            video_path = record.get("video_abs_path") or record.get("video_path") or record.get("video")
            
            # Resolve to absolute paths
            audios = None
            videos = None
            
            if audio_path:
                abs_audio = get_absolute_path(audio_path, base_dir)
                audios = [abs_audio]
            
            if video_path:
                abs_video = get_absolute_path(video_path, base_dir)
                videos = [abs_video]
            
            # Build user content
            if user_template:
                user_content = user_template.format(
                    text=record.get("text", ""),
                    modality_tag=modality_tag,
                )
            else:
                user_content = get_default_user_prompt(modality, "asr")
            
            # Build assistant content (only for SFT)
            assistant_content = None
            if task_type == "sft":
                assistant_content = record.get("text", "")
                if not assistant_content:
                    logger.warning(f"Empty text for sample {record.get('id', stats['total'])}")
            
            # Create MS-SWIFT sample
            sample = create_msswift_sample(
                user_content=user_content,
                assistant_content=assistant_content,
                system_prompt=system_prompt,
                audios=audios,
                videos=videos,
            )
            
            output_records.append(sample)
            stats["converted"] += 1
            
        except Exception as e:
            logger.warning(f"Error converting record {stats['total']}: {e}")
            stats["errors"] += 1
    
    # Write output
    write_jsonl(output_path, output_records)
    
    logger.info(f"Converted {stats['converted']}/{stats['total']} samples "
                f"({stats['errors']} errors)")
    
    return stats


def convert_manifest_to_msswift(
    manifest_path: str | Path,
    output_path: str | Path,
    task_type: Literal["sft", "grpo"] = "sft",
    system_prompt: Optional[str] = None,
    user_template: Optional[str] = None,
) -> Dict[str, int]:
    """
    Convert a dataset manifest.json to MS-SWIFT format.
    
    Manifest format:
    {
        "name": "dataset_name",
        "samples": [
            {"id": "...", "audio_path": "...", "text": "..."},
            ...
        ]
    }
    """
    manifest_path = Path(manifest_path)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    samples = manifest.get("samples", [])
    base_dir = manifest_path.parent.parent  # Typically datasets/data/
    
    output_records = []
    stats = {"total": 0, "converted": 0, "errors": 0}
    
    for record in samples:
        stats["total"] += 1
        
        try:
            modality = record.get("modality", "audio")
            modality_tag = get_modality_tag(modality)
            
            # Get audio path
            audio_path = record.get("audio_abs_path") or record.get("audio_path")
            if audio_path:
                audio_path = get_absolute_path(audio_path, base_dir)
            
            # Build content
            if user_template:
                user_content = user_template.format(
                    text=record.get("text", ""),
                    modality_tag=modality_tag,
                )
            else:
                user_content = get_default_user_prompt(modality, "asr")
            
            assistant_content = record.get("text", "") if task_type == "sft" else None
            
            sample = create_msswift_sample(
                user_content=user_content,
                assistant_content=assistant_content,
                system_prompt=system_prompt,
                audios=[audio_path] if audio_path else None,
            )
            
            output_records.append(sample)
            stats["converted"] += 1
            
        except Exception as e:
            logger.warning(f"Error converting sample: {e}")
            stats["errors"] += 1
    
    write_jsonl(output_path, output_records)
    
    return stats


def convert_qa_to_msswift(
    input_path: str | Path,
    output_path: str | Path,
    question_key: str = "question",
    answer_key: str = "answer",
    audio_key: str = "audio",
    video_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, int]:
    """
    Convert QA format data to MS-SWIFT format.
    
    Input format:
    {
        "question": "What is being discussed?",
        "answer": "The speaker talks about...",
        "audio": "/path/to/audio.wav"
    }
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    base_dir = input_path.parent
    
    output_records = []
    stats = {"total": 0, "converted": 0, "errors": 0}
    
    for record in read_jsonl(input_path):
        stats["total"] += 1
        
        try:
            question = record.get(question_key, "")
            answer = record.get(answer_key, "")
            
            # Determine modality and build user content
            audio_path = record.get(audio_key)
            video_path = record.get(video_key) if video_key else None
            
            if audio_path:
                user_content = f"<audio>{question}" if question else "<audio>Answer the question."
                audios = [get_absolute_path(audio_path, base_dir)]
                videos = None
            elif video_path:
                user_content = f"<video>{question}" if question else "<video>Answer the question."
                videos = [get_absolute_path(video_path, base_dir)]
                audios = None
            else:
                user_content = question
                audios = None
                videos = None
            
            sample = create_msswift_sample(
                user_content=user_content,
                assistant_content=answer,
                system_prompt=system_prompt,
                audios=audios,
                videos=videos,
            )
            
            output_records.append(sample)
            stats["converted"] += 1
            
        except Exception as e:
            logger.warning(f"Error converting QA record: {e}")
            stats["errors"] += 1
    
    write_jsonl(output_path, output_records)
    
    return stats


# === Validation ===

def validate_msswift_format(path: str | Path) -> Dict[str, Any]:
    """
    Validate MS-SWIFT format JSONL file.
    
    Returns statistics and validation results.
    """
    path = Path(path)
    
    stats = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "sft_samples": 0,
        "grpo_samples": 0,
        "with_audio": 0,
        "with_video": 0,
        "with_image": 0,
        "errors": [],
    }
    
    for idx, record in enumerate(read_jsonl(path)):
        stats["total"] += 1
        line_num = idx + 1
        
        # Check required fields
        if "messages" not in record:
            stats["invalid"] += 1
            stats["errors"].append(f"Line {line_num}: Missing 'messages' field")
            continue
        
        messages = record["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            stats["invalid"] += 1
            stats["errors"].append(f"Line {line_num}: 'messages' must be a non-empty list")
            continue
        
        # Check message structure
        valid_roles = {"system", "user", "assistant"}
        has_user = False
        has_assistant = False
        
        for msg in messages:
            if not isinstance(msg, dict):
                stats["invalid"] += 1
                stats["errors"].append(f"Line {line_num}: Message must be a dict")
                break
            
            role = msg.get("role")
            if role not in valid_roles:
                stats["invalid"] += 1
                stats["errors"].append(f"Line {line_num}: Invalid role '{role}'")
                break
            
            if role == "user":
                has_user = True
            elif role == "assistant":
                has_assistant = True
        else:
            # All messages valid
            if not has_user:
                stats["invalid"] += 1
                stats["errors"].append(f"Line {line_num}: Missing user message")
                continue
            
            stats["valid"] += 1
            
            if has_assistant:
                stats["sft_samples"] += 1
            else:
                stats["grpo_samples"] += 1
            
            if record.get("audios"):
                stats["with_audio"] += 1
            if record.get("videos"):
                stats["with_video"] += 1
            if record.get("images"):
                stats["with_image"] += 1
    
    return stats


# === Merge/Split Utilities ===

def merge_jsonl_files(
    input_paths: List[str | Path],
    output_path: str | Path,
    shuffle: bool = False,
) -> int:
    """Merge multiple JSONL files into one."""
    all_records = []
    
    for path in input_paths:
        records = load_jsonl(path)
        all_records.extend(records)
        logger.info(f"Loaded {len(records)} records from {path}")
    
    if shuffle:
        import random
        random.shuffle(all_records)
    
    write_jsonl(output_path, all_records)
    
    return len(all_records)


def split_by_modality(
    input_path: str | Path,
    output_dir: str | Path,
) -> Dict[str, int]:
    """Split JSONL file by modality."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    modality_records: Dict[str, List[Dict]] = {
        "audio": [],
        "video": [],
        "mixed": [],
        "text": [],
    }
    
    for record in read_jsonl(input_path):
        has_audio = bool(record.get("audios"))
        has_video = bool(record.get("videos"))
        
        if has_audio and has_video:
            modality_records["mixed"].append(record)
        elif has_audio:
            modality_records["audio"].append(record)
        elif has_video:
            modality_records["video"].append(record)
        else:
            modality_records["text"].append(record)
    
    stats = {}
    for modality, records in modality_records.items():
        if records:
            output_path = output_dir / f"{modality}.jsonl"
            write_jsonl(output_path, records)
            stats[modality] = len(records)
    
    return stats


# === CLI ===

def main():
    parser = argparse.ArgumentParser(
        description="Data format conversion utilities for Speech SFT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # msswift command
    msswift_parser = subparsers.add_parser(
        "msswift",
        help="Convert to MS-SWIFT format"
    )
    msswift_parser.add_argument("input", help="Input JSONL file")
    msswift_parser.add_argument("output", help="Output JSONL file")
    msswift_parser.add_argument(
        "--task", choices=["sft", "grpo"], default="sft",
        help="Task type: sft (with response) or grpo (prompt only)"
    )
    msswift_parser.add_argument(
        "--system", type=str, default=None,
        help="System prompt to add"
    )
    msswift_parser.add_argument(
        "--user-template", type=str, default=None,
        help="Custom user message template (use {text}, {modality_tag})"
    )
    
    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate MS-SWIFT format"
    )
    validate_parser.add_argument("input", help="Input JSONL file to validate")
    
    # merge command
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge multiple JSONL files"
    )
    merge_parser.add_argument("inputs", nargs="+", help="Input JSONL files")
    merge_parser.add_argument("-o", "--output", required=True, help="Output JSONL file")
    merge_parser.add_argument("--shuffle", action="store_true", help="Shuffle records")
    
    # split command
    split_parser = subparsers.add_parser(
        "split",
        help="Split JSONL file by modality"
    )
    split_parser.add_argument("input", help="Input JSONL file")
    split_parser.add_argument("-o", "--output", required=True, help="Output directory")
    
    # manifest command
    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Convert manifest.json to MS-SWIFT format"
    )
    manifest_parser.add_argument("input", help="Input manifest.json file")
    manifest_parser.add_argument("output", help="Output JSONL file")
    manifest_parser.add_argument(
        "--task", choices=["sft", "grpo"], default="sft",
        help="Task type"
    )
    manifest_parser.add_argument("--system", type=str, default=None)
    manifest_parser.add_argument("--user-template", type=str, default=None)
    
    # qa command
    qa_parser = subparsers.add_parser(
        "qa",
        help="Convert QA format to MS-SWIFT"
    )
    qa_parser.add_argument("input", help="Input JSONL file")
    qa_parser.add_argument("output", help="Output JSONL file")
    qa_parser.add_argument("--question-key", default="question")
    qa_parser.add_argument("--answer-key", default="answer")
    qa_parser.add_argument("--audio-key", default="audio")
    qa_parser.add_argument("--system", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.command == "msswift":
        stats = convert_unified_to_msswift(
            args.input,
            args.output,
            task_type=args.task,
            system_prompt=args.system,
            user_template=args.user_template,
        )
        print(f"Converted {stats['converted']} samples")
        
    elif args.command == "validate":
        stats = validate_msswift_format(args.input)
        print("\n" + "=" * 60)
        print("Validation Results")
        print("=" * 60)
        print(f"Total samples: {stats['total']}")
        print(f"Valid: {stats['valid']}")
        print(f"Invalid: {stats['invalid']}")
        print(f"SFT samples: {stats['sft_samples']}")
        print(f"GRPO samples: {stats['grpo_samples']}")
        print(f"With audio: {stats['with_audio']}")
        print(f"With video: {stats['with_video']}")
        if stats['errors']:
            print(f"\nFirst 5 errors:")
            for err in stats['errors'][:5]:
                print(f"  - {err}")
        print("=" * 60)
        
    elif args.command == "merge":
        count = merge_jsonl_files(args.inputs, args.output, shuffle=args.shuffle)
        print(f"Merged {count} records into {args.output}")
        
    elif args.command == "split":
        stats = split_by_modality(args.input, args.output)
        print(f"Split into: {stats}")
        
    elif args.command == "manifest":
        stats = convert_manifest_to_msswift(
            args.input,
            args.output,
            task_type=args.task,
            system_prompt=args.system,
            user_template=args.user_template,
        )
        print(f"Converted {stats['converted']} samples")
        
    elif args.command == "qa":
        stats = convert_qa_to_msswift(
            args.input,
            args.output,
            question_key=args.question_key,
            answer_key=args.answer_key,
            audio_key=args.audio_key,
            system_prompt=args.system,
        )
        print(f"Converted {stats['converted']} samples")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
