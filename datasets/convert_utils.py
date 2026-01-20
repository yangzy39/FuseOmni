"""
REAP-OMNI 数据格式转换工具

将不同格式的数据集转换为统一的 JSONL 格式。
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class UnifiedSample:
    """统一数据格式"""
    id: str
    text: Optional[str] = None
    audio: Optional[str] = None
    video: Optional[str] = None
    modality: str = "mixed"
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"id": self.id, "modality": self.modality}
        if self.text is not None:
            result["text"] = self.text
        if self.audio is not None:
            result["audio"] = self.audio
        if self.video is not None:
            result["video"] = self.video
        return result


def extract_audio_from_video(video_path: Path, output_dir: Path) -> Optional[Path]:
    """从视频中提取音频"""
    audio_path = output_dir / f"{video_path.stem}.wav"
    
    if audio_path.exists():
        return audio_path
    
    try:
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn",  # 不要视频
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(audio_path),
            "-y"
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        return audio_path
    except Exception as e:
        logger.warning(f"Failed to extract audio from {video_path}: {e}")
        return None


def convert_csv_to_jsonl(
    csv_path: Path,
    output_path: Path,
    id_col: str = "id",
    text_col: str = "text",
    audio_col: Optional[str] = None,
    video_col: Optional[str] = None,
    modality: str = "mixed"
) -> int:
    """将CSV格式转换为JSONL"""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    count = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            sample = UnifiedSample(
                id=str(row.get(id_col, f"sample_{idx:05d}")),
                text=row.get(text_col) if text_col and text_col in row else None,
                audio=row.get(audio_col) if audio_col and audio_col in row else None,
                video=row.get(video_col) if video_col and video_col in row else None,
                modality=modality
            )
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
            count += 1
    
    logger.info(f"Converted {count} samples from {csv_path} to {output_path}")
    return count


def convert_json_to_jsonl(
    json_path: Path,
    output_path: Path,
    id_key: str = "id",
    text_key: str = "text",
    audio_key: Optional[str] = None,
    video_key: Optional[str] = None,
    modality: str = "mixed"
) -> int:
    """将JSON格式转换为JSONL"""
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 处理列表或字典格式
    if isinstance(data, dict):
        items = list(data.values()) if not any(isinstance(v, dict) for v in data.values()) else [data]
    else:
        items = data
    
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(items):
            if isinstance(item, dict):
                sample = UnifiedSample(
                    id=str(item.get(id_key, f"sample_{idx:05d}")),
                    text=item.get(text_key),
                    audio=item.get(audio_key) if audio_key else None,
                    video=item.get(video_key) if video_key else None,
                    modality=modality
                )
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
                count += 1
    
    logger.info(f"Converted {count} samples from {json_path} to {output_path}")
    return count


def convert_folder_to_jsonl(
    folder_path: Path,
    output_path: Path,
    audio_ext: str = ".wav",
    video_ext: str = ".mp4",
    text_ext: str = ".txt",
    modality: str = "mixed",
    dataset_name: str = "dataset"
) -> int:
    """将文件夹结构转换为JSONL"""
    
    # 查找所有媒体文件
    audio_files = list(folder_path.rglob(f"*{audio_ext}"))
    video_files = list(folder_path.rglob(f"*{video_ext}"))
    
    # 合并文件列表
    all_files = set()
    for f in audio_files + video_files:
        all_files.add(f.stem)
    
    count = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, stem in enumerate(sorted(all_files)):
            # 查找对应的文件
            audio_path = None
            video_path = None
            text = None
            
            for af in audio_files:
                if af.stem == stem:
                    audio_path = str(af)
                    break
            
            for vf in video_files:
                if vf.stem == stem:
                    video_path = str(vf)
                    break
            
            # 查找文本文件
            for parent in [folder_path] + list(folder_path.parents):
                txt_file = parent / f"{stem}{text_ext}"
                if txt_file.exists():
                    with open(txt_file, 'r', encoding='utf-8') as tf:
                        text = tf.read().strip()
                    break
            
            sample = UnifiedSample(
                id=f"{dataset_name}_{idx:05d}",
                text=text,
                audio=audio_path,
                video=video_path,
                modality=modality
            )
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
            count += 1
    
    logger.info(f"Converted {count} samples from {folder_path} to {output_path}")
    return count


def merge_jsonl_files(
    input_files: List[Path],
    output_path: Path,
    max_samples: Optional[int] = None
) -> int:
    """合并多个JSONL文件"""
    
    count = 0
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for input_file in input_files:
            if not input_file.exists():
                logger.warning(f"File not found: {input_file}")
                continue
            
            with open(input_file, 'r', encoding='utf-8') as in_f:
                for line in in_f:
                    if max_samples and count >= max_samples:
                        break
                    out_f.write(line)
                    count += 1
            
            if max_samples and count >= max_samples:
                break
    
    logger.info(f"Merged {count} samples to {output_path}")
    return count


def split_jsonl_by_modality(
    input_path: Path,
    output_dir: Path
) -> Dict[str, int]:
    """按模态分割JSONL文件"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = {
        "audio": open(output_dir / "audio.jsonl", 'w', encoding='utf-8'),
        "video": open(output_dir / "video.jsonl", 'w', encoding='utf-8'),
        "mixed": open(output_dir / "mixed.jsonl", 'w', encoding='utf-8'),
    }
    
    counts = {"audio": 0, "video": 0, "mixed": 0}
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            modality = data.get("modality", "mixed")
            
            if modality not in files:
                modality = "mixed"
            
            files[modality].write(line)
            counts[modality] += 1
    
    for f in files.values():
        f.close()
    
    logger.info(f"Split by modality: {counts}")
    return counts


def validate_jsonl(jsonl_path: Path) -> Dict[str, Any]:
    """验证JSONL文件格式"""
    
    stats = {
        "total": 0,
        "with_text": 0,
        "with_audio": 0,
        "with_video": 0,
        "modalities": {"audio": 0, "video": 0, "mixed": 0},
        "errors": []
    }
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
                stats["total"] += 1
                
                if data.get("text"):
                    stats["with_text"] += 1
                if data.get("audio"):
                    stats["with_audio"] += 1
                if data.get("video"):
                    stats["with_video"] += 1
                
                modality = data.get("modality", "mixed")
                if modality in stats["modalities"]:
                    stats["modalities"][modality] += 1
                    
            except json.JSONDecodeError as e:
                stats["errors"].append(f"Line {idx + 1}: {str(e)}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="REAP-OMNI 数据格式转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # CSV 转换
    csv_parser = subparsers.add_parser("csv", help="Convert CSV to JSONL")
    csv_parser.add_argument("input", type=Path, help="Input CSV file")
    csv_parser.add_argument("output", type=Path, help="Output JSONL file")
    csv_parser.add_argument("--id-col", default="id", help="ID column name")
    csv_parser.add_argument("--text-col", default="text", help="Text column name")
    csv_parser.add_argument("--audio-col", help="Audio column name")
    csv_parser.add_argument("--video-col", help="Video column name")
    csv_parser.add_argument("--modality", default="mixed", help="Modality type")
    
    # JSON 转换
    json_parser = subparsers.add_parser("json", help="Convert JSON to JSONL")
    json_parser.add_argument("input", type=Path, help="Input JSON file")
    json_parser.add_argument("output", type=Path, help="Output JSONL file")
    json_parser.add_argument("--id-key", default="id", help="ID key")
    json_parser.add_argument("--text-key", default="text", help="Text key")
    json_parser.add_argument("--audio-key", help="Audio key")
    json_parser.add_argument("--video-key", help="Video key")
    json_parser.add_argument("--modality", default="mixed", help="Modality type")
    
    # 文件夹转换
    folder_parser = subparsers.add_parser("folder", help="Convert folder to JSONL")
    folder_parser.add_argument("input", type=Path, help="Input folder")
    folder_parser.add_argument("output", type=Path, help="Output JSONL file")
    folder_parser.add_argument("--name", default="dataset", help="Dataset name prefix")
    folder_parser.add_argument("--modality", default="mixed", help="Modality type")
    
    # 合并
    merge_parser = subparsers.add_parser("merge", help="Merge JSONL files")
    merge_parser.add_argument("inputs", type=Path, nargs="+", help="Input JSONL files")
    merge_parser.add_argument("--output", "-o", type=Path, required=True, help="Output file")
    merge_parser.add_argument("--max-samples", type=int, help="Max samples")
    
    # 分割
    split_parser = subparsers.add_parser("split", help="Split JSONL by modality")
    split_parser.add_argument("input", type=Path, help="Input JSONL file")
    split_parser.add_argument("--output-dir", "-o", type=Path, required=True, help="Output directory")
    
    # 验证
    validate_parser = subparsers.add_parser("validate", help="Validate JSONL file")
    validate_parser.add_argument("input", type=Path, help="Input JSONL file")
    
    args = parser.parse_args()
    
    if args.command == "csv":
        convert_csv_to_jsonl(
            args.input, args.output,
            args.id_col, args.text_col,
            args.audio_col, args.video_col,
            args.modality
        )
    elif args.command == "json":
        convert_json_to_jsonl(
            args.input, args.output,
            args.id_key, args.text_key,
            args.audio_key, args.video_key,
            args.modality
        )
    elif args.command == "folder":
        convert_folder_to_jsonl(
            args.input, args.output,
            modality=args.modality,
            dataset_name=args.name
        )
    elif args.command == "merge":
        merge_jsonl_files(args.inputs, args.output, args.max_samples)
    elif args.command == "split":
        split_jsonl_by_modality(args.input, args.output_dir)
    elif args.command == "validate":
        stats = validate_jsonl(args.input)
        print("\nValidation Results:")
        print(f"  Total samples: {stats['total']}")
        print(f"  With text: {stats['with_text']}")
        print(f"  With audio: {stats['with_audio']}")
        print(f"  With video: {stats['with_video']}")
        print(f"  Modalities: {stats['modalities']}")
        if stats['errors']:
            print(f"  Errors: {len(stats['errors'])}")
            for err in stats['errors'][:5]:
                print(f"    - {err}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
