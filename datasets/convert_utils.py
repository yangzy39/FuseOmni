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


# =============================================================================
# MS-SWIFT 格式转换函数
# =============================================================================

@dataclass
class MSSwiftSample:
    """MS-SWIFT 标准数据格式
    
    用于 Qwen3-Omni 等多模态模型的 SFT/GRPO 训练。
    
    格式说明:
    - messages: 对话列表，包含 role (system/user/assistant) 和 content
    - images: 图片路径列表（可选）
    - videos: 视频路径列表（可选）
    - audios: 音频路径列表（可选）
    
    特殊标记:
    - <image>: 在 content 中标记图片位置
    - <video>: 在 content 中标记视频位置
    - <audio>: 在 content 中标记音频位置
    """
    messages: List[Dict[str, str]]
    images: Optional[List[str]] = None
    videos: Optional[List[str]] = None
    audios: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"messages": self.messages}
        if self.images:
            result["images"] = self.images
        if self.videos:
            result["videos"] = self.videos
        if self.audios:
            result["audios"] = self.audios
        return result


def convert_unified_to_msswift(
    input_path: Path,
    output_path: Path,
    task_type: str = "sft",
    system_prompt: Optional[str] = None,
    user_template: str = "{modality_tag}{text}",
    assistant_template: Optional[str] = None
) -> int:
    """将统一格式转换为 MS-SWIFT 格式
    
    Args:
        input_path: 输入的统一格式 JSONL 文件
        output_path: 输出的 MS-SWIFT 格式 JSONL 文件
        task_type: 任务类型，"sft" 或 "grpo"
            - sft: 包含完整的 user-assistant 对话
            - grpo: 仅包含 user 提示（用于强化学习）
        system_prompt: 系统提示（可选）
        user_template: 用户消息模板，支持 {modality_tag} 和 {text} 占位符
        assistant_template: 助手回复模板（仅 sft 模式），None 表示使用原始 text
        
    Returns:
        转换的样本数量
    """
    count = 0
    
    with open(input_path, 'r', encoding='utf-8') as in_f, \
         open(output_path, 'w', encoding='utf-8') as out_f:
        
        for line in in_f:
            data = json.loads(line)
            
            # 构建模态标记
            modality_tags = []
            images = []
            videos = []
            audios = []
            
            # 处理音频
            if data.get("audio"):
                audio_path = data["audio"]
                if isinstance(audio_path, str):
                    audios.append(audio_path)
                    modality_tags.append("<audio>")
                elif isinstance(audio_path, list):
                    audios.extend(audio_path)
                    modality_tags.extend(["<audio>"] * len(audio_path))
            
            # 处理视频
            if data.get("video"):
                video_path = data["video"]
                if isinstance(video_path, str):
                    videos.append(video_path)
                    modality_tags.append("<video>")
                elif isinstance(video_path, list):
                    videos.extend(video_path)
                    modality_tags.extend(["<video>"] * len(video_path))
            
            # 处理图片
            if data.get("image"):
                image_path = data["image"]
                if isinstance(image_path, str):
                    images.append(image_path)
                    modality_tags.append("<image>")
                elif isinstance(image_path, list):
                    images.extend(image_path)
                    modality_tags.extend(["<image>"] * len(image_path))
            
            # 构建消息
            messages = []
            
            # 添加系统提示
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # 构建用户消息
            text = data.get("text", "")
            modality_tag = "".join(modality_tags)
            user_content = user_template.format(
                modality_tag=modality_tag,
                text=text if text else "Describe the content."
            )
            messages.append({"role": "user", "content": user_content})
            
            # SFT 模式：添加助手回复
            if task_type == "sft" and text:
                if assistant_template:
                    assistant_content = assistant_template.format(text=text)
                else:
                    assistant_content = text
                messages.append({"role": "assistant", "content": assistant_content})
            
            # 构建样本
            sample = MSSwiftSample(
                messages=messages,
                images=images if images else None,
                videos=videos if videos else None,
                audios=audios if audios else None
            )
            
            out_f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
            count += 1
    
    logger.info(f"Converted {count} samples to MS-SWIFT format: {output_path}")
    return count


def convert_qa_to_msswift(
    input_path: Path,
    output_path: Path,
    question_key: str = "question",
    answer_key: str = "answer",
    image_key: Optional[str] = "image",
    audio_key: Optional[str] = "audio",
    video_key: Optional[str] = "video",
    system_prompt: Optional[str] = None,
    task_type: str = "sft"
) -> int:
    """将 QA 格式数据转换为 MS-SWIFT 格式
    
    适用于常见的 VQA、AudioQA 等问答数据集。
    
    Args:
        input_path: 输入 JSONL 文件
        output_path: 输出 MS-SWIFT 格式 JSONL 文件
        question_key: 问题字段名
        answer_key: 答案字段名
        image_key: 图片字段名（可选）
        audio_key: 音频字段名（可选）
        video_key: 视频字段名（可选）
        system_prompt: 系统提示（可选）
        task_type: "sft" 或 "grpo"
        
    Returns:
        转换的样本数量
    """
    count = 0
    
    with open(input_path, 'r', encoding='utf-8') as in_f, \
         open(output_path, 'w', encoding='utf-8') as out_f:
        
        for line in in_f:
            data = json.loads(line)
            
            messages = []
            images = []
            videos = []
            audios = []
            
            # 系统提示
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # 构建用户问题
            question = data.get(question_key, "")
            modality_prefix = ""
            
            # 处理多模态输入
            if image_key and data.get(image_key):
                img = data[image_key]
                if isinstance(img, str):
                    images.append(img)
                    modality_prefix += "<image>"
                elif isinstance(img, list):
                    images.extend(img)
                    modality_prefix += "<image>" * len(img)
            
            if audio_key and data.get(audio_key):
                aud = data[audio_key]
                if isinstance(aud, str):
                    audios.append(aud)
                    modality_prefix += "<audio>"
                elif isinstance(aud, list):
                    audios.extend(aud)
                    modality_prefix += "<audio>" * len(aud)
            
            if video_key and data.get(video_key):
                vid = data[video_key]
                if isinstance(vid, str):
                    videos.append(vid)
                    modality_prefix += "<video>"
                elif isinstance(vid, list):
                    videos.extend(vid)
                    modality_prefix += "<video>" * len(vid)
            
            user_content = f"{modality_prefix}{question}" if modality_prefix else question
            messages.append({"role": "user", "content": user_content})
            
            # SFT 模式添加答案
            if task_type == "sft" and data.get(answer_key):
                messages.append({"role": "assistant", "content": data[answer_key]})
            
            sample = MSSwiftSample(
                messages=messages,
                images=images if images else None,
                videos=videos if videos else None,
                audios=audios if audios else None
            )
            
            out_f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + '\n')
            count += 1
    
    logger.info(f"Converted {count} QA samples to MS-SWIFT format: {output_path}")
    return count


def create_msswift_sample(
    user_content: str,
    assistant_content: Optional[str] = None,
    system_prompt: Optional[str] = None,
    images: Optional[List[str]] = None,
    audios: Optional[List[str]] = None,
    videos: Optional[List[str]] = None
) -> Dict[str, Any]:
    """创建单个 MS-SWIFT 格式样本
    
    便捷函数，用于在代码中直接创建样本。
    
    Example:
        >>> sample = create_msswift_sample(
        ...     user_content="<audio>What did the speaker say?",
        ...     assistant_content="The speaker said hello.",
        ...     audios=["/path/to/audio.wav"]
        ... )
        >>> print(json.dumps(sample, ensure_ascii=False))
    """
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    messages.append({"role": "user", "content": user_content})
    
    if assistant_content:
        messages.append({"role": "assistant", "content": assistant_content})
    
    result = {"messages": messages}
    
    if images:
        result["images"] = images
    if audios:
        result["audios"] = audios
    if videos:
        result["videos"] = videos
    
    return result


def validate_msswift_format(jsonl_path: Path) -> Dict[str, Any]:
    """验证 MS-SWIFT 格式的 JSONL 文件
    
    Returns:
        包含验证结果的字典
    """
    stats = {
        "total": 0,
        "valid": 0,
        "with_images": 0,
        "with_audios": 0,
        "with_videos": 0,
        "with_system": 0,
        "sft_samples": 0,  # 有 assistant 回复的
        "grpo_samples": 0,  # 只有 user 提问的
        "errors": []
    }
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            stats["total"] += 1
            try:
                data = json.loads(line)
                
                # 检查必需字段
                if "messages" not in data:
                    stats["errors"].append(f"Line {idx + 1}: Missing 'messages' field")
                    continue
                
                messages = data["messages"]
                if not isinstance(messages, list) or len(messages) == 0:
                    stats["errors"].append(f"Line {idx + 1}: 'messages' must be a non-empty list")
                    continue
                
                # 检查消息格式
                has_user = False
                has_assistant = False
                has_system = False
                
                for msg in messages:
                    if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                        stats["errors"].append(f"Line {idx + 1}: Invalid message format")
                        continue
                    
                    role = msg["role"]
                    if role == "user":
                        has_user = True
                    elif role == "assistant":
                        has_assistant = True
                    elif role == "system":
                        has_system = True
                
                if not has_user and not has_assistant:
                    stats["errors"].append(f"Line {idx + 1}: No user or assistant message found")
                    continue
                
                stats["valid"] += 1
                
                if has_system:
                    stats["with_system"] += 1
                if has_assistant:
                    stats["sft_samples"] += 1
                else:
                    stats["grpo_samples"] += 1
                
                # 检查多模态字段
                if data.get("images"):
                    stats["with_images"] += 1
                if data.get("audios"):
                    stats["with_audios"] += 1
                if data.get("videos"):
                    stats["with_videos"] += 1
                    
            except json.JSONDecodeError as e:
                stats["errors"].append(f"Line {idx + 1}: JSON decode error - {str(e)}")
    
    return stats


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
    
    # MS-SWIFT 格式转换
    msswift_parser = subparsers.add_parser("msswift", help="Convert to MS-SWIFT format")
    msswift_parser.add_argument("input", type=Path, help="Input JSONL file (unified format)")
    msswift_parser.add_argument("output", type=Path, help="Output MS-SWIFT JSONL file")
    msswift_parser.add_argument("--task", choices=["sft", "grpo"], default="sft",
                                help="Task type: sft (with responses) or grpo (prompts only)")
    msswift_parser.add_argument("--system", type=str, default=None,
                                help="System prompt to add")
    msswift_parser.add_argument("--user-template", type=str, 
                                default="{modality_tag}{text}",
                                help="User message template")
    
    # QA 格式转换
    qa_parser = subparsers.add_parser("qa-msswift", help="Convert QA format to MS-SWIFT")
    qa_parser.add_argument("input", type=Path, help="Input QA JSONL file")
    qa_parser.add_argument("output", type=Path, help="Output MS-SWIFT JSONL file")
    qa_parser.add_argument("--question-key", default="question", help="Question field name")
    qa_parser.add_argument("--answer-key", default="answer", help="Answer field name")
    qa_parser.add_argument("--image-key", default="image", help="Image field name")
    qa_parser.add_argument("--audio-key", default="audio", help="Audio field name")
    qa_parser.add_argument("--video-key", default="video", help="Video field name")
    qa_parser.add_argument("--system", type=str, default=None, help="System prompt")
    qa_parser.add_argument("--task", choices=["sft", "grpo"], default="sft", help="Task type")
    
    # 验证 MS-SWIFT 格式
    validate_msswift_parser = subparsers.add_parser("validate-msswift", 
                                                     help="Validate MS-SWIFT format")
    validate_msswift_parser.add_argument("input", type=Path, help="Input MS-SWIFT JSONL file")
    
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
    elif args.command == "msswift":
        convert_unified_to_msswift(
            args.input, args.output,
            task_type=args.task,
            system_prompt=args.system,
            user_template=args.user_template
        )
    elif args.command == "qa-msswift":
        convert_qa_to_msswift(
            args.input, args.output,
            question_key=args.question_key,
            answer_key=args.answer_key,
            image_key=args.image_key,
            audio_key=args.audio_key,
            video_key=args.video_key,
            system_prompt=args.system,
            task_type=args.task
        )
    elif args.command == "validate-msswift":
        stats = validate_msswift_format(args.input)
        print("\nMS-SWIFT Format Validation:")
        print(f"  Total samples: {stats['total']}")
        print(f"  Valid samples: {stats['valid']}")
        print(f"  SFT samples (with response): {stats['sft_samples']}")
        print(f"  GRPO samples (prompts only): {stats['grpo_samples']}")
        print(f"  With system prompt: {stats['with_system']}")
        print(f"  With images: {stats['with_images']}")
        print(f"  With audios: {stats['with_audios']}")
        print(f"  With videos: {stats['with_videos']}")
        if stats['errors']:
            print(f"  Errors: {len(stats['errors'])}")
            for err in stats['errors'][:10]:
                print(f"    - {err}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
