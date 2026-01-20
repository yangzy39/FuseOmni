#!/usr/bin/env python3
"""
REAP-OMNI 快速开始脚本

快速下载推荐的数据集组合用于模型校准。
"""

import subprocess
import sys
from pathlib import Path


def install_requirements():
    """安装必要的依赖"""
    requirements = [
        "datasets",
        "soundfile",
        "tqdm",
        "pandas",
    ]
    
    print("Installing required packages...")
    for req in requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", req, "-q"])
    print("Done!")


def download_recommended_datasets(output_dir: str = "./calibration_data", samples: int = 100):
    """下载推荐的数据集组合"""
    
    from download_datasets import process_datasets, save_jsonl
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 推荐的数据集组合
    recommended = {
        "audio": ["librispeech", "gigaspeech"],
        "video": ["longvideobench"],
        "mixed": ["voxceleb", "lrs2"],
    }
    
    print("\n" + "=" * 60)
    print("REAP-OMNI 推荐数据集下载")
    print("=" * 60)
    
    all_datasets = []
    for modality, datasets in recommended.items():
        print(f"\n{modality.upper()} 数据集: {', '.join(datasets)}")
        all_datasets.extend(datasets)
    
    print(f"\n输出目录: {output_path}")
    print(f"每个数据集采样数: {samples}")
    print("=" * 60)
    
    # 处理数据集
    results = process_datasets(all_datasets, output_path, samples)
    
    # 保存结果
    calibration_dir = output_path / "calibration"
    calibration_dir.mkdir(exist_ok=True)
    
    for modality, samples_list in results.items():
        if samples_list:
            save_jsonl(samples_list, calibration_dir, modality)
    
    print("\n" + "=" * 60)
    print("下载完成!")
    print("=" * 60)
    print(f"  Audio samples: {len(results['audio'])}")
    print(f"  Video samples: {len(results['video'])}")
    print(f"  Mixed samples: {len(results['mixed'])}")
    
    print(f"\n校准数据位置: {calibration_dir}")
    print("\n运行 REAP 剪枝命令:")
    print(f"""
python reap_expert_pruning.py \\
    --model-path <your_model_path> \\
    --output-path <output_path> \\
    --audio-data {calibration_dir / 'audio.jsonl'} \\
    --video-data {calibration_dir / 'video.jsonl'} \\
    --mixed-data {calibration_dir / 'mixed.jsonl'} \\
    --retention-rate 0.5
""")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="REAP-OMNI 快速开始")
    parser.add_argument("--install", action="store_true", help="安装依赖")
    parser.add_argument("--output", "-o", default="./calibration_data", help="输出目录")
    parser.add_argument("--samples", "-s", type=int, default=100, help="每个数据集采样数")
    
    args = parser.parse_args()
    
    if args.install:
        install_requirements()
    
    download_recommended_datasets(args.output, args.samples)


if __name__ == "__main__":
    main()
