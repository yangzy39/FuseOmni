#!/usr/bin/env python3
"""
Unified Speech Model Evaluation CLI.

This is the main entry point for running evaluations on speech/audio models
using vLLM-Omni as the inference engine.

Usage:
    python -m eval.eval --dataset librispeech_clean --model-path Qwen/Qwen2-Audio-7B-Instruct
    
    # List available datasets
    python -m eval.eval --list-datasets
    
    # With custom parameters
    python -m eval.eval --dataset mmau --model-path Qwen/Qwen2-Audio-7B-Instruct \
        --temperature 0.0 --max-tokens 128 --batch-size 8 --output-dir outputs/mmau_run1
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Import to register all datasets and metrics at module load
# pylint: disable=unused-import
from . import datasets  # noqa: F401
from . import metrics  # noqa: F401

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup_args() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="Speech Model Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run LibriSpeech evaluation
    python -m eval.eval --dataset librispeech_clean --model-path Qwen/Qwen2-Audio-7B-Instruct
    
    # Run MMAU with custom sampling
    python -m eval.eval --dataset mmau --model-path Qwen/Qwen2-Audio-7B-Instruct \\
        --temperature 0.0 --max-tokens 128
    
    # Resume a previous run
    python -m eval.eval --dataset librispeech_clean --model-path Qwen/Qwen2-Audio-7B-Instruct \\
        --resume --output-dir outputs/previous_run
        
    # Quick test with limited samples
    python -m eval.eval --dataset librispeech_clean --model-path Qwen/Qwen2-Audio-7B-Instruct \\
        --limit 10
""",
    )
    
    # Dataset selection
    dataset_group = parser.add_argument_group("Dataset Options")
    dataset_group.add_argument(
        "--dataset", "-d",
        type=str,
        help="Dataset name to evaluate on (use --list-datasets to see available)",
    )
    dataset_group.add_argument(
        "--list-datasets",
        action="store_true",
        help="List all available datasets and exit",
    )
    dataset_group.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to use (default: dataset's default split)",
    )
    dataset_group.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing)",
    )
    dataset_group.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Custom data directory (for locally downloaded datasets)",
    )
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--model-path", "-m",
        type=str,
        help="Path to the model (HuggingFace model ID or local path)",
    )
    model_group.add_argument(
        "--tensor-parallel-size", "-tp",
        type=int,
        default=1,
        help="Tensor parallel size for distributed inference (default: 1)",
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype (default: auto)",
    )
    model_group.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum model context length",
    )
    model_group.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0, default: 0.9)",
    )
    model_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code when loading model (default: True)",
    )
    
    # Sampling parameters
    sampling_group = parser.add_argument_group("Sampling Parameters")
    sampling_group.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0 for greedy)",
    )
    sampling_group.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling (default: 1.0)",
    )
    sampling_group.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling (default: -1, disabled)",
    )
    sampling_group.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    sampling_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty (default: 1.0)",
    )
    sampling_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    
    # Runtime options
    runtime_group = parser.add_argument_group("Runtime Options")
    runtime_group.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    runtime_group.add_argument(
        "--output-dir", "-o",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs)",
    )
    runtime_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip already processed samples)",
    )
    runtime_group.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser


def list_datasets():
    """Print all available datasets."""
    from .registry import list_datasets as get_datasets
    from .datasets import *  # noqa: F401, F403 - Import to register datasets
    
    datasets = get_datasets()
    
    print("\n" + "=" * 80)
    print("Available Datasets")
    print("=" * 80)
    
    # Group by task type
    by_task = {}
    for name, cls in sorted(datasets.items()):
        task = getattr(cls, "task_type", "other")
        if task not in by_task:
            by_task[task] = []
        by_task[task].append((name, cls))
    
    task_order = ["asr", "mcq", "qa", "translation", "emotion", "instruction"]
    for task in task_order:
        if task not in by_task:
            continue
        
        print(f"\n[{task.upper()}]")
        for name, cls in by_task[task]:
            desc = getattr(cls, "description", "")
            metrics = getattr(cls, "metrics", [])
            lang = getattr(cls, "language", "en")
            print(f"  {name:25s} | {lang:5s} | {', '.join(metrics):15s} | {desc}")
    
    # Print any remaining tasks
    for task, items in by_task.items():
        if task not in task_order:
            print(f"\n[{task.upper()}]")
            for name, cls in items:
                desc = getattr(cls, "description", "")
                print(f"  {name:25s} | {desc}")
    
    print("\n" + "=" * 80)
    print(f"Total: {len(datasets)} datasets")
    print("=" * 80 + "\n")


def run_evaluation(args: argparse.Namespace):
    """Run the evaluation."""
    from .schema import RunConfig, SamplingConfig, EngineConfig, EvalResult
    from .registry import get_dataset, get_metric
    from .datasets import *  # noqa: F401, F403 - Import to register datasets
    from .metrics import *  # noqa: F401, F403 - Import to register metrics
    from .engine import VllmOmniEngine
    from .io import load_existing_predictions, save_predictions, get_completed_ids, write_jsonl
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.dataset}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset_cls = get_dataset(args.dataset)
    dataset = dataset_cls(
        split=args.split,
        data_dir=args.data_dir,
        limit=args.limit,
    )
    
    # Collect samples
    logger.info("Loading samples...")
    samples = list(dataset.load())
    logger.info(f"Loaded {len(samples)} samples")
    
    # Handle resume
    completed_ids = set()
    if args.resume:
        completed_ids = get_completed_ids(output_dir)
        logger.info(f"Resuming: {len(completed_ids)} samples already completed")
        samples = [s for s in samples if s.id not in completed_ids]
        logger.info(f"Remaining: {len(samples)} samples")
    
    if not samples:
        logger.info("No samples to process")
        return
    
    # Initialize engine
    engine_config = EngineConfig(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )
    
    sampling_config = SamplingConfig(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    
    logger.info(f"Initializing engine with model: {args.model_path}")
    engine = VllmOmniEngine(engine_config)
    
    # Run inference
    logger.info("Starting inference...")
    start_time = time.time()
    
    predictions = engine.generate_batch(
        samples,
        sampling_config,
        batch_size=args.batch_size,
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Inference completed in {elapsed:.2f}s ({len(samples) / elapsed:.2f} samples/s)")
    
    # Post-process predictions
    logger.info("Post-processing predictions...")
    for pred, sample in zip(predictions, samples):
        pred.text = dataset.postprocess_prediction(pred.text, sample)
    
    # Save predictions
    save_predictions(output_dir, predictions, append=args.resume)
    logger.info(f"Saved predictions to {output_dir / 'predictions.jsonl'}")
    
    # Compute metrics
    logger.info("Computing metrics...")
    
    # Collect all predictions (including resumed ones)
    if args.resume:
        all_predictions = load_existing_predictions(output_dir)
        predictions_list = [all_predictions[s.id] for s in list(dataset_cls(
            split=args.split,
            data_dir=args.data_dir,
            limit=args.limit,
        ).load()) if s.id in all_predictions]
    else:
        predictions_list = predictions
    
    # Match predictions with references
    sample_map = {s.id: s for s in list(dataset_cls(
        split=args.split,
        data_dir=args.data_dir,
        limit=args.limit,
    ).load())}
    
    references = []
    hypotheses = []
    metas = []
    
    for pred in predictions_list:
        if pred.sample_id in sample_map:
            sample = sample_map[pred.sample_id]
            references.append(sample.reference)
            hypotheses.append(pred.text)
            metas.append(sample.meta)
    
    # Compute all metrics for this dataset
    all_metrics = {}
    for metric_name in dataset.metrics:
        try:
            metric_cls = get_metric(metric_name)
            metric = metric_cls()
            result = metric.compute(references, hypotheses, metas)
            all_metrics.update(result)
            logger.info(f"  {metric_name}: {result}")
        except Exception as e:
            logger.warning(f"Failed to compute metric {metric_name}: {e}")
    
    # Create result
    result = EvalResult(
        dataset=args.dataset,
        model_path=args.model_path,
        metrics=all_metrics,
        num_samples=len(predictions_list),
        config={
            "sampling": sampling_config.to_dict(),
            "engine": engine_config.to_dict(),
            "split": args.split or dataset.default_split,
            "limit": args.limit,
        },
        predictions_path=str(output_dir / "predictions.jsonl"),
    )
    
    # Save result
    result.save(output_dir / "metrics.json")
    logger.info(f"Saved metrics to {output_dir / 'metrics.json'}")
    
    # Save run config
    config = {
        "dataset": args.dataset,
        "model_path": args.model_path,
        "split": args.split or dataset.default_split,
        "limit": args.limit,
        "sampling": sampling_config.to_dict(),
        "engine": engine_config.to_dict(),
        "timestamp": timestamp,
        "elapsed_seconds": elapsed,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Dataset:    {args.dataset}")
    print(f"Model:      {args.model_path}")
    print(f"Samples:    {len(predictions_list)}")
    print(f"Time:       {elapsed:.2f}s")
    print("-" * 60)
    print("METRICS:")
    for metric_name, value in all_metrics.items():
        print(f"  {metric_name}: {value:.2f}")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = setup_args()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.list_datasets:
        list_datasets()
        return
    
    if not args.dataset:
        parser.error("--dataset is required (use --list-datasets to see options)")
    
    if not args.model_path:
        parser.error("--model-path is required")
    
    try:
        run_evaluation(args)
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
