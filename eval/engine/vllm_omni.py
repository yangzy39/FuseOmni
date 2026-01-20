"""
vLLM-Omni inference engine wrapper.

Based on https://github.com/vllm-project/vllm-omni
Provides a unified interface for running inference with vLLM-Omni on audio-language models.
"""

from typing import Any, Optional, Iterator
from pathlib import Path
import logging
import os

import numpy as np

from ..schema import EvalSample, EvalPrediction, SamplingConfig, EngineConfig

logger = logging.getLogger(__name__)

# Default system prompt for Qwen-Omni models
DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)


class VllmOmniEngine:
    """
    Wrapper for vLLM-Omni multimodal inference with audio support.
    
    Based on the official vLLM-Omni API:
    https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/examples/offline_inference/qwen3_omni/
    
    Handles:
        - Model loading with vLLM-Omni's Omni class
        - Audio prompt construction with proper placeholders
        - Batch generation with py_generator mode
        - Result parsing (text only, no audio generation for eval)
    
    Example:
        >>> engine = VllmOmniEngine(EngineConfig(model_path="Qwen/Qwen3-Omni-30B-A3B-Instruct"))
        >>> predictions = engine.generate(samples, SamplingConfig(temperature=0.0))
    """
    
    # Audio placeholder tokens for Qwen-Omni models
    AUDIO_PLACEHOLDER = "<|audio_start|><|audio_pad|><|audio_end|>"
    
    def __init__(self, config: EngineConfig):
        """
        Initialize the vLLM-Omni engine.
        
        Args:
            config: Engine configuration
        """
        self.config = config
        self.system_prompt = config.system_prompt or DEFAULT_SYSTEM_PROMPT
        
        self._omni = None
        self._initialized = False
    
    def _load_model(self):
        """Lazy load the vLLM-Omni model."""
        if self._initialized:
            return
        
        try:
            from vllm_omni.entrypoints.omni import Omni
        except ImportError:
            raise ImportError(
                "vLLM-Omni is required for inference. Install with:\n"
                "  pip install vllm-omni\n"
                "See: https://github.com/vllm-project/vllm-omni"
            )
        
        logger.info(f"Loading model: {self.config.model_path}")
        
        # Initialize Omni engine
        self._omni = Omni(
            model=self.config.model_path,
            stage_configs_path=self.config.stage_configs_path,
            log_stats=self.config.log_stats,
            stage_init_timeout=self.config.stage_init_timeout,
        )
        
        self._initialized = True
        logger.info("Model loaded successfully")
    
    def _build_audio_prompt(self, text_prompt: str) -> str:
        """
        Build the full prompt with audio placeholder in Qwen-Omni format.
        
        Args:
            text_prompt: The text prompt/question
            
        Returns:
            Full prompt with audio placeholder inserted
        """
        return (
            f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{self.AUDIO_PLACEHOLDER}"
            f"{text_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
    
    def _load_audio(self, audio_path: str | Path, sample_meta: dict = None) -> tuple:
        """
        Load audio file or use in-memory audio data.
        
        Args:
            audio_path: Path to audio file
            sample_meta: Sample metadata that may contain audio_array and sampling_rate
            
        Returns:
            Tuple of (audio_array, sample_rate) in float32 format
        """
        # Check if audio is already in memory (from HuggingFace datasets)
        if sample_meta and "audio_array" in sample_meta:
            audio_array = sample_meta["audio_array"]
            sample_rate = sample_meta.get("sampling_rate", 16000)
            
            # Convert to float32 numpy array
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array)
            audio_array = audio_array.astype(np.float32)
            
            return (audio_array, sample_rate)
        
        # Load from file
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            import librosa
            audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            return (audio.astype(np.float32), sr)
        except ImportError:
            try:
                import soundfile as sf
                audio, sr = sf.read(str(audio_path))
                # Resample to 16kHz if needed
                if sr != 16000:
                    import scipy.signal
                    audio = scipy.signal.resample(audio, int(len(audio) * 16000 / sr))
                    sr = 16000
                return (audio.astype(np.float32), sr)
            except ImportError:
                raise ImportError(
                    "Either librosa or soundfile is required for audio loading.\n"
                    "Install with: pip install librosa soundfile"
                )
    
    def _get_sampling_params(self, sampling: SamplingConfig):
        """Create vLLM SamplingParams from our config."""
        from vllm import SamplingParams
        
        return SamplingParams(
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            top_k=sampling.top_k if sampling.top_k > 0 else -1,
            max_tokens=sampling.max_tokens,
            repetition_penalty=sampling.repetition_penalty,
            seed=sampling.seed,
        )
    
    def generate(
        self,
        samples: list[EvalSample],
        sampling: SamplingConfig,
    ) -> list[EvalPrediction]:
        """
        Generate predictions for a batch of samples.
        
        Uses vLLM-Omni's Omni.generate() with py_generator=True for efficiency.
        Only extracts text output (no audio generation for evaluation).
        
        Args:
            samples: List of evaluation samples
            sampling: Sampling configuration
            
        Returns:
            List of predictions
        """
        self._load_model()
        
        # Build sampling params for thinker (text generation only)
        thinker_params = self._get_sampling_params(sampling)
        
        # For eval, we only need text output - use simplified params
        # We set modalities to ["text"] to skip audio generation
        sampling_params_list = [thinker_params]
        
        # Prepare inputs
        prompts = []
        for sample in samples:
            prompt_text = self._build_audio_prompt(sample.text_prompt)
            audio_data = self._load_audio(sample.audio_path, sample.meta)
            
            prompts.append({
                "prompt": prompt_text,
                "multi_modal_data": {
                    "audio": audio_data,
                },
                "modalities": ["text"],  # Text-only output for eval
            })
        
        # Generate with py_generator mode for efficient batch processing
        logger.info(f"Generating predictions for {len(samples)} samples")
        
        predictions = []
        sample_idx = 0
        
        # Use py_generator=True for streaming results
        omni_generator = self._omni.generate(
            prompts, 
            sampling_params_list, 
            py_generator=True
        )
        
        for stage_outputs in omni_generator:
            if stage_outputs.final_output_type == "text":
                for output in stage_outputs.request_output:
                    if sample_idx >= len(samples):
                        break
                    
                    sample = samples[sample_idx]
                    text_output = output.outputs[0].text
                    
                    predictions.append(EvalPrediction(
                        sample_id=sample.id,
                        text=text_output,
                        raw={
                            "finish_reason": str(output.outputs[0].finish_reason),
                            "request_id": output.request_id,
                        },
                        meta={
                            "prompt": output.prompt[:200] if output.prompt else "",
                        },
                    ))
                    sample_idx += 1
        
        return predictions
    
    def generate_batch(
        self,
        samples: list[EvalSample],
        sampling: SamplingConfig,
        batch_size: int = 8,
    ) -> list[EvalPrediction]:
        """
        Generate predictions with batching for memory efficiency.
        
        Args:
            samples: List of evaluation samples
            sampling: Sampling configuration
            batch_size: Number of samples per batch
            
        Returns:
            List of predictions
        """
        all_predictions = []
        
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            predictions = self.generate(batch, sampling)
            all_predictions.extend(predictions)
            
            logger.info(f"Processed {min(i + batch_size, len(samples))}/{len(samples)} samples")
        
        return all_predictions
    
    def close(self):
        """Close the engine and release resources."""
        if self._omni is not None:
            self._omni.close()
            self._omni = None
            self._initialized = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
