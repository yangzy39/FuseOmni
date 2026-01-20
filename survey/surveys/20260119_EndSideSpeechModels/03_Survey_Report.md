# End-side Speech Models in the Large Model Era: A 2025 Survey

**Date**: January 19, 2026
**Scope**: 2025-2026 High-Impact Research
**Focus**: On-Device Inference, Speech-Language Models (SLMs), Edge Efficiency

---

## 1. Executive Summary
The landscape of on-device speech processing has undergone a paradigm shift in 2025. We have moved from the era of **"Small Specialized Models"** (e.g., dedicated acoustic models for ASR, KWS) to the era of **"Edge Speech-Language Models (Edge SLMs)"**.

The defining characteristic of this new era is the **Integration of Reasoning**. Modern edge speech models are no longer just transcribers; they are reasoning agents capable of understanding context, managing dialogue state, and generating speech, all within the power envelope of a mobile device (typically <5W).

Key drivers identified in this survey:
1.  **Unified Architectures**: Convergence of ASR and LLM into single end-to-end differentiable stacks (e.g., LLaMA-Omni2).
2.  **System-Level Optimization**: Shift from purely algorithmic compression to hardware-aware pipelining (e.g., WhisperKit on CoreML, ChipChat on MLX).
3.  **Streaming First**: Static processing is obsolete. 2025 models prioritize streaming token generation for sub-second latency.

---

## 2. Taxonomy of 2025 Edge Speech Models

We categorize the recent advancements into four distinct pillars:

### 2.1 Native Edge SLMs (Speech-Language Models)
These are 1B-3B parameter models designed from scratch (or heavily adapted) to run on mobile NPUs. They possess native audio understanding capabilities.

*   **MobileLLM-Pro (arXiv:2511.06719)**: Sets the standard for mobile-first architecture. It introduces *Implicit Positional Distillation* to handle long contexts (128k) on device, proving that 1B models can rival older 7B models in reasoning if optimized correctly.
*   **Granite-speech (arXiv:2505.08699)**: Represents the "Enterprise Open Weights" trend. IBM's 2B model aligns a strong text backbone with a Conformer encoder, capable of running on high-end edge hardware (NVIDIA Orin, Laptops) for private enterprise ASR.

### 2.2 End-to-End Speech Interaction (S2S)
The traditional cascade (ASR -> LLM -> TTS) is being challenged by unified models that offer lower latency and better expressivity.

*   **LLaMA-Omni2 (arXiv:2505.02625)**: A breakthrough in data efficiency. It achieves high-quality spoken dialogue using only 200k samples. Its **Autoregressive Streaming Decoder** allows it to "speak while thinking," drastically reducing perceived latency.
*   **VocalNet (arXiv:2504.04060)**: Introduces **Multi-Token Prediction (MTP)** to speech generation. By predicting multiple acoustic tokens per step, it overcomes the memory-bandwidth bottleneck of autoregressive generation on edge devices.

### 2.3 Optimization & Alignment Techniques
How do we fit these giants onto a phone?

*   **WhisperKit (arXiv:2507.10860)**: A masterclass in *System-Software Co-design*. It doesn't change the Whisper model but rewrites the compute graph to fit Apple's Neural Engine (ANE), achieving 0.46s latency (matching cloud) on an iPad/iPhone.
*   **Tiny-Align (arXiv:2411.13766)**: Addresses the *Training* bottleneck. It enables **Personalized Alignment** on device (e.g., Jetson Orin), allowing the model to adapt to a specific user's voice without cloud finetuning.
*   **On-device Streaming DSU (arXiv:2506.01845)**: Solves the input bottleneck. By optimizing self-supervised models to produce **Discrete Speech Units** in a streaming fashion, it provides a compact semantic input for SLMs.

### 2.4 Specialized Niches: Wearables & Agents
*   **SING (arXiv:2504.08907)**: Bringing **Spatial Awareness** to LLMs. For wearables (glasses), it's not enough to hear "what" was said, but "where" (e.g., "the person to my left"). SING fuses spatial microstructure data with Whisper embeddings.
*   **ChipChat (arXiv:2509.00078)**: Apple's MLX-based agent. It proves that *Cascaded* systems can still be SOTA if the pipeline is optimized (0.8s latency). It uses a "State-Action" augmented LLM to control device functions while talking.
*   **Flavors of Moonshine (arXiv:2509.02523)**: A counter-trend. Instead of one giant multilingual model, it proposes **Tiny (27M)** specialized models for specific languages, beating generalist models on constrained IoT hardware.

---

## 3. Emerging Trends & Future Outlook

### Trend 1: The "Streaming" Imperative
In 2025, "Real-time" means "Streaming". Whether it's **LLaMA-Omni2**'s streaming decoder or **ChipChat**'s pipelined execution, the industry has declared war on the "Turn-Taking Latency" (the silence between user finish and bot start).

### Trend 2: Privacy via "Small-but-Smart"
**MobileLLM-Pro** and **Granite-speech 2B** demonstrate that we don't need to offload to the cloud for intelligence. With 4-bit quantization and architectural tricks (Specialist Merging), reasonable intelligence is now local.

### Trend 3: Modality Fusion on Chip
**SING** shows the future of wearables: fusing Audio + Spatial + Text. Future models will likely include Vision (Egocentric Video) in the same 2B parameter envelope for smart glasses.

### Trend 4: Personalized On-Device Training
**Tiny-Align** hints at a future where your AI assistant learns your accent and vocabulary *locally*, updating its weights overnight while charging, ensuring privacy and personalization.

---

## 4. Conclusion
The "End-side Speech Model" field in 2025 is defined by **Efficiency** and **Integration**. We are no longer compressing models just to save space; we are compressing them to enable *agency* and *interaction*. The convergence of ASR, LLM, and TTS into unified, streaming, silicon-optimized stacks is the defining technical achievement of this year.

---
*Generated by Auto-Survey Agent via 'sisyphus'. Based on arXiv 2025 literature.*
