# Research Plan: End-side Speech Models in the LLM Era (2025+)

## 1. Research Goal
To survey the landscape of **On-Device Speech Models in the Era of Large Models** (2025-2026). The focus shifts from traditional specific-task small models to **Speech-Language Models (SLMs)**, **Multimodal Edge Models**, and **Generative Speech Models** adapted for edge deployment.

## 2. Keywords Strategy
*   **Core**: `on-device speech language model`, `edge multimodal LLM`, `mobile generative speech`
*   **Specific Models/Tech**: `Whisper distillation`, `Gemini Nano`, `Llama edge speech`, `Audio-LLM quantization`, `streaming speech LLM`
*   **Tasks**: `Speech-to-Text (ASR) via LLM`, `Speech-to-Speech (S2S) translation edge`, `Zero-shot TTS mobile`
*   **Optimization**: `LLM quantization for mobile`, `LoRA for speech edge`, `KV-cache compression speech`

## 3. Sub-directions
1.  **Edge Speech-Language Models (Edge SLMs)**: Adapting 1B-3B parameter speech-text models for devices.
2.  **Unified Multimodal Architectures**: Models handling Audio + Text (and potentially Vision) on chip.
3.  **Generative ASR & TTS**: Moving beyond discriminative models to generative approaches (e.g., codec-based).
4.  **Efficient Fine-tuning & Inference**: Running adapter-based speech models (LoRA) on NPU.
5.  **Agentic Speech Interfaces**: Low-latency voice interaction with local intelligence.

## 4. Selection Criteria
*   **Time Range**: **Strictly 2025 - 2026**.
*   **Sources**:
    *   **arXiv** (Primary source for latest 2025/2026 work)
    *   **Conferences**: ICASSP 2025, Interspeech 2025, NeurIPS 2025, ICLR 2025/2026.
    *   **Industry Reports**: Google (Gemini Nano), Apple (Apple Intelligence/Siri), Meta (Llama), Microsoft (Phi-Audio).
*   **Relevance**: Must involve "Speech" AND "Large Model/Generative" AND "On-device/Edge Constraints".

## 5. Target Scale
*   **Broad Scan**: ~100 papers (Due to strict time window, might be fewer but more recent).
*   **Deep Reading**: ~15 key papers representing the SOTA of 2025.
