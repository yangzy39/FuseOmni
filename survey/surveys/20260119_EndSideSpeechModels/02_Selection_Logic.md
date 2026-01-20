# Selection Logic for Deep Analysis

## Selection Criteria
Based on the "Large Model Era" and "2025+" constraints, we selected papers that represent:
1.  **Architecture Shifts**: From pure ASR/TTS to unified Speech-LLMs (SLMs).
2.  **Edge Constraints**: Explicit focus on latency, memory, or specific edge hardware (NPU/DSP).
3.  **New Capabilities**: Streaming, Speech-to-Speech, Agentic interaction.

## Selected Papers (Deep Dive List)

### A. Core Edge Architectures & Optimization
1.  **WhisperKit: On-device Real-time ASR with Billion-Scale Transformers** (arXiv:2507.10860)
    *   *Reason*: Defines the SOTA for deploying heavy Transformers (Whisper-v3) on consumer hardware via CoreML.
2.  **MobileLLM-Pro: On-Device LLM for Mobile** (arXiv:2511.06719)
    *   *Reason*: Essential reading for understanding generic LLM optimization for mobile, which applies to SLMs.
3.  **Tiny-Align: Bridging ASR and LLM on the Edge** (arXiv:2411.13766, v2 2025)
    *   *Reason*: Addresses the specific problem of aligning small acoustic encoders with LLMs under resource constraints.

### B. Speech-to-Speech & Interaction (End-to-End)
4.  **LLaMA-Omni2: LLM-based Real-time Spoken Chatbot** (arXiv:2505.02625)
    *   *Reason*: Represents the shift to "Speech-in, Speech-out" without cascading latency.
5.  **ChipChat: Low-Latency Cascaded Conversational Agent in MLX** (arXiv:2509.00078)
    *   *Reason*: Apple's implementation of local conversational agents, focusing on system-level latency optimization.
6.  **VocalNet: Speech LLM with Multi-Token Prediction** (arXiv:2504.04060)
    *   *Reason*: Novel decoding technique (Multi-token) specifically to speed up speech generation on device.

### C. Wearables & Context Aware
7.  **SING: Spatial Context in Large Language Model for Next-Gen Wearables** (arXiv:2504.08907)
    *   *Reason*: Explores the "Wearable + LLM" niche, critical for glasses/earbuds.
8.  **LLAMAPIE: Proactive In-Ear Conversation Assistants** (arXiv:2505.04066)
    *   *Reason*: Focuses on the "Proactive" aspect of edge agents, a 2025 trend.

### D. Specialized & Streaming
9.  **Flavors of Moonshine: Tiny Specialized ASR Models** (arXiv:2509.02523)
    *   *Reason*: Counter-trend – highly specialized tiny models vs general purpose LLMs.
10. **On-device Streaming Discrete Speech Units** (arXiv:2506.01845)
    *   *Reason*: Technical deep dive into "Discrete Units" which are key for compression and streaming.

### E. Open Foundation Models
11. **Granite-speech: Open-source speech-aware LLMs** (arXiv:2505.08699)
    *   *Reason*: Represents the "Open Weights" trend for enterprise edge speech.
12. **Omnilingual ASR** (arXiv:2511.09690)
    *   *Reason*: Massive scale multilingualism, relevant for global edge deployment.

### F. Contextual Survey
13. **Empowering Edge Intelligence** (arXiv:2503.06027)
    *   *Reason*: Provides the taxonomy for the final report.
