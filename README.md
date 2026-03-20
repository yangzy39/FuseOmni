# 🌌 FuseOmni: Edge-Side Multi-Modal Speech Model Fusion

**FuseOmni** is an end-to-end research project focused on **Edge-side Multi-modal (Speech/Voice) Model Fusion**. Our objective is to build highly efficient, sparsely-activated Mixture-of-Experts (SMoE) models that can run seamlessly on edge devices while maintaining state-of-the-art performance in complex speech and reasoning tasks.

---

## 🚀 Key Features

- **Training Excellence**: Leveraging `ms-swift` for efficient fine-tuning and adaptation.
- **Advanced Pruning**: Utilizing **REAP (Router-weighted Expert Activation Pruning)** for near-lossless expert compression (up to 50%).
- **GPU Resource Management**: `ydj` (Perpetual Motion Machine) for stable training on SCO ACP clusters.
- **Edge-First Design**: Optimized for multi-modal fusion on resource-constrained platforms.

---

## 📂 Project Structure

```text
FuseOmni/
├── ms-swift/          # 🚀 Model training & adaptation framework
├── reap/              # ✂️ REAP: Expert Pruning & Merging logic
├── ydj/               # ⚡ GPU Perpetual Motion & Job Management (SCO ACP)
├── models/            # 📦 Model checkpoints & configurations
├── eval/              # 📊 Model evaluation suite (Planned)
├── data/              # 🧪 Dataset preprocessing utilities (Planned)
└── README.md          # 📖 You are here
```

---

## 🛠️ Core Components

### 1. Model Training (`ms-swift`)
Powered by [ms-swift](https://github.com/modelscope/ms-swift), this component handles:
- **SFT (Supervised Fine-Tuning)** on multi-modal speech datasets.
- **LoRA/QLoRA** adaptation for parameter-efficient tuning.
- Support for diverse backbone architectures (Qwen, GLM, etc.).

### 2. Model Pruning (`reap`)
Implements **Router-weighted Expert Activation Pruning (REAP)**:
- Considers both **router gate-values** and **average activation norms** to select experts for pruning.
- Achieves significant memory reduction (e.g., 50% compression) with minimal accuracy degradation.
- Supports modern SMoE architectures like Qwen3-Coder and GLM-4.5.

### 3. GPU Task Management (`ydj`)
The **"Perpetual Motion Machine"** for SCO ACP clusters:
- **Dynamic Keep-Alive**: Prevents GPU idle recovery by maintaining >70% utilization.
- **Remote Submission**: Submit and manage jobs via `client.sh` without direct node access.
- **Queue System**: Supports prioritized task execution and log tracking.

---

## 🏗️ Roadmap

- [ ] **Model Evaluation**: Integrate automated benchmarks for speech and multi-modal tasks.
- [ ] **Dataset Preprocessing**: Unified pipeline for multi-modal data cleaning and tokenization.
- [ ] **Deployment**: Optimized inference kernels for edge-side NPU/GPU.

---

## 📥 Installation

```bash
# Clone the repository
git clone --recursive https://github.com/yangzy39/FuseOmni.git
cd FuseOmni

# Setup pruning environment (requires uv)
cd reap
bash scripts/build.sh
```