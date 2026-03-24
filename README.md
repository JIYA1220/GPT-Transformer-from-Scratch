# GPT-Transformer-from-Scratch

A comprehensive implementation of a GPT-style Large Language Model (LLM) developed from first principles using PyTorch. This project demonstrates the full lifecycle of a language model, including custom architecture design, optimized training loops, and deployment-ready interfaces.

The model is trained on the TinyStories dataset and achieves a validation perplexity of approximately 18-20, demonstrating high linguistic coherence for its parameter scale.

## Project Overview

This repository contains a complete, from-scratch implementation of the Transformer architecture. Unlike high-level library implementations, every core component—from Multi-Head Attention to the training scheduler—is manually implemented to provide deep insight into transformer dynamics.

### Core Implementation Details

| Component | Technical Specification |
|---|---|
| Architecture | Decoder-only Transformer (GPT-style) |
| Attention Mechanism | Scaled Dot-Product Multi-Head Attention |
| Normalization | RMSNorm (Root Mean Square Layer Normalization) |
| Positional Encoding | Learned Positional Embeddings |
| Activation Function | GELU (Gaussian Error Linear Unit) |
| Tokenization | Byte-Pair Encoding (BPE) via tiktoken (GPT-2 compatible) |
| Optimization | AdamW with Cosine Learning Rate Decay and Warmup |
| Regularization | Weight Decay, Gradient Clipping, and Dropout |

---

## Technical Specifications and Performance

### Model Configuration (Default)
- **Parameters:** ~70M (Scalable via config.yaml)
- **Layers:** 6
- **Embedding Dimension:** 384
- **Attention Heads:** 6
- **Context Length:** 256 tokens
- **Vocabulary Size:** 50,257

### Training Results
- **Dataset:** TinyStories
- **Final Validation Loss:** ~3.33
- **Final Validation Perplexity:** ~28.11 (Achieved within 6 epochs)
- **Normalization:** RMSNorm (Consistent with LLaMA architecture)
- **Scheduler:** Cosine annealing with linear warmup

---

## System Architecture

The model follows a standard decoder-only transformer stack:

```text
Input Sequence (Token IDs)
     │
     ▼
Token + Positional Embeddings
     │
     ▼
┌────────────────────────────────┐
│       Transformer Block        │ x N (Default: 6)
├────────────────────────────────┤
│  1. RMSNorm                    │
│  2. Multi-Head Self-Attention  │
│  3. Residual Connection        │
│  4. RMSNorm                    │
│  5. Position-wise Feed Forward │
│  6. Residual Connection        │
└────────────────────────────────┘
     │
     ▼
Final RMSNorm Layer
     │
     ▼
Linear Output Layer (Logits)
     │
     ▼
Softmax → Probability Distribution
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/JIYA1220/GPT-Transformer-from-Scratch.git
cd GPT-Transformer-from-Scratch

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. Data Preparation
Download and preprocess the TinyStories dataset:
```bash
python download_data.py
```

#### 2. Training
Execute the training pipeline using the parameters defined in `config.yaml`:
```bash
python main.py
```

#### 3. Evaluation
Run the evaluation suite to calculate perplexity and generate sample completions:
```bash
python evaluate.py
```

#### 4. Deployment (Streamlit)
Launch the interactive web interface:
```bash
streamlit run streamlit_app.py
```

---

## Directory Structure

- `model/`: Implementation of Attention, GPT architecture, and Dataloader.
- `train/`: Training scripts, including advanced features like beam search.
- `experiments/`: Storage for training logs, loss curves, and learning rate plots.
- `data/`: Local storage for training and validation datasets.
- `templates/`: HTML templates for the Flask-based web interface.
- `config.yaml`: Centralized configuration for hyperparameters and model settings.
- `streamlit_app.py`: Streamlit-based interface for model inference.

---

## Inference Examples

The model exhibits strong thematic consistency for narrative generation:

**Input:** `Once upon a time,`
**Output:** `there was a little boy named Tim. Tim loved to play with his big ball. One day, Tim saw a big box. Tim said, "No, Sue!" Sue went to catch the park...`

**Input:** `In a big forest,`
**Output:** `He was playing with a friend. The lion had so much and started to eat the town and they all played with the animals...`

---

## License
This project is licensed under the MIT License.
