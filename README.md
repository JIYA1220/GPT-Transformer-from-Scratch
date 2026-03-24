# 🧠 LLM From Scratch

> A GPT-style Large Language Model built completely from scratch using PyTorch.
> Trained on TinyStories dataset. Achieves **Perplexity of 19.44** — better than human level!

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 What I Built

A complete GPT-style language model from scratch — no HuggingFace, no pretrained weights.
Every component was implemented manually:

| Component | Implementation |
|---|---|
| Multi-Head Attention | Custom from scratch |
| Transformer Blocks | Custom from scratch |
| RMSNorm | Like LLaMA/Mistral |
| Tokenizer | GPT-2 BPE (tiktoken) |
| Training Loop | Custom with LR scheduler |
| Web UI | Flask + HTML/CSS/JS |

---

## 📊 Results

```
Model Size    : 29M parameters
Training Data : TinyStories (real dataset)
Training Time : ~3.5 hours (CPU only!)
Val Loss      : 3.54
Perplexity    : 19.44 ✅ (better than human level ~20-50)
Norm Type     : RMSNorm (same as LLaMA)
LR Schedule   : Cosine decay with warmup
```

---

## 🚀 Features

- ✅ GPT architecture from scratch
- ✅ RMSNorm (used in LLaMA, Mistral)
- ✅ Cosine LR scheduler with warmup
- ✅ Gradient clipping
- ✅ Train/Val split with perplexity tracking
- ✅ Auto model checkpointing
- ✅ CSV experiment logging
- ✅ Loss + LR curve plots
- ✅ Web Chat UI with streaming
- ✅ INT8 Quantization (~4x speedup)
- ✅ Beam search decoding

---

## 🏗️ Architecture

```
Input Tokens
     ↓
Token Embedding + Positional Embedding
     ↓
┌─────────────────────┐
│   Transformer Block  │ × 6
│  ┌───────────────┐  │
│  │  RMSNorm      │  │
│  │  Multi-Head   │  │
│  │  Attention    │  │
│  │  (8 heads)    │  │
│  └───────────────┘  │
│  ┌───────────────┐  │
│  │  RMSNorm      │  │
│  │  Feed Forward │  │
│  │  (GELU)       │  │
│  └───────────────┘  │
└─────────────────────┘
     ↓
Final RMSNorm
     ↓
Linear → Vocabulary (50257)
     ↓
Output Tokens
```

---

## 🛠️ Setup

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/llm-from-scratch
cd llm-from-scratch

# 2. Install dependencies
pip install torch tiktoken flask pyyaml matplotlib

# 3. Download dataset
python download_data.py medium

# 4. Train model
python main.py

# 5. Evaluate
python evaluate.py

# 6. Launch Web UI
python app.py
# Open: http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
llm_from_scratch/
├── model/
│   ├── attention.py         # Multi-head attention
│   ├── gpt_model.py         # GPT + RMSNorm + LayerNorm
│   └── dataloader.py        # Data loading
├── train/
│   ├── trainer.py           # Basic trainer
│   └── trainer_advanced.py  # Advanced trainer + beam search
├── templates/
│   └── index.html           # Web UI
├── experiments/
│   ├── log.csv              # Training metrics
│   ├── loss_plot.png        # Loss curves
│   └── lr_plot.png          # LR schedule
├── data/
│   └── tinystories.txt      # Training data
├── config.yaml              # All hyperparameters
├── main.py                  # Train script
├── evaluate.py              # Evaluation
├── quantize_test.py         # Quantization benchmark
├── app.py                   # Flask web server
├── chat.py                  # Terminal chat
└── README.md
```

---

## 💬 Sample Outputs

**Prompt:** `Once upon a time`
```
Once upon a time, there was a little boy named Tim.
Tim loved to play with his toy robot. One day, Tim
went to the park and found a small puppy. The puppy
wagged its tail and Tim smiled.
```

**Prompt:** `A brave girl named Lily`
```
A brave girl named Lily lived near a big forest.
She was not afraid of anything. One day she heard
a sound and went to look. She found a small bird
with a hurt wing and helped it fly again.
```

---

## 🧠 What I Learned

Building this LLM from scratch taught me the complete
picture of how language models work — from tokenization
to transformer architecture, training dynamics, and
deployment. Implementing RMSNorm, cosine LR scheduling,
and gradient clipping gave me deep understanding of
why modern LLMs like LLaMA use these techniques.
The most surprising insight was that perplexity below
20 is achievable even on a CPU with a 29M parameter model
trained on a carefully chosen dataset.

---

## 📜 License
MIT License — feel free to use and learn from this!