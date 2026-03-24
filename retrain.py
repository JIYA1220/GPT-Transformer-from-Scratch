"""
Retrain the LLM on your own custom text data.
Put your text file in data/ folder and run this!
"""

import torch
import tiktoken
import yaml
import os
from model.gpt_model import GPTModel
from model.dataloader import create_dataloader_v1
from train.trainer import train_model_simple, plot_losses

# ── Load Config ──────────────────────────────────────────────
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

GPT_CONFIG = config["model"]

# ── Load YOUR custom text ──────────────────────────────────────
# 👇 Replace this with your own .txt file path!
TEXT_FILE = "data/my_text.txt"

try:
    with open(TEXT_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"✅ Loaded: {TEXT_FILE} ({len(text):,} characters)")
except FileNotFoundError:
    print(f"⚠️  File not found: {TEXT_FILE}")
    print("Using default sample text instead...")
    text = "The future of AI is bright. " * 500

# ── Setup ──────────────────────────────────────────────────
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")

split     = int(0.9 * len(text))
train_loader = create_dataloader_v1(text[:split], batch_size=4,
                                     max_length=GPT_CONFIG["context_length"], stride=128)
val_loader   = create_dataloader_v1(text[split:],  batch_size=4,
                                     max_length=GPT_CONFIG["context_length"], stride=128)

# ── Load existing model OR start fresh ─────────────────────────
model = GPTModel(GPT_CONFIG)

model_file = "best_model.pth" if os.path.exists("best_model.pth") else "gpt_model.pth"

if os.path.exists(model_file):
    checkpoint = torch.load(model_file, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    print(f"✅ Loaded existing model ({model_file}) — continuing training!")
else:
    print("🆕 Starting fresh training!")

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# ── Train ──────────────────────────────────────────────────
train_losses, val_losses, tokens_seen = train_model_simple(
    model=model, train_loader=train_loader, val_loader=val_loader,
    optimizer=optimizer, device=device, num_epochs=5,
    eval_freq=5, eval_iter=2,
    start_context="The future of AI",
    tokenizer=tokenizer
)

torch.save({
    "model_state": model.state_dict(),
    "config": GPT_CONFIG
}, "gpt_model.pth")
print("\n✅ Model saved to gpt_model.pth")

epochs_tensor = torch.linspace(0, 5, len(train_losses)).tolist()
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
