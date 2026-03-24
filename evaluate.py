"""
Standalone evaluation script.
Run: python evaluate.py
Shows: Loss, Perplexity, Sample generations
"""

import torch
import tiktoken
import yaml
import math

from model.gpt_model import GPTModel
from model.dataloader import create_dataloader_v1
from train.trainer_advanced import (
    calc_loss_loader, generate,
    text_to_token_ids, token_ids_to_text, perplexity
)

# ── Load Config ───────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

GPT_CONFIG = config["model"]
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer  = tiktoken.get_encoding("gpt2")

# ── Load Model ────────────────────────────────
import os
model_file = "best_model.pth" if os.path.exists("best_model.pth") else "gpt_model.pth"
model      = GPTModel(GPT_CONFIG)
checkpoint = torch.load(model_file, map_location=device)

if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
    print(f"✅ Loaded checkpoint (Ep {checkpoint.get('epoch','?')}, "
          f"Val Loss: {checkpoint.get('val_loss', '?'):.4f})")
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

# ── Load Data ─────────────────────────────────
data_path = config["data"]["file_path"]
if os.path.exists(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
else:
    raw_text = "Once upon a time there was a little girl. " * 200

split    = int(0.9 * len(raw_text))
val_loader = create_dataloader_v1(
    raw_text[split:], batch_size=4,
    max_length=GPT_CONFIG["context_length"],
    stride=128, shuffle=False, drop_last=False
)

# ── Evaluate ──────────────────────────────────
print(f"\n{'='*45}")
print("  📊 Model Evaluation")
print(f"{'='*45}")

val_loss = calc_loss_loader(val_loader, model, device)
val_ppl  = perplexity(val_loss)

print(f"  Model File  : {model_file}")
print(f"  Val Loss    : {val_loss:.4f}")
print(f"  Perplexity  : {val_ppl:.2f}")
print(f"\n  💡 What is Perplexity?")
print(f"     = exp(loss) = how 'surprised' the model is")
print(f"     Lower is better.")
print(f"     Human level ≈ 20-50")
print(f"     Your model  ≈ {val_ppl:.0f}")

# ── Generate Samples ─────────────────────────
print(f"\n{'='*45}")
print("  ✍️  Sample Generations")
print(f"{'='*45}")

prompts = [
    "Once upon a time",
    "The little boy",
    "She smiled and said",
    "In a big forest",
]

for prompt in prompts:
    ids    = text_to_token_ids(prompt, tokenizer).to(device)
    output = generate(
        model, ids,
        max_new_tokens = 60,
        context_size   = GPT_CONFIG["context_length"],
        temperature    = 0.8,
        top_k          = 40
    )
    text = token_ids_to_text(output, tokenizer)
    print(f"\n  Prompt : '{prompt}'")
    print(f"  Output : {text}")