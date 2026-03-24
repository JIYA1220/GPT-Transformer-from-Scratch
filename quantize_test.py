"""
Test INT8 Quantization.
Run: python quantize_test.py
Shows speed comparison before/after quantization.
"""

import torch, yaml, time, tiktoken
from model.gpt_model import GPTModel
from train.trainer_advanced import (
    generate, quantize_model, compare_model_sizes,
    text_to_token_ids, token_ids_to_text
)

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

GPT_CONFIG = config["model"]
device     = torch.device("cpu")
tokenizer  = tiktoken.get_encoding("gpt2")

# Load model
model = GPTModel(GPT_CONFIG)
import os
model_file = "best_model.pth" if os.path.exists("best_model.pth") else "gpt_model.pth"
ckpt  = torch.load(model_file, map_location=device)
if isinstance(ckpt, dict) and "model_state" in ckpt:
    model.load_state_dict(ckpt["model_state"])
else:
    model.load_state_dict(ckpt)
model.eval()

prompt    = "Once upon a time there was a little"
token_ids = text_to_token_ids(prompt, tokenizer)

print("\n" + "="*50)
print("  ⚡ Quantization Benchmark")
print("="*50)

# ── Test original ─────────────────────────────
print("\n📦 Original Model (FP32):")
t0 = time.time()
for _ in range(3):
    out = generate(model, token_ids, 50,
                   GPT_CONFIG["context_length"], 0.8, 40)
orig_time = (time.time() - t0) / 3
print(f"   Avg time : {orig_time:.2f}s")
print(f"   Output   : {token_ids_to_text(out, tokenizer)[:80]}...")

# ── Quantize ──────────────────────────────────
print("\n⚡ Quantizing to INT8...")
q_model = quantize_model(model)
compare_model_sizes(model, q_model)

# ── Test quantized ────────────────────────────
print("\n⚡ Quantized Model (INT8):")
t0 = time.time()
for _ in range(3):
    out = generate(q_model, token_ids, 50,
                   GPT_CONFIG["context_length"], 0.8, 40)
quant_time = (time.time() - t0) / 3
print(f"   Avg time : {quant_time:.2f}s")
print(f"   Speedup  : {orig_time/quant_time:.1f}x faster!")
print(f"   Output   : {token_ids_to_text(out, tokenizer)[:80]}...")

print("\n" + "="*50)
print(f"  ✅ Speedup: {orig_time/quant_time:.1f}x faster!")
print(f"  💡 Use in app.py: USE_QUANTIZATION = True")
print("="*50)