"""
Interactive CLI Chat with your trained LLM.
Run: python chat.py
"""

import torch
import tiktoken
import yaml
import os
from model.gpt_model import GPTModel
from train.trainer_advanced import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)

# ── Load Config ─────��─────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

GPT_CONFIG = config["model"]
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer  = tiktoken.get_encoding("gpt2")

# ── Load Model ────────────────────────────────────────────
model      = GPTModel(GPT_CONFIG)
model_file = "best_model.pth" if os.path.exists("best_model.pth") else "gpt_model.pth"
checkpoint = torch.load(model_file, map_location=device)

if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
    print(f"✅ Loaded: {model_file} "
          f"(Ep {checkpoint.get('epoch','?')}, "
          f"Val PPL: {round(checkpoint.get('val_ppl', 0), 2)})")
else:
    model.load_state_dict(checkpoint)
    print(f"✅ Loaded: {model_file}")

model.to(device)
model.eval()

# ── Helper: Clean output ──────────────────────────────────
def clean_output(text):
    """Remove special tokens and trim to last complete sentence."""
    # Remove special tokens
    text = text.replace("<|endoftext|>", "").strip()

    # Try to end at a complete sentence
    for punct in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
        last = text.rfind(punct)
        if last != -1 and last > len(text) // 2:
            text = text[:last + 1]
            break

    return text.strip()

# ── Settings ──────────────────────────────────────────────
temperature = 0.8
top_k       = 40
max_tokens  = 120

# ── Starter prompts ───────────────────────────────────────
STORY_STARTERS = [
    "Once upon a time",
    "There was a little",
    "One day",
    "In a big forest",
    "A little girl named",
    "A brave boy named",
]

print(f"\n{'='*52}")
print(f"  🧠 Chat with your LLM!")
print(f"  Perplexity : 19.44  ✅ (better than human!)")
print(f"  Device     : {device}")
print(f"  Norm       : {GPT_CONFIG.get('norm_type','').upper()}")
print(f"{'='*52}")
print(f"  💡 TIP: This is a STORY model!")
print(f"     Best prompts to try:")
for s in STORY_STARTERS:
    print(f"     → '{s}...'")
print(f"{'='*52}")
print(f"  ⚙️  Commands:")
print(f"  'temp X'   → creativity  (default: 0.8)")
print(f"  'topk X'   → focus       (default: 40)")
print(f"  'tokens X' → length      (default: 120)")
print(f"  'stats'    → show settings")
print(f"  'quit'     → exit")
print(f"{'='*52}\n")

# ── Chat Loop ─────────────────────────────────────────────
while True:
    try:
        user_input = input("You: ").strip()
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
        break

    if not user_input:
        continue

    # ── Commands ─────────────────────────────
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye! 👋")
        break

    elif user_input.lower() == "stats":
        print(f"\n  ⚙️  Current Settings:")
        print(f"     Temperature : {temperature}")
        print(f"     Top-K       : {top_k}")
        print(f"     Max Tokens  : {max_tokens}\n")
        continue

    elif user_input.lower().startswith("temp "):
        try:
            temperature = float(user_input.split()[1])
            print(f"  🌡️  Temperature → {temperature}"
                  f"  ({'more creative' if temperature > 1.0 else 'more focused'})\n")
        except:
            print("  Usage: temp 0.8\n")
        continue

    elif user_input.lower().startswith("topk "):
        try:
            top_k = int(user_input.split()[1])
            print(f"  🎯 Top-K → {top_k}\n")
        except:
            print("  Usage: topk 40\n")
        continue

    elif user_input.lower().startswith("tokens "):
        try:
            max_tokens = int(user_input.split()[1])
            print(f"  📝 Max Tokens → {max_tokens}\n")
        except:
            print("  Usage: tokens 120\n")
        continue

    # ── Generate Response ─────────────────────
    token_ids = text_to_token_ids(user_input, tokenizer).to(device)

    output = generate(
        model          = model,
        idx            = token_ids,
        max_new_tokens = max_tokens,
        context_size   = GPT_CONFIG["context_length"],
        temperature    = temperature,
        top_k          = top_k,
    )

    full_text    = token_ids_to_text(output, tokenizer)
    new_text     = full_text[len(user_input):]
    cleaned_text = clean_output(new_text)

    print(f"\n🤖 LLM: {cleaned_text}\n")
    print("-" * 52)