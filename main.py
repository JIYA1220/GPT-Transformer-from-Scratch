"""
LLM From Scratch — Main Training Script
Run: python main.py
"""

import torch
import tiktoken
import yaml
import os

from model.gpt_model import GPTModel
from model.dataloader import create_dataloader_v1
from train.trainer_advanced import (
    train_model_advanced,
    plot_losses,
    plot_lr_schedule,
    text_to_token_ids,
    token_ids_to_text,
    generate,
    perplexity,
    calc_loss_loader
)

# ── 1. Load Config ────────────────────────────
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

GPT_CONFIG   = config["model"]
TRAIN_CONFIG = {
    **config["training"],
    **config["paths"],
    "start_context": config["data"]["start_context"]
}

print("✅ Config loaded!")
print(f"   Norm    : {GPT_CONFIG['norm_type'].upper()}")
print(f"   Layers  : {GPT_CONFIG['n_layers']}")
print(f"   Heads   : {GPT_CONFIG['n_heads']}")
print(f"   Emb Dim : {GPT_CONFIG['emb_dim']}")

# ── 2. Device ─────────────────────────────────
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"\n💻 Device: {device}")

# ── 3. Load Data ──────────────────────────────
data_path = config["data"]["file_path"]

if os.path.exists(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print(f"✅ Dataset : {data_path}")
    print(f"   Size    : {len(raw_text):,} chars")
else:
    print(f"⚠️  {data_path} not found — using sample text")
    raw_text = (
        "Once upon a time there was a little girl named Lily. "
        "She loved to play in the garden with her friends. "
        "Every day she learned something new about the world. "
        "The sun was bright and the sky was blue. "
        "She was happy and kind to everyone she met. "
    ) * 300

# ── 4. Train / Val Split ──────────────────────
split     = int(TRAIN_CONFIG["train_ratio"] * len(raw_text))
train_txt = raw_text[:split]
val_txt   = raw_text[split:]

print(f"\n📊 Data Split:")
print(f"   Train : {len(train_txt):,} chars")
print(f"   Val   : {len(val_txt):,} chars")

train_loader = create_dataloader_v1(
    train_txt,
    batch_size = TRAIN_CONFIG["batch_size"],
    max_length = GPT_CONFIG["context_length"],
    stride     = TRAIN_CONFIG["stride"],
    shuffle    = True,
    drop_last  = True
)
val_loader = create_dataloader_v1(
    val_txt,
    batch_size = TRAIN_CONFIG["batch_size"],
    max_length = GPT_CONFIG["context_length"],
    stride     = TRAIN_CONFIG["stride"],
    shuffle    = False,
    drop_last  = False
)

print(f"   Train Batches : {len(train_loader)}")
print(f"   Val Batches   : {len(val_loader)}")

# ── 5. Model ──────────────────────────────────
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n🧠 Model:")
print(f"   Parameters : {total_params:,}")
print(f"   Norm Type  : {GPT_CONFIG['norm_type'].upper()}")
print(f"   Layers     : {GPT_CONFIG['n_layers']}")
print(f"   Heads      : {GPT_CONFIG['n_heads']}")
print(f"   Emb Dim    : {GPT_CONFIG['emb_dim']}")

# ── 6. Optimizer ──────────────────────────────
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr           = TRAIN_CONFIG["learning_rate"],
    weight_decay = 0.2
)

# ── 7. Train ──────────────────────────────────
tokenizer = tiktoken.get_encoding("gpt2")

(train_losses, val_losses,
 tokens_seen, all_lrs, all_steps) = train_model_advanced(
    model        = model,
    train_loader = train_loader,
    val_loader   = val_loader,
    optimizer    = optimizer,
    device       = device,
    cfg          = GPT_CONFIG,
    train_cfg    = TRAIN_CONFIG,
    tokenizer    = tokenizer
)

# ── 8. Save Final Model ───────────────────────
torch.save(model.state_dict(), TRAIN_CONFIG["model_save"])
print(f"💾 Final model saved: {TRAIN_CONFIG['model_save']}")

# ── 9. Plot Curves ────────────────────────────
import torch as _torch
epochs_seen = _torch.linspace(
    0, TRAIN_CONFIG["num_epochs"], len(train_losses)
).tolist()

plot_losses(
    epochs_seen, tokens_seen,
    train_losses, val_losses,
    save_path=TRAIN_CONFIG["loss_plot"]
)
plot_lr_schedule(
    all_steps, all_lrs,
    save_path=TRAIN_CONFIG["lr_plot"]
)

# ── 10. Final Evaluation ──────────────────────
print(f"\n{'='*45}")
print(f"  📊 Final Evaluation")
final_train = calc_loss_loader(train_loader, model, device)
final_val   = calc_loss_loader(val_loader,   model, device)
print(f"  Train Loss : {final_train:.4f}")
print(f"  Val Loss   : {final_val:.4f}")
print(f"  Train PPL  : {perplexity(final_train):.2f}")
print(f"  Val PPL    : {perplexity(final_val):.2f}")
print(f"{'='*45}")

# ── 11. Sample Generations ────────────────────
print(f"\n{'='*45}")
print("  ✍️  Sample Outputs")
print(f"{'='*45}")

model.eval()
prompts = [
    "Once upon a time",
    "The little dog",
    "She opened the door and",
    "In a magical forest",
]

for prompt in prompts:
    ids    = text_to_token_ids(prompt, tokenizer).to(device)
    output = generate(
        model, ids,
        max_new_tokens = 80,
        context_size   = GPT_CONFIG["context_length"],
        temperature    = 0.8,
        top_k          = 40,
    )
    print(f"\n📝 '{prompt}'")
    print(f"   {token_ids_to_text(output, tokenizer)}")

print(f"\n✅ All done! Check experiments/ for logs and plots.")