"""
Advanced Trainer — Complete Version with Early Stopping
Features:
  ✅ Cosine LR schedule with warmup
  ✅ Gradient clipping
  ✅ Perplexity tracking
  ✅ CSV experiment logging
  ✅ Best model checkpointing
  ✅ Early stopping (no more overfitting!)
  ✅ Loss + LR plots
  ✅ Beam search decoding
  ✅ INT8 Quantization
"""

import torch
import torch.nn as nn
import math
import csv
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ─────────────────────────────────────────────
# 1. Perplexity
# ─────────────────────────────────────────────

def perplexity(loss):
    return math.exp(min(loss, 20))


# ─────────────────────────────────────────────
# 2. Cosine LR Scheduler with Warmup
# ─────────────────────────────────────────────

def get_lr(step, warmup_steps, total_steps, max_lr, min_lr):
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (
        1 + math.cos(math.pi * progress)
    )


# ─────────────────────────────────────────────
# 3. Loss Utilities
# ─────────────────────────────────────────────

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch  = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits       = model(input_batch)
    return nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )


def calc_loss_loader(loader, model, device, num_batches=None):
    total_loss  = 0.0
    num_batches = num_batches or len(loader)
    num_batches = min(num_batches, len(loader))

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
            total_loss += calc_loss_batch(
                x, y, model, device
            ).item()
    model.train()
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader,
                   device, eval_iter):
    train_loss = calc_loss_loader(
        train_loader, model, device, eval_iter
    )
    val_loss = calc_loss_loader(
        val_loader, model, device, eval_iter
    )
    return train_loss, val_loss


# ─────────────────────────────────────────────
# 4. CSV Logger
# ─────────────────────────────────────────────

class ExperimentLogger:
    def __init__(self, path="experiments/log.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        with open(self.path, "w", newline="",
                  encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "step", "train_loss", "val_loss",
                "train_ppl", "val_ppl", "lr", "tokens_seen"
            ])
        print(f"📝 Logging to: {self.path}")

    def log(self, epoch, step, train_loss,
            val_loss, lr, tokens_seen):
        with open(self.path, "a", newline="",
                  encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, step,
                round(train_loss, 4),
                round(val_loss,   4),
                round(perplexity(train_loss), 2),
                round(perplexity(val_loss),   2),
                f"{lr:.2e}",
                tokens_seen
            ])


# ─────────────────────────────────────────────
# 5. Plotting
# ─────────────────────────────────────────────

def plot_losses(epochs_seen, tokens_seen,
                train_losses, val_losses,
                save_path="experiments/loss_plot.png"):
    os.makedirs("experiments", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_seen, train_losses,
             label="Train Loss", color="#00d4ff")
    ax1.plot(epochs_seen, val_losses,
             label="Val Loss", color="#ff6b6b",
             linestyle="--")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train vs Validation Loss")
    ax1.legend()
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    train_ppls = [perplexity(l) for l in train_losses]
    val_ppls   = [perplexity(l) for l in val_losses]
    ax2.plot(epochs_seen, train_ppls,
             label="Train PPL", color="#00d4ff")
    ax2.plot(epochs_seen, val_ppls,
             label="Val PPL", color="#ff6b6b",
             linestyle="--")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Perplexity")
    ax2.set_title("Train vs Validation Perplexity")
    ax2.legend()
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Loss plot saved: {save_path}")


def plot_lr_schedule(steps, lrs,
                     save_path="experiments/lr_plot.png"):
    os.makedirs("experiments", exist_ok=True)
    plt.figure(figsize=(8, 3))
    plt.plot(steps, lrs, color="#00d4ff")
    plt.xlabel("Steps")
    plt.ylabel("Learning Rate")
    plt.title("LR Schedule — Warmup + Cosine Decay")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 LR plot saved: {save_path}")


# ─────────────────────────────────────────────
# 6. Text Generation Utilities
# ─────────────────────────────────────────────

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(
        text, allowed_special={"<|endoftext|>"}
    )
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(
        token_ids.squeeze(0).tolist()
    )


def clean_output(text):
    text = text.replace("<|endoftext|>", "").strip()
    if len(text) > 500:
        text = text[:500]
    for p in [".", "!", "?"]:
        last = text.rfind(p)
        if last != -1 and last > len(text) // 3:
            return text[:last + 1].strip()
    return text.strip()


def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            logits = torch.where(
                logits < top_logits[:, -1],
                torch.tensor(float("-inf")).to(logits.device),
                logits
            )

        if temperature > 0.0:
            logits   = logits / temperature
            logits   = logits - logits.max(
                dim=-1, keepdim=True
            ).values
            probs    = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(
                logits, dim=-1, keepdim=True
            )

        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def generate_and_print_sample(model, tokenizer,
                               device, start_context, cfg):
    model.eval()
    token_ids = text_to_token_ids(
        start_context, tokenizer
    ).to(device)
    with torch.no_grad():
        output = generate(
            model, token_ids,
            max_new_tokens = 60,
            context_size   = cfg["context_length"],
            temperature    = 0.8,
            top_k          = 40
        )
    text = token_ids_to_text(output, tokenizer)
    print(text.replace("\n", " "))
    model.train()


# ─────────────────────────────────────────────
# 7. Beam Search Decoding
# ─────────────────────────────────────────────

def beam_search(model, idx, max_new_tokens, context_size,
                beam_width=5, temperature=1.0):
    device = idx.device
    beams  = [(0.0, idx)]

    for _ in range(max_new_tokens):
        all_candidates = []
        for score, seq in beams:
            seq_cond = seq[:, -context_size:]
            with torch.no_grad():
                logits = model(seq_cond)
            logits    = logits[:, -1, :] / max(temperature, 1e-8)
            log_probs = torch.log_softmax(logits, dim=-1)
            top_lp, top_tok = torch.topk(
                log_probs, beam_width, dim=-1
            )
            for i in range(beam_width):
                tok       = top_tok[0, i].unsqueeze(0).unsqueeze(0)
                new_seq   = torch.cat([seq, tok], dim=1)
                new_score = score + top_lp[0, i].item()
                all_candidates.append((new_score, new_seq))

        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:beam_width]

    return beams[0][1]


# ─────────────────────────────────────────────
# 8. INT8 Quantization
# ─────────────────────────────────────────────

def quantize_model(model):
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return quantized


def compare_model_sizes(original, quantized):
    import tempfile

    def get_size(m):
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pth"
        ) as f:
            torch.save(m.state_dict(), f.name)
            size = os.path.getsize(f.name) / (1024 * 1024)
            os.unlink(f.name)
        return size

    orig_sz  = get_size(original)
    quant_sz = get_size(quantized)
    print(f"\n📊 Quantization Results:")
    print(f"   Original  : {orig_sz:.1f} MB")
    print(f"   Quantized : {quant_sz:.1f} MB")
    print(f"   Reduction : {(1-quant_sz/orig_sz)*100:.1f}%")
    print(f"   Speedup   : ~4x on CPU!")
    return orig_sz, quant_sz


# ─────────────────────────────────────────────
# 9. Main Training Loop (with Early Stopping)
# ─────────────────────────────────────────────

def train_model_advanced(model, train_loader, val_loader,
                         optimizer, device, cfg,
                         train_cfg, tokenizer):
    """
    Full training loop with early stopping.
    Automatically stops when validation loss
    stops improving — prevents overfitting!
    """

    # ── Config ────────────────────────────────
    num_epochs    = train_cfg["num_epochs"]
    eval_freq     = train_cfg["eval_freq"]
    eval_iter     = train_cfg["eval_iter"]
    max_lr        = train_cfg["learning_rate"]
    min_lr        = train_cfg["min_lr"]
    max_grad_norm = train_cfg["max_grad_norm"]
    warmup_epochs = train_cfg["warmup_epochs"]
    start_context = train_cfg["start_context"]
    checkpoint    = train_cfg.get("best_model",
                                  "best_model.pth")
    log_path      = train_cfg.get("experiment_log",
                                  "experiments/log.csv")

    total_steps  = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * warmup_epochs
    logger       = ExperimentLogger(log_path)

    # ── Tracking variables ───────────────��────
    train_losses     = []
    val_losses       = []
    track_tokens     = []
    all_lrs          = []
    all_steps        = []
    tokens_seen      = 0
    global_step      = 0
    best_val_loss    = float("inf")

    # ── Early stopping ────────────────────────
    patience         = 3       # stop after 3 bad evals
    patience_counter = 0

    print(f"\n{'='*57}")
    print(f"  🚀 Advanced Training Started")
    print(f"  Total Steps    : {total_steps:,}")
    print(f"  Warmup Steps   : {warmup_steps:,}")
    print(f"  Norm Type      : {cfg.get('norm_type','layernorm').upper()}")
    print(f"  Grad Clip      : {max_grad_norm}")
    print(f"  LR Schedule    : {max_lr} → {min_lr} (cosine)")
    print(f"  Layers         : {cfg['n_layers']}")
    print(f"  Emb Dim        : {cfg['emb_dim']}")
    print(f"  Heads          : {cfg['n_heads']}")
    print(f"  Early Stopping : patience={patience}")
    print(f"{'='*57}\n")

    start_time   = time.time()
    stop_early   = False

    for epoch in range(num_epochs):
        if stop_early:
            break

        model.train()

        for input_batch, target_batch in train_loader:
            if stop_early:
                break

            # ── LR schedule ───────────────────
            lr = get_lr(
                global_step, warmup_steps,
                total_steps, max_lr, min_lr
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            all_lrs.append(lr)
            all_steps.append(global_step)

            # ── Forward + backward ────────────
            optimizer.zero_grad()
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()

            # ── Gradient clipping ─────────────
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm
            )
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            # ── Evaluate ──────────────────────
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader,
                    device, eval_iter
                )
                train_ppl = perplexity(train_loss)
                val_ppl   = perplexity(val_loss)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens.append(tokens_seen)

                logger.log(
                    epoch + 1, global_step,
                    train_loss, val_loss,
                    lr, tokens_seen
                )

                # ── Check improvement ─────────
                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    patience_counter = 0
                    is_best          = " ✅ BEST!"

                    # Save best model
                    torch.save({
                        "epoch"      : epoch + 1,
                        "step"       : global_step,
                        "model_state": model.state_dict(),
                        "optimizer"  : optimizer.state_dict(),
                        "val_loss"   : best_val_loss,
                        "val_ppl"    : val_ppl,
                        "config"     : cfg,
                    }, checkpoint)

                else:
                    patience_counter += 1
                    is_best = (
                        f" ⚠️  no improve "
                        f"{patience_counter}/{patience}"
                    )

                    # ── Early stopping check ──
                    if patience_counter >= patience:
                        print(
                            f"\n🛑 Early stopping! "
                            f"No improvement for "
                            f"{patience} evaluations."
                        )
                        print(
                            f"   Best Val Loss : "
                            f"{best_val_loss:.4f}"
                        )
                        print(
                            f"   Best Val PPL  : "
                            f"{perplexity(best_val_loss):.2f}"
                        )
                        stop_early = True
                        break

                print(
                    f"Ep {epoch+1:02d} | "
                    f"Step {global_step:05d} | "
                    f"LR {lr:.1e} | "
                    f"Loss {train_loss:.3f}/"
                    f"{val_loss:.3f} | "
                    f"PPL {train_ppl:.1f}/"
                    f"{val_ppl:.1f}"
                    f"{is_best}"
                )

        if not stop_early:
            # Sample after each epoch
            elapsed = (time.time() - start_time) / 60
            print(
                f"\n📝 Sample (Ep {epoch+1}) "
                f"[{elapsed:.1f} min]: ", end=""
            )
            generate_and_print_sample(
                model, tokenizer, device,
                start_context, cfg
            )
            print()

    # ── Final summary ─────────────────────────
    total_time = time.time() - start_time
    print(f"\n{'='*57}")
    print(f"  ✅ Training Complete!")
    print(f"  Time           : {total_time/60:.1f} minutes")
    print(f"  Best Val Loss  : {best_val_loss:.4f}")
    print(f"  Best Val PPL   : {perplexity(best_val_loss):.2f}")
    print(f"  Early Stopped  : {stop_early}")
    print(f"  Saved to       : {checkpoint}")
    print(f"  CSV Log        : {log_path}")
    print(f"{'='*57}\n")

    return (
        train_losses, val_losses,
        track_tokens, all_lrs, all_steps
    )