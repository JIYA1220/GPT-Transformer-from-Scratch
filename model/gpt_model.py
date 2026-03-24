# Source: rasbt/LLMs-from-scratch
# https://github.com/rasbt/LLMs-from-scratch

import torch
import torch.nn as nn
from model.attention import MultiHeadAttention


# ─────────────────────────────────────────────
# Normalization Layers
# ─────────────────────────────────────────────

class LayerNorm(nn.Module):
    """Original LayerNorm from GPT-2."""
    def __init__(self, emb_dim):
        super().__init__()
        self.eps   = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean   = x.mean(dim=-1, keepdim=True)
        var    = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class RMSNorm(nn.Module):
    """
    RMSNorm — used in LLaMA, Mistral, and most modern LLMs.
    Faster than LayerNorm (no mean subtraction).
    Formula: x / RMS(x) * scale
    """
    def __init__(self, emb_dim, eps=1e-5):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))

    def forward(self, x):
        # Root Mean Square
        rms    = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.scale * x_norm


def get_norm(norm_type, emb_dim):
    """Factory: returns LayerNorm or RMSNorm based on config."""
    if norm_type == "rmsnorm":
        return RMSNorm(emb_dim)
    return LayerNorm(emb_dim)


# ─────────────────────────────────────────────
# Activation
# ─────────────────────────────────────────────

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# ─────────────────────────────────────────────
# Feed Forward
# ───────────────────────────────────────────��─

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Dropout(cfg["drop_rate"]),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            nn.Dropout(cfg["drop_rate"]),
        )

    def forward(self, x):
        return self.layers(x)


# ─────────────────────────────────────────────
# Transformer Block
# ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        norm_type = cfg.get("norm_type", "layernorm")

        self.att        = MultiHeadAttention(
            d_in=cfg["emb_dim"], d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.ff         = FeedForward(cfg)
        self.norm1      = get_norm(norm_type, cfg["emb_dim"])
        self.norm2      = get_norm(norm_type, cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention + residual
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut

        # FFN + residual
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x


# ─────────────────────────────────────────────
# GPT Model
# ─────────────────────────────────────────────

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        norm_type = cfg.get("norm_type", "layernorm")

        self.tok_emb  = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb  = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = get_norm(norm_type, cfg["emb_dim"])
        self.out_head   = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        return self.out_head(x)


def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        idx      = torch.cat((idx, idx_next), dim=1)
    return idx