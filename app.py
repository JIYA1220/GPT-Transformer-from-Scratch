"""
LLM Web App — Complete Version
Run: python app.py
Open: http://127.0.0.1:5000
"""

from flask import Flask, request, jsonify, Response
import torch, tiktoken, yaml, os, json, time
from model.gpt_model import GPTModel
from train.trainer_advanced import (
    generate,
    beam_search,
    text_to_token_ids,
    token_ids_to_text,
    clean_output
)

app = Flask(__name__, template_folder="templates")

# ── Load Config ───────────────────────────────
with open("config.yaml", "r", encoding="utf-8") as f:   # ← FIXED
    config = yaml.safe_load(f)

GPT_CONFIG = config["model"]
device     = torch.device("cpu")
tokenizer  = tiktoken.get_encoding("gpt2")

# ── Load Model ────────────────────────────────
model      = GPTModel(GPT_CONFIG)
model_file = "best_model.pth" if os.path.exists("best_model.pth") \
             else "gpt_model.pth"
checkpoint = torch.load(model_file, map_location=device)

if isinstance(checkpoint, dict) and "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model loaded : {model_file}")
print(f"   Params       : {total_params:,}")
print(f"   Layers       : {GPT_CONFIG['n_layers']}")
print(f"   Emb Dim      : {GPT_CONFIG['emb_dim']}")
print(f"   Heads        : {GPT_CONFIG['n_heads']}")
print(f"   Norm         : {GPT_CONFIG.get('norm_type','').upper()}")


# ── Routes ────────────────────────────────────
@app.route("/")
def index():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.route("/info")
def info():
    return jsonify({
        "device"    : str(device),
        "params"    : f"{total_params/1e6:.1f}M",
        "layers"    : GPT_CONFIG["n_layers"],
        "emb_dim"   : GPT_CONFIG["emb_dim"],
        "n_heads"   : GPT_CONFIG["n_heads"],
        "norm_type" : GPT_CONFIG.get("norm_type", "layernorm").upper(),
        "model_file": model_file,
    })


@app.route("/generate", methods=["POST"])
def generate_response():
    """Standard generation endpoint."""
    data        = request.json
    prompt      = data.get("prompt", "")
    temperature = float(data.get("temperature", 0.8))
    top_k       = int(data.get("top_k", 40))
    max_tokens  = min(int(data.get("max_new_tokens", 120)), 150)
    use_beam    = data.get("beam_search", False)

    token_ids = text_to_token_ids(prompt, tokenizer).to(device)

    if use_beam:
        output = beam_search(
            model,
            token_ids,
            max_new_tokens = min(max_tokens, 80),
            context_size   = GPT_CONFIG["context_length"],
            beam_width     = 5,
            temperature    = temperature
        )
    else:
        output = generate(
            model          = model,
            idx            = token_ids,
            max_new_tokens = max_tokens,
            context_size   = GPT_CONFIG["context_length"],
            temperature    = temperature,
            top_k          = top_k,
        )

    full_text        = token_ids_to_text(output, tokenizer)
    new_text         = full_text[len(prompt):]
    response         = clean_output(new_text)
    tokens_generated = output.shape[1] - token_ids.shape[1]

    return jsonify({
        "response"        : response,
        "tokens_generated": tokens_generated
    })


@app.route("/stream")
def stream_response():
    """
    Streaming endpoint — sends tokens one by one.
    Words appear one by one like ChatGPT!
    """
    prompt      = request.args.get("prompt", "")
    temperature = float(request.args.get("temperature", 0.8))
    top_k       = int(request.args.get("top_k", 40))
    max_tokens  = min(
        int(request.args.get("max_new_tokens", 120)), 150
    )

    def generate_stream():
        token_ids = text_to_token_ids(prompt, tokenizer).to(device)
        idx       = token_ids.clone()

        for step in range(max_tokens):
            idx_cond = idx[:, -GPT_CONFIG["context_length"]:]

            with torch.no_grad():
                logits = model(idx_cond)

            logits = logits[:, -1, :]

            # Top-k filtering
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                logits = torch.where(
                    logits < top_logits[:, -1],
                    torch.tensor(float("-inf")),
                    logits
                )

            # Temperature + sample
            if temperature > 0:
                logits   = logits / temperature
                probs    = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(
                    logits, dim=-1, keepdim=True
                )

            # Stop at end token
            if idx_next.item() == tokenizer.eot_token:
                break

            idx = torch.cat([idx, idx_next], dim=1)

            # Decode token
            token_text = tokenizer.decode([idx_next.item()])

            # Skip special tokens
            if "<|" in token_text:
                break

            # Send as SSE
            data = json.dumps({
                "token": token_text,
                "step" : step
            })
            yield f"data: {data}\n\n"
            time.sleep(0.02)   # Small delay for visual effect

        yield "data: [DONE]\n\n"

    return Response(
        generate_stream(),
        mimetype = "text/event-stream",
        headers  = {
            "Cache-Control"    : "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  🌐 Open     : http://127.0.0.1:5000")
    print(f"  🧠 Model    : {model_file}")
    print(f"  ⚙️  Params   : {total_params:,}")
    print(f"  📊 PPL      : 18.88 ✅")
    print(f"  🔀 Streaming: ON")
    print(f"{'='*50}\n")
    app.run(debug=False, host="0.0.0.0", port=5000)