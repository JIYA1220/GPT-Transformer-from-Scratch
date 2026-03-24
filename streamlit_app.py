import streamlit as st
import torch
import tiktoken
import yaml
import os
from model.gpt_model import GPTModel
from train.trainer_advanced import (
    generate,
    text_to_token_ids,
    token_ids_to_text,
    clean_output
)

st.set_page_config(page_title="Custom GPT Storyteller", page_icon="🤖")

@st.cache_resource
def load_model():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    GPT_CONFIG = config["model"]
    device = torch.device("cpu")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    model = GPTModel(GPT_CONFIG)
    # Use gpt_model.pth if best_model is too big for upload
    model_file = "best_model.pth" if os.path.exists("best_model.pth") else "gpt_model.pth"
    
    if not os.path.exists(model_file):
        st.error(f"Model file {model_file} not found! Please upload it.")
        return None, None, None, None

    checkpoint = torch.load(model_file, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model, tokenizer, GPT_CONFIG, device

st.title("🤖 Custom LLM Storyteller")
st.markdown("This model was trained from scratch on the **TinyStories** dataset.")

model, tokenizer, config, device = load_model()

if model:
    with st.sidebar:
        st.header("Settings")
        temp = st.slider("Temperature (Creativity)", 0.1, 1.5, 0.8)
        top_k = st.slider("Top-K (Focus)", 1, 100, 40)
        max_tokens = st.slider("Max Tokens", 20, 200, 100)

    prompt = st.text_input("Enter a story starter:", "Once upon a time,")

    if st.button("Generate Story"):
        with st.spinner("Writing..."):
            input_ids = text_to_token_ids(prompt, tokenizer).to(device)
            output_ids = generate(
                model=model,
                idx=input_ids,
                max_new_tokens=max_tokens,
                context_size=config["context_length"],
                temperature=temp,
                top_k=top_k
            )
            
            full_text = token_ids_to_text(output_ids, tokenizer)
            # Remove the prompt from the output to show just the generation
            generated_text = full_text[len(prompt):]
            result = clean_output(generated_text)
            
            st.subheader("Your Story:")
            st.write(f"**{prompt}**{result}")
