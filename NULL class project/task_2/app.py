import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import os

# Supported models for text embeddings
MODEL_OPTIONS = {
    "BERT (bert-base-uncased)": "bert-base-uncased",
    "DistilBERT (distilbert-base-uncased)": "distilbert-base-uncased",
    "CLIP Text Encoder (openai/clip-vit-base-patch32)": "openai/clip-vit-base-patch32"
}

st.title("Text Embedding Generator")

selected_model_label = st.selectbox("Select embedding model:", list(
    MODEL_OPTIONS.keys())) or "bert-base-uncased"
model_name = MODEL_OPTIONS[selected_model_label]


@st.cache_resource
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


tokenizer, model = load_model_and_tokenizer(model_name)

user_text = st.text_input("Enter a text description:")

if user_text:
    encoded_input = tokenizer(
        user_text, return_tensors="pt", padding=True, truncation=True
    )

    # Display input text info
    st.subheader("Tokenized Input")
    st.write("Input IDs:", encoded_input['input_ids'][0].tolist())
    tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
    st.write("Tokens:", tokens)

    with torch.no_grad():
        model_output = model(**encoded_input)

    # Use full token embeddings: last_hidden_state [batch, seq_len, embed_dim]
    embeddings = model_output.last_hidden_state

    st.subheader("Embedding Output")
    st.write(f"Embedding shape: {embeddings.shape}")
    st.write("Embedding for first token:", embeddings[0][0].tolist())

    # Save embeddings to file
    save_folder = "saved_embeddings"
    os.makedirs(save_folder, exist_ok=True)
    filename = os.path.join(save_folder, "embedding.pt")
    torch.save(embeddings.cpu(), filename)
    st.success(f"Full token embeddings saved to {filename}")
