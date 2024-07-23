import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
from zipfile import ZipFile

# Streamlit UI for uploading model
st.title("Text Summarizer")
uploaded_file = st.file_uploader("bart-base.zip", type="zip")

if uploaded_file is not None:
    # Extract the uploaded zip file
    with ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall("model_directory")

    # Load the model from the extracted directory
    try:
        model_path = "model_directory"
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

# Text area for input
text = st.text_area("Enter the text to generate its Summary:")

# Configuration for generation
generation_config = {'max_length': 100, 'do_sample': True, 'temperature': 0.7}

if text:
    try:
        # Encode input
        inputs_encoded = tokenizer(text, return_tensors='pt')

        # Generate output
        with torch.no_grad():
            model_output = model.generate(inputs_encoded["input_ids"], **generation_config)[0]

        # Decode output
        output = tokenizer.decode(model_output, skip_special_tokens=True)

        # Display results
        with st.expander("Output", expanded=True):
            st.write(output)

    except Exception as e:
        st.error(f"An error occurred during summarization: {e}")

