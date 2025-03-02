import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login



repo_name = "WarisJaima/dpo-model" 

# Load the fine-tuned model from Hugging Face
model = AutoModelForCausalLM.from_pretrained(repo_name)

# Load the GPT-2 tokenizer (since you didn’t modify it)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Ensure padding token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ Model successfully loaded from Hugging Face.")

# Load the fine-tuned model from Hugging Face
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained(repo_name)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model()

# Function to generate responses
def generate_response(prompt, max_tokens=100):
    """
    Generates a response from the fine-tuned model.

    Parameters:
    - prompt (str): The input text for the model.
    - max_tokens (int): The maximum number of new tokens to generate.

    Returns:
    - response (str): The model-generated response.
    """
    try:
        # Format input as a structured dialogue
        formatted_prompt = f"Human: {prompt}\n\nAssistant:"

        # Tokenize input
        input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,  # Controls response length
                temperature=0.7,  # Adds diversity (lower = more deterministic)
                top_p=0.9,  # Nucleus sampling (higher = more diverse)
                do_sample=True,  # Enables randomness for varied responses
                pad_token_id=tokenizer.eos_token_id,  # Handles padding properly
            )

        # Decode and clean response
        full_response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        response = full_response.replace(formatted_prompt, "").strip()

        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

# Streamlit UI
st.title(" Fine-Tuned GPT-2 Chatbot")
st.markdown("Enter a prompt below and the AI will generate a response.")

# User input
user_input = st.text_area("Your Prompt", "How can I improve my productivity?")

# Generation settings
col1, col2 = st.columns(2)
max_tokens = col1.slider("Max Tokens", 50, 300, 100)

# Generate response button
if st.button("Generate Response"):
    with st.spinner("Generating response..."):
        response = generate_response(user_input, max_tokens=max_tokens)
        st.success("Response Generated!")
        st.write(response)


