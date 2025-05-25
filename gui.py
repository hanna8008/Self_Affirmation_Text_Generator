# ----------------------------------------------------------------------------
# gui.py
# ----------------------------------------------------------------------------

# --- Imports ---
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml



# --- Load Configuration ---
def load_config(path="configs/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)



# --- Load Model and Tokenizer ---
def load_model_and_tokenizer(model_dir):
    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model



# --- Format Input (emotion conditioning optional)
def format_input(text, emotion=None):
    if emotion: 
        return f"[{emotion.upper()}] {text.strip()}"
    return text.strip()



 # --- Generate Affirmation ---
 def generate_affirmation(prompt, tokenizer, model, max_length=100, temperature=0.8, top_k=50, top_p=0.95):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    with torch.no_grad():
        output = model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_length = max_length,
            temperature = temperature,
            top_k = top_k,
            top_p = top_p,
            do_sample = True,
            pad_token_id = tokenizer.pad_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text



# --- Interface Wrapper ---
def affirmation(journal_text, emotion_tag):
    prompt = format_input(journal_text, emotion_tag)
    generate = generate_affirmation(prompt, tokenizer, model)
    return generate



# --- Initialize ---
config = load_config()
tokenizer, model = load_model_and_tokenizer(config["output_dir"])



# --- Gradio UI ---
interface = gr.Interface(
    fn = affirmation,
    inputs = [
        gr.Textbox(label = "What's on your mind?", placeholder = "Typeyour thoughts here...", lines=4),
        gr.Textbox(label = "Optional Emotion Tag", placeholder = "e.g. love, happiness, gratitude (optional)")
    ],
    outputs = "text"
    title = "Mirror Me: Affirmation Generator", 
    description = "Enter how you're feeling and receive a personalized affirmation."
)



# --- main ---
if __name__ == "__main__":
    interface.launch()