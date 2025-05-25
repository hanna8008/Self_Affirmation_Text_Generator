# ----------------------------------------------------------------------------
# gui.py
# ----------------------------------------------------------------------------



# --- Imports ---
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import os
from datetime import datetime



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
            input_ids = inputs_ids,
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

    #save to results/affirmations_generated/inference_gui_log
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #clean file-safe version of the prompt
    safe_prompt = journal_text[:30].replace(" ", "_").replace("/", "_")
    safe_emotion = emotion_tag.replace(" ", "_") if emotion_tag else "none"

    filename = f"results/affirmations_generated/inference_{safe_prompt}_{safe_emotion}_{timestamp}.txt"

    with open(filename, "w") as f:
        f.write(f"[{datetime.now()}]\nPrompt: {prompt}\nOutput: {generate}\n\n")

    return generate



# --- Initialize ---
config = load_config()
tokenizer, model = load_model_and_tokenizer(config["output_dir"])



# --- Gradio UI ---
"""interface = gr.Interface(
    fn = affirmation,
    inputs = [
        gr.Textbox(label = "What's on your mind?", placeholder = "Type your thoughts here...", lines=4),
        gr.Textbox(label = "Optional Emotion Tag", placeholder = "e.g. love, happiness, gratitude (optional)")
    ],
    outputs = "text",
    title = "Mirror Me: Affirmation Generator", 
    description = "Enter how you're feeling and receive a personalized affirmation."
)"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="pink"))as interface:
    gr.Markdown("""
    # *Mirror Me: Affirmation Generator*  
    *Enter how you're feeling and receive a personalized affirmation.*
    """)

    with gr.Column():
        journal_input = gr.Textbox(
            lines = 4,
            label="What's on your mind?",
            placeholder="Type your thoughts here...",
            elem_id="journal_input"
        )
        emotion_input = gr.Textbox(
            label="Optional Emotion Tag",
            placeholder="worry, sad, stress, anger, fear",
            elem_id="emotion_input"
        )
        submit_button = gr.Button("Submit")

        output_box = gr.Textbox(
            lines=3,
            label="Affirmation",
            placeholder="Your affirmation will appear here",
            elem_id="output_box"
        )
    
    def wrapped_affirmation(journal_text, emotion_tag):
        return affirmation(journal_text, emotion_tag)

    submit_button.click(fn=wrapped_affirmation, inputs=[journal_input, emotion_input], outputs=output_box)




# --- main ---
if __name__ == "__main__":
    interface.launch(share=True, favicon_path="static/heart_icon.png")