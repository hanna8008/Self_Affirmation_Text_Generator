# ----------------------------------------------------------------------------
# gui.py
# ----------------------------------------------------------------------------
# 
# This script launches a Gradio web app for real-time affirmation generation. It
# loads a trained GPT-2 model, formats user input (with optional emotion), generates
# an affirmation using beam sampling, and displays it via a clean user interface. It 
# also logs outputs with timestamps for reproducibility and user insight.



# --- Imports ---
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import yaml
import os
from datetime import datetime
import re



# --- Load Configuration ---
def load_config(path="configs/config.yaml"):
    #load model/training config as a Python dictionary
    with open(path, "r") as f:
        return yaml.safe_load(f)



# --- Load Model and Tokenizer ---
def load_model_and_tokenizer(model_dir):
    #log which model is being loaded
    print(f"Loading model from: {model_dir}")
    #load tokenizer from saved model directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    #load trained GPT-2 model
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    #set model to evaluation (non-training) mode
    model.eval()
    return tokenizer, model



# --- Format Input (emotion conditioning optional)
def format_input(text, emotion=None):
    #prepend optional emotion tag if provided, formatted like [SAD]
    if emotion: 
        return f"[{emotion.upper()}] {text.strip()}"
    #otherwise, return stripped plain input
    return text.strip()




# --- Clean Text of Unnecessary Symbols ---
def clean_text(text):
    #remove URLs
    text = re.sub(r"http\S+", "", text)
    #remove hashtags and mentions       
    text = re.sub(r"[@#]\w+", "", text)
    #remove any content inside square brackets (e.g., [SAD])         
    text = re.sub(r"\[.*?\]", "", text)
    #keep only alphanumerics, basic punctuation
    text = re.sub(r"[^\w\s.,!?'\-]", "", text)
    #collapse multiple spaces into a single space 
    text = re.sub(r"\s+", " ", text).strip()    

    #normalize multiple periods of newlines
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\n+", " ", text)

    #split the cleaned text into rough sentence segments
    sentences = re.split(r'[.!?]', text)
    #remove empty strings
    senteces = [s.strip() for s in sentences if s.strip()]

    #define phrases that indicate toxicity or negativity
    forbidden_phrases = [
        "why can't i", "why do they", "everyone else", "nobody", "hate myself",
        "ugly", "fat", "kill", "better than", "they all", "no one", "not loved"
    ]

    #keep only meaningful, safe lines within >2 words and no toxic content
    filtered = [
        s for s in sentences
        if len(s.split()) > 2 and not any(phrase in s.lower() for phrase in forbidden_phrases)
    ]

    #join up to 3 filtered lines back into a single string
    return ". ".join(filtered[:3]) + "." if filtered else text.strip()




# --- Generate Affirmation ---
def generate_affirmation(prompt, tokenizer, model, max_length=150, temperature=0.6, top_k=50, top_p=0.95):
    #tokenize the input prompt into tensor format
    inputs = tokenizer(prompt, return_tensors="pt")
    #move input IDs to same device as model
    inputs_ids = inputs.input_ids.to(model.device)
    #same for attention mask, move mask IDs to same device as model
    attention_mask = inputs.attention_mask.to(model.device)

    #disable gradient calculation for efficiency
    with torch.no_grad():
        output = model.generate(
            input_ids=inputs_ids,
            attention_mask=attention_mask,
            #total length of input + output tokens
            max_length=max_length,
            #adds randomness (lower = more deterministic)
            temperature = temperature,
            #limit sampling pool to top-k highest-probability words
            top_k=top_k,
            #nucleus sampling: only sample from top-p cumulative prob
            top_p=top_p,
            #enables sampling instea of greedy decoding
            do_sample=True,
            #prevent repetitive phrases (e.g., "I am I am")
            no_repeat_ngram_size=3,
            #stop when end-of-sequence token is generated
            eos_token_id=tokenizer.eos_token_id,
            #used for alignment if needed
            pad_token_id=tokenizer.pad_token_id
        )

    #decode full generated text and extract only the new portion (not including prompt)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    prompt_text = tokenizer.decode(inputs_ids[0], skip_special_tokens=True)
    final_generated_affirmation = generated_text[len(prompt_text):].strip()

    #clean the generated test for grammar, safety, readability
    final_generated_affirmation = clean_text(final_generated_affirmation)

    #return polished output
    return final_generated_affirmation



# --- Interface Wrapper ---
def affirmation(journal_text, emotion_tag):
    #format user input
    prompt = format_input(journal_text, emotion_tag)
    #generate affirmation
    generate = generate_affirmation(prompt, tokenizer, model)

    #create directory for logs if not present
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #create file-safe preview for file naming
    safe_prompt = journal_text[:30].replace(" ", "_").replace("/", "_")
    safe_emotion = emotion_tag.replace(" ", "_") if emotion_tag else "none"

    #save output to results folder with timestamp
    filename = f"results/affirmations_generated/inference_{safe_prompt}_{safe_emotion}_{timestamp}.txt"

    with open(filename, "w") as f:
        f.write(f"[{datetime.now()}]\nPrompt: {prompt}\nOutput: {generate}\n\n")

    #return generated affirmation to the UI
    return generate



# --- Initialize ---
#load settings from config.yaml
config = load_config()
#load model from output_dir
tokenizer, model = load_model_and_tokenizer(config["output_dir"])



# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="pink"))as interface:
    gr.Markdown("""
    # *Mirror Me: Affirmation Generator*  
    *Enter how you're feeling and receive a personalized affirmation.*
    """)

    #organize input components vertically
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
        #button to trigger generation
        submit_button = gr.Button("Submit")

        output_box = gr.Textbox(
            lines=3,
            label="Affirmation",
            placeholder="Your affirmation will appear here",
            elem_id="output_box"
        )
    
    #wrapper function for click logic
    def wrapped_affirmation(journal_text, emotion_tag):
        return affirmation(journal_text, emotion_tag)

    #connect button to input/output logic
    submit_button.click(fn=wrapped_affirmation, inputs=[journal_input, emotion_input], outputs=output_box)




# --- main ---
if __name__ == "__main__":
    #launch the web interface
    interface.launch(share=True, favicon_path="static/heart_icon.png")
    #share=True allows public access to a temporary Gradio-hosted version