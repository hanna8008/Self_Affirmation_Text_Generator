# ----------------------------------------------------------------------------
# generate_inference_affirmation.py
# ----------------------------------------------------------------------------



# --- Imports ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from datetime import datetime
import os
import yaml



# --- Load Configuration ---
def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
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
            input_ids=inputs_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature = temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text



# --- Save Outputs to Log File ---
def save_output_log(prompt, output, emotion=None):
    os.makedirs("results/affirmations_generated", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #clean file-safe version of the prompt
    safe_prompt = prompt[:30].replace(" ", "_").replace("/", "_")
    safe_emotion = emotion.replace(" ", "_") if emotion else "none"

    filename = f"results/affirmations_generated/inference_{safe_prompt}_{safe_emotion}_{timestamp}.txt"

    with open(filename, "w") as f:
        f.write(f"[{datetime.now()}]\nPrompt: {prompt}\nOutput: {output}\n\n")

    print(f"Saved output to: {filename}")



# --- Main Function ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="outputs/checkpoints/checkpoint-389640", help="Path to model directory")
    parser.add_argument("--input", type=str, help="Input text prompt (journal entry, etc.)")
    parser.add_argument("--emotion", type=str, help="Optional emotion tag")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--log_output", action="store_true", help="Log output to results folder")
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer, model = load_model_and_tokenizer(args.model_dir)

    if args.input:
        prompt = format_input(args.input, args.emotion)
    else:
        raw = input("Hey, what's on your mind today? Type anything you're feeling: :")
        mood = input("Optional mood label: ")
        prompt = format_input(raw, mood if mood else None)

    output = generate_affirmation(prompt, tokenizer, model)

    print("\n Prompt: \n", prompt)
    print("\n Generated Affirmation \n", output)

    if args.log_output:
        save_output_log(prompt, output, args.emotion)



# --- main ---
if __name__ == "__main__":
    main()