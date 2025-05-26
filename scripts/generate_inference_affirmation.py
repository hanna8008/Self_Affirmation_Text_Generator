# ----------------------------------------------------------------------------
# generate_inference_affirmation.py
# ----------------------------------------------------------------------------
#
# This script loads a fine-tuned GPT-2 model and generates a positive affirmation
# in response to user input, optionally condiitoned on emotion tags. It supports
# real-time prompting, infernece hyperparameteres, cleaning outputs, and optional
# logging of generated affirmations to disk.



# --- Imports ---
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from datetime import datetime
import os
import re
import yaml



# --- Load Configuration ---
def load_config(config_path="configs/config.yaml"):
    #loads YAML file with model and generation settings
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



# --- Load Model and Tokenizer ---
def load_model_and_tokenizer(model_dir):
    #show model path for user awareness
    print(f"Loading model from: {model_dir}")
    #load tokenizer and model from specified directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    #puts model in evaluation (inference) mode
    model.eval()
    #return both objects for use in generation
    return tokenizer, model



# --- Format Input (emotion conditioning optional)
def format_input(text, emotion=None):
    #cue = " Now give me a short positive affirmation in response to this sad, negative, etc. response:"
    #if emotion is provided, prepend it as a tag to the user input
    if emotion: 
        #format as [EMOTION] prompt
        return f"[{emotion.upper()}] {text.strip()}"
    #otherwise return raw text stripped of leading/trailing spaces
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



# --- Save Outputs to Log File ---
def save_output_log(prompt, output, emotion=None):
    #ensure output directory exists
    os.makedirs("results/affirmations_generated", exist_ok=True)
    #format current timestamp for unique file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    #clean the prompt and emotion for use in filename
    safe_prompt = prompt[:30].replace(" ", "_").replace("/", "_")
    safe_emotion = emotion.replace(" ", "_") if emotion else "none"

    #compose full filepath for saving
    filename = f"results/affirmations_generated/inference_{safe_prompt}_{safe_emotion}_{timestamp}.txt"

    #write prompt, output, and timestamp to the file
    with open(filename, "w") as f:
        f.write(f"[{datetime.now()}]\nPrompt: {prompt}\nOutput: {output}\n\n")

    #confirm log save 
    print(f"Saved output to: {filename}")



# --- Main Function ---
def main():
    #setup CLI parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="outputs/checkpoints/checkpoint-389640", help="Path to model directory")
    parser.add_argument("--input", type=str, help="Input text prompt (journal entry, etc.)")
    parser.add_argument("--emotion", type=str, help="Optional emotion tag")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--log_output", action="store_true", help="Log output to results folder")
    #parse all arguments
    args = parser.parse_args()

    #load model config and objects
    config = load_config(args.config)
    tokenizer, model = load_model_and_tokenizer(args.model_dir)

    #handle prompt input: from command-line or ask interactively
    if args.input:
        prompt = format_input(args.input, args.emotion)
    else:
        raw = input("Hey, what's on your mind today? Type anything you're feeling: :")
        mood = input("Optional mood label: ")
        prompt = format_input(raw, mood if mood else None)

    #generate affirmation
    output = generate_affirmation(prompt, tokenizer, model)

    #print result to console
    print("\n Prompt: \n", prompt)
    print("\n Generated Affirmation \n", output)

    #save the result (optional)
    if args.log_output:
        save_output_log(prompt, output, args.emotion)



# --- main ---
if __name__ == "__main__":
    #entry point of script
    main()