# ----------------------------------------------------------------------------
# generate_batch.py
# ----------------------------------------------------------------------------
#
# This script performs batch inference using a fine-tuned GPT-2 model to generate 
# affirmations based on a CSV of inputs and optional emotions. It takes the 
# input/output file paths via command-line arguments and saves the generated 
# affirmatinos row by row into a new CSV file.



# --- Imports ---
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import time
from tqdm import tqdm
from generate_inference_affirmation import format_input, generate_affirmation, load_config



# --- Start Timer ---
#record the starting time of the batch process
start_time = time.time()



# --- Load model & tokenizer
#load configuration from the YAML file
config = load_config()
#load the tokenizer from trained model directory
tokenizer = AutoTokenizer.from_pretrained(config["output_dir"])
#load fine-tuned GPT-2 model
model = AutoModelForCausalLM.from_pretrained(config["output_dir"])
#set model to evaluation mode (disables dropout, gradients)
model.eval()



# --- Load Input CSV ---
"""df = pd.read_csv("data/test.csv")"""
#accept input CSV path from the first command-line argument
df = pd.read_csv(sys.argv[1])
#accept output CSV path from the second command-line argument
output_file = sys.argv[2]



# --- Initialize Output List ---
#will store dictionaries of Input, Emotion, and Generated Affirmation
results = []



# --- Loop with Progress Bar ---
#tqdm gives a real-time visual on progress, total rows, and completion estimate
progress_bar = tqdm(total=len(df), desc="Generating Affirmations", ncols=100)



# --- Generate Affirmations row by row ---
#iterate through each row in the DataFrame
for _, row in df.iterrows():
    #extract user journal entry
    text = row["Input"]
    #extract emotion tag only if it exists and is not null
    emotion = row["Emotion"] if "Emotion" in row and pd.notnull(row["Emotion"]) else None
    #format input as [EMOTION] text
    prompt = format_input(text, emotion)
    #generate affirmation from model
    output = generate_affirmation(prompt, tokenizer, model)
    #append result dictionary to list
    results.append({
        "Input": text,
        "Emotion": emotion,
        "Affirmation": output
    })
    #update progress bar by 1 row
    progress_bar.update(1)

#cleanly close the progress bar after all rows are processed
progress_bar.close()



# --- Save Results ---
"""output_df = pd.DataFrame(results)
output_df.to_csv(output_file, index=False)"""
pd.DataFrame(results).to_csv(output_file, index=False)



# --- Completion Message ---
#compute total processing time
total_time = time.time() - start_time
#success message
print(f"Batch Generation Complete! {len(results)} row saved to {output_file}")
#summary
print(f"Finished Generating {len(results)} affirmations.")
#time taken
print(f"Total Time: {total_time:.2f} seconds", flush=Truegener)