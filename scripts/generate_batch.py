# ----------------------------------------------------------------------------
# generate_batch.py
# ----------------------------------------------------------------------------



# --- Imports ---
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
from generate_inference_affirmation import format_input, generate_affirmation, load_config



# --- Load model & tokenizer
config = load_config()
tokenizer = AutoTokenizer.from_pretrained(config["output_dir"])
model = AutoModelForCausalLM.from_pretrained(config["output_dir"])
model.eval()



# --- Load input CSV ---
"""df = pd.read_csv("data/test.csv")"""
results = []
df = pd.read_csv(sys.argv[1])
output_file = sys.argv[2]



# --- Generate Affirmations row by row ---
for _, row in df.iterrows():
    text = row["Input"]
    emotion = row["Emotion"] if "Emotion" in row and pd.notnull(row["Emotion"]) else None
    prompt = format_input(text, emotion)
    output = generate_affirmation(prompt, tokenizer, model)
    results.append({
        "Input": text,
        "Emotion": emotion,
        "Affirmation": output
    })



# --- Save Results ---
output_df = pd.DataFrame(results)
output_df.to_csv(output_file, index=False)
print("Batch generation complete for test.csv")