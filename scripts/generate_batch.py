# ----------------------------------------------------------------------------
# generate_batch.py
# ----------------------------------------------------------------------------



# --- Imports ---
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys
import time
from tqdm import tqdm
from generate_inference_affirmation import format_input, generate_affirmation, load_config



# --- Start Timer ---
start_time = time.time()



# --- Load model & tokenizer
config = load_config()
tokenizer = AutoTokenizer.from_pretrained(config["output_dir"])
model = AutoModelForCausalLM.from_pretrained(config["output_dir"])
model.eval()



# --- Load Input CSV ---
"""df = pd.read_csv("data/test.csv")"""
df = pd.read_csv(sys.argv[1])
output_file = sys.argv[2]



# --- Initialize Output List ---
results = []



# --- Loop with Progress Bar ---
progress_bar = tqdm(total=len(df), desc="Generating Affirmations", ncols=100)



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
    progress_bar.update(1)

    """# --- Print Progress every 100 Rows ---
    if (i + 1) % 100 == 0:
        elapsed = time.time() - start_time
        print(f"[{i + 1} / {len(df)}] rows processed | Elapsed time: {elapsed:.2f}s", flush=True)"""

progress_bar.close()



# --- Save Results ---
"""output_df = pd.DataFrame(results)
output_df.to_csv(output_file, index=False)"""
pd.DataFrame(results).to_csv(output_file, index=False)



# --- Completion Message ---
total_time = time.time() - start_time
print(f"Batch Generation Complete! {len(results)} row saved to {output_file}")
print(f"Finished Generating {len(results)} affirmations.")
print(f"Total Time: {total_time:.2f} seconds", flush=True)