# ----------------------------------------------------------------------------
# train_gpt2.py
# ----------------------------------------------------------------------------


import yaml
import argparse
import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)



# --- Reproducibiity ---
set_seed(42)



# --- Load Configuration ---
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



# --- Load Dataset ---
def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    #merge input and output into a single prompt + target text
    df['text'] = df['Input'].astype(str).str.strip() + " " + df['Output'].astype(str).str.strip()
    return Dataset.from_pandas(df[['text']])



# --- Tokenization ---
def tokenize_data(dataset, tokenizer, max_length=512):
    return dataset.map(
        lambda e:tokenizer(e['text'], 
        truncation=True, 
        padding='max_length', 
        max_length=max_length), 
        batched=True)



# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)

    
    # --- Load Tokenizer & Model ---
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    #add pad token if missing
    if tokenizer.pad_token is None:
        print("Padding token not found. Setting pad_token = eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config['model_name'])

    #resize embeddings to accommodate new pad token
    model.resize_token_embeddings(len(tokenizer))


    # --- Load & Prepare Datasets ---
    train_ds = load_dataset(config['train_file'])
    val_ds = load_dataset(config['val_file'])

    train_ds = tokenize_data(train_ds, tokenizer)
    val_ds = tokenize_data(val_ds, tokenizer)


    """# --- shorten training with smaller datasets ---
    train_ds = train_ds.select(range(min(1000, len(train_ds))))
    val_ds = val_ds.select(range(min(200, len(val_ds))))"""


    # --- Training Arguments ---
    training_args = TrainingArguments(
        output_dir = config['output_dir'],
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_strategy = "epoch",
        logging_steps = 0, 
        learning_rate = float(config['learning_rate']),
        per_device_train_batch_size = config['batch_size'],
        per_device_eval_batch_size = config['batch_size'],
        num_train_epochs = config['num_train_epochs'],
        weight_decay = config['weight_decay'],
        logging_dir = config['logging_dir'],
        save_total_limit = 2,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        greater_is_better = False,
        report_to = "tensorboard",
        fp16 = torch.cuda.is_available()
    )


    # --- Data Collator (for language modeling) ---
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


    # --- Trainer ---
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_ds,
        eval_dataset = val_ds,
        tokenizer = tokenizer,
        data_collator = data_collator
    )


    # --- Train ---
    trainer.train()


    # --- Save Final Model ---
    model.save_pretrained(config['output_dir'])
    tokenizer.save_pretrained(config['output_dir'])
    print(f"Model and tokenizer saved to {config['output_dir']}")



    # --- Print Best Model ---
    print("Best model path:", trainer.state.best_model_checkpoint)



# --- main ---
if __name__ == "__main__":
    main()