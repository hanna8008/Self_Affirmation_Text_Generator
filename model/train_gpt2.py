# ----------------------------------------------------------------------------
# train_gpt2.py
# ----------------------------------------------------------------------------
#
# This script fine-tunes a pre-trained GPT-2 model on a custom paired dataset of
# emotional inputs and affirmations using HuggingFace Transformers. It loads 
# configuration from a YAML file, preprocesses data, applies tokenization, sets
# training arguments, and runs model training with built-in evaluation.



# --- Imports ---
#used to load model/training configuration form a YAML file
import yaml
#enables command-line argument parsing for specifying config file path
import argparse
#for interacting with the file system
import os
#PyTorch base library for tensor operations and GPU support
import torch
#for loading CSV data into a DataFrame
import pandas as pd
#HuggingFace's Dataset object for efficient preprocessing
from datasets import Dataset
#core HuggingFace classes for training
from transformers import (
    #automatically loads tokenizer based on model name
    AutoTokenizer,
    #loads a causal language model (e.g., GPT-2)
    AutoModelForCausalLM,
    #high-level training loop abstraction from HuggingFace
    Trainer,
    #configuration class for training behavior
    TrainingArguments,
    #dynamically pads inputs during training
    DataCollatorForLanguageModeling,
    #sets random seed for reproducibility 
    set_seed
)



# --- Reproducibiity ---
#ensures consistent results across training runs by fixing random seeds
set_seed(42)



# --- Load Configuration ---
def load_config(config_path):
    #opens and loads the config YAML file as a Python dictionary 
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



# --- Load Dataset ---
def load_dataset(csv_path):
    #reads the CSV into a pandas DataFrame
    df = pd.read_csv(csv_path)

    #combines the 'Input' (tweet) and 'Output' (affirmation) into a single sequence (merge input and output into a single prompt + target text)
    df['text'] = df['Input'].astype(str).str.strip() + " " + df['Output'].astype(str).str.strip()
    
    #returns a HuggingFace Dataset object with just the combined 'text' column
    return Dataset.from_pandas(df[['text']])



# --- Tokenization ---
def tokenize_data(dataset, tokenizer, max_length=512):
    #applies the tokenizer to each row of the dataset with truncation and padding
    return dataset.map(
        lambda e:tokenizer(
            #the combined text sequence to tokenize
            e['text'], 
            #truncate to max_length
            truncation=True, 
            #pad all sequences to max_length
            padding='max_length', 
            #set upper limit on token count
            max_length=max_length
        ), 
        #process multiple examples at once for speed
        batched=True
    )



# --- Main ---
def main():
    #initialize command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/config.yaml', help='Path to config file')
    #parse arguments from command line
    args = parser.parse_args()

    #load the YAML config into a dictionary 
    config = load_config(args.config)

    
    # --- Load Tokenizer & Model ---
    #load tokenizer from HuggingFace hub
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])

    #add padding token if it doesn't exist (GPT-2 doesn't include one by default)
    if tokenizer.pad_token is None:
        #log fallback
        print("Padding token not found. Setting pad_token = eos_token.")
        #use EOS token as padding token
        tokenizer.pad_token = tokenizer.eos_token

    #load pre-trained GPT-2 model for language modeling
    model = AutoModelForCausalLM.from_pretrained(config['model_name'])

    #update embedding layer to accommodate any new tokens (e.g., padding) (resize embeddings to accommodate new pad token)
    model.resize_token_embeddings(len(tokenizer))


    # --- Load & Prepare Datasets ---
    #load and format training data
    train_ds = load_dataset(config['train_file'])
    #load and format validation data
    val_ds = load_dataset(config['val_file'])

    #tokenize training dataset
    train_ds = tokenize_data(train_ds, tokenizer)
    #tokenize validation dataset
    val_ds = tokenize_data(val_ds, tokenizer)


    """# --- shorten training with smaller datasets ---
    train_ds = train_ds.select(range(min(1000, len(train_ds))))
    val_ds = val_ds.select(range(min(200, len(val_ds))))"""


    # --- Training Arguments ---
    training_args = TrainingArguments(
        #directory to save checkpoints
        output_dir = config['output_dir'],
        #evaluate on validation set once per epoch
        evaluation_strategy = "epoch",
        #save model checkpoint once per epoch
        save_strategy = "epoch",
        #log metrics once per epoch
        logging_strategy = "epoch",
        #disable intermediate logging between epochs
        logging_steps = 0, 
        #set optimizer learning rate 
        learning_rate = float(config['learning_rate']),
        #batch size for training
        per_device_train_batch_size = config['batch_size'],
        #batch size for evaluation
        per_device_eval_batch_size = config['batch_size'],
        #total number of epochs to train
        num_train_epochs = config['num_train_epochs'],
        #regularization to prevent overfitting
        weight_decay = config['weight_decay'],
        #directory to store TensorBoard logs
        logging_dir = config['logging_dir'],
        #only keep the 2 most recent checkpoints
        save_total_limit = 2,
        #reload the best model (lowest eval loss) after training
        load_best_model_at_end = True,
        #use evaluation loss to track best model
        metric_for_best_model = "eval_loss",
        #lower loss is better
        greater_is_better = False,
        #enable TensorBoard logging
        report_to = "tensorboard",
        #use half-precision training if GPU is available 
        fp16 = torch.cuda.is_available()
    )


    # --- Data Collator (for language modeling) ---
    #prepares batches by padding sequences to max length and shifting inputs/targets
    data_collator = DataCollatorForLanguageModeling(
        #uses same tokenizer as model
        tokenizer=tokenizer, 
        #causal LM (like GPT-2) doesn't use masked language modeling
        mlm=False
    )


    # --- Trainer ---
    trainer = Trainer(
        #the GPT-2 model to train
        model = model,
        #all training configurations
        args = training_args,
        #tokenized training data
        train_dataset = train_ds,
        #tokenized validation data
        eval_dataset = val_ds,
        #tokenizer used to decode/encode data
        tokenizer = tokenizer,
        #collator that batches and formats inputs
        data_collator = data_collator
    )


    # --- Train ---
    #start the fine-tuning process
    trainer.train()


    # --- Save Final Model ---
    #save final trained model
    model.save_pretrained(config['output_dir'])
    #save tokenizer with any updates (e.g., pad token)
    tokenizer.save_pretrained(config['output_dir'])
    #confirm save path
    print(f"Model and tokenizer saved to {config['output_dir']}")



    # --- Print Best Model ---
    #output the path to best checkpoint (lowest eval loss)
    print("Best model path:", trainer.state.best_model_checkpoint)



# --- main ---
if __name__ == "__main__":
    #run the main training pipeline when script is executed 
    main()