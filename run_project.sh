#!/bin/bash

# --- Activate Envrionment ---
source activate affirmgen

# --- Step 1: Download and Preprocess ---
echo "Running data preparation..."
python data/combine_affirmations_tweets_datasets.py

# Optional: Step 2: Run EDA
# jupyter nbconvert --execute --to notebook data/paired_dataset_eda.ipynb --output data/eda_affirmations_tweets.ipynb