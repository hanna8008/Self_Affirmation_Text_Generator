# ----------------------------------------------------------------------------
# run_project.sh
# ----------------------------------------------------------------------------
# 
# A shell script that runs the dataset preprocessing pipeline. It executes the 
# script that builds the final paired dataset from raw affirmations and tweets.



#!/bin/bash

# --- Activate Envrionment ---
#source activate affirmgen

# --- Step 1: Download and Preprocess ---
echo "Running data preparation..."
python data/combine_affirmations_tweets_datasets.py