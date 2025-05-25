#!/bin/bash

# ---- Setup Environment Script ----
ENV_NAME="affirmgen"
PYTHON_VERSION="3.9"



echo "Creating Conda environment '$ENV_NAME'..."



# Use user-installed Miniconda (already in your home directory)
# Initialize Conda manually from your local install
CONDA_BASE="$HOME/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"



# Create environment only if it doesn't already exist
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating Conda environment..."
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
fi



# Activate the environment
conda activate $ENV_NAME
echo "Environment '$ENV_NAME' activated."



# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK resources
python -m nltk.downloader punkt
echo "NLTK Punkt downloaded"

echo "Environment '$ENV_NAME' is fully set up!"