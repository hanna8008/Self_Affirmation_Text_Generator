#!/bin/bash

# ---- Setup Environment Script ----
ENV_NAME="affirmgen"



echo "Creating Conda environment '$ENV_NAME'..."
conda create -n $ENV_NAME python=3.10 -y



echo "Activating environment and installing dependencies..."
source activate $ENV_NAME
pip install --upgrade pip
pip install -r requirements.txt



echo "Environment '$ENV_NAME' is ready!"