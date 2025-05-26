# ----------------------------------------------------------------------------
# evaluation.py
# ----------------------------------------------------------------------------
#
# Sets up a Conda envrionment named 'affirmgen' with Python 3.9, installs all
# project dependencies via 'requirements.txt', and ensures compatibility with
# user-installed Miniconda setups.



#!/bin/bash



# ---- Setup Environment Script ----
ENV_NAME="affirmgen"
PYTHON_VERSION="3.9"

echo "Creating Conda environment '$ENV_NAME'..."



#use user-installed Miniconda and initialize Conda manually from local install
CONDA_BASE="$HOME/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"



#create environment only if it doesn't already exist
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
else
    echo "Creating Conda environment..."
    conda create -y -n $ENV_NAME python=$PYTHON_VERSION
fi



#activate the environment
conda activate $ENV_NAME
echo "Environment '$ENV_NAME' activated."



#install dependencies
pip install --upgrade pip
pip install -r requirements.txt



#final confirmation message
echo "Environment '$ENV_NAME' is fully set up!"