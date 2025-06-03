#!/bin/bash

# Download the model if not already present
python -c "
from huggingface_hub import hf_hub_download
import os

model_dir = 'outputs/checkpoints/checkpoint-389640'
if not os.path.exists(os.path.join(model_dir, 'pytorch_model.bin')):
    print('Downloading model from Hugging Face...')
    hf_hub_download(repo_id='hanna8008/affirmation-gpt2', filename='model.safetensors', local_dir=model_dir)
else:
    print('Model already downloaded!')
"

#source ~/miniconda/etc/profile.d/conda.sh
#conda activate affirmgen



#runs the python script that launches the Gradio web app interface
python gui.py
