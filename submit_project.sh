# ----------------------------------------------------------------------------
# submit_project.sh
# ----------------------------------------------------------------------------
#
# Submits a SLURM job to fine-tune the GPT-2 model on Quest. It specifies resource
# requirements, activates the correct Conda envrionment, and executes the training 
# script with logging and version checks. It originally also executed two jobs, 
# but those jobs only needed to be ran once so they have been commented out: 
# download & preprocess the data and split the datasets.  



#!/bin/bash
#project account for resource tracking
#SBATCH --account=e32706
#job name displayed in SLURM queue
#SBATCH --job-name=train_gpt2
#log file output (%j = job ID)
#SBATCH --output=outputs/logs/train_%j.log
#max wall time for the job
#SBATCH --time=48:00:00
#GPU partition to use
#SBATCH --partition=gengpu
#reuqest 1 A100 GPU
#SBATCH --gres=gpu:a100:1
#total memory required
#SBATCH --mem=40G
#number of CPU threads
#SBATCH --cpus-per-task=2

#SBATCH --mail-user=hannazelis2025@u.northwestern.edu
#SBATCH --mail-type=END,FAIL



# --- Status Logging ---
echo "Starting .sh file"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Log file: outputs/logs/combine_data_${SLURM_JOB_ID}.log"



# --- Load Conda Envrionment ---
#load Conda into shell
source ~/miniconda3/etc/profile.d/conda.sh
#activate 'affirmgen' envrionment
conda activate affirmgen

echo "Conda envrionemnt 'affirmgen' activated."

echo "Python path: $(which python)"
python -c "import transformers; print('Transformers version:', transformers.__version__)"



# --- Set working directory to job submission location ---
cd $SLURM_SUBMIT_DIR



# --- Set PYTHONPATH so that local modules can be imported ---
export PYTHONPATH=$(pwd)



echo "Environment activated..."



# --- Launch the Training Script ---
### Ran this on Saturday, May 24, 2025, don't have to do this again ###
#echo "Starting Data Downloading & Preprocessing..."
#bash run_project.sh


# --- Split the Dataset into Train/Val/Test ---
### Ran this on Saturday, May 24, 2025, don't have to do this again ###
#echo "Split Dataset..."
#python scripts/split_dataset.py



# --- Train the GPT 2 Model ---
echo "Training GPT-2 Model..."
#triggers model training using the configuration file and preprocessed dataset
python model/train_gpt2.py --config configs/config.yaml