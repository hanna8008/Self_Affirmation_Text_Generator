#!/bin/bash
#project account for resource tracking
#SBATCH --account=e32706
#job name displayed in SLURM queue
#SBATCH --job-name=combine_data
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



# --- Status Messagess ---
echo "Starting .sh file"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Log file: outputs/logs/combine_data_${SLURM_JOB_ID}.log"



# --- Load Conda Envrionment ---
#unload existing modules to avoid conflicts
#module purge
#module load anaconda3

#activate the desired Conda envrionment for training 
#source $(conda info --base)/etc/profile.d/conda.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda activate affirmgen

echo "Conda envrionemnt 'affirmgen' activated."



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
echo "Split Dataset..."
python scripts/split_dataset.py