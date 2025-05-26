#!/bin/bash
#project account for resource tracking
#SBATCH --account=e32706
#job name displayed in SLURM queue
#SBATCH --job-name=run_test_data_for_results_and_evaluation
#log file output (%j = job ID)
#SBATCH --output=outputs/logs/test_eval_%j.log
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



# --- Activate Envrionment ---
source ~/.bashrc
conda activate affirmgen



# --- Generate Affirmation from Test Set ---
python scripts/generate_batch.py data/test.csv results/batch_affirmations_test.csv



# --- Run Evaluation on the Generated Affirmations ---
python scripts/evaluation.py results/batch_affirmations_test.csv results/batch_affirmations_evaluated_test.csv