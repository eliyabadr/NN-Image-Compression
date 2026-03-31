#!/bin/bash

#SBATCH --job-name=fyp_distill
#SBATCH --account=egb11           # Replace with your actual project account if different

## specify the required resources
#SBATCH --partition=gpu           # Use the GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4         # CPUs for data loading
#SBATCH --gres=gpu:1              # Standard GPU request (or use v100d32q:1 if required)
#SBATCH --mem=32000               # 32GB RAM
#SBATCH --time=0-6:00:00         # 12 hours (Format: D-HH:MM:SS)

## email notifications
#SBATCH --mail-type=ALL
#SBATCH --mail-user=egb11@mail.aub.edu

## Load modules and run
module load python/ai-4

# Optional: Install compressai if it's missing in the module
pip install --user compressai
python avif_complete.py
