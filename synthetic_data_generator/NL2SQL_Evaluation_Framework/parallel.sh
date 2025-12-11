#!/bin/bash -l

#SBATCH -J paraphrasing_job
#SBATCH -o logs/output.txt
#SBATCH -p gpu-all
#SBATCH -N 1
#SBATCH -w crimv3mgpu026
#SBATCH --gres=gpu:2
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# Load CUDA
module load cuda12.1/toolkit/12.1.1

# Activate conda
source /export/home/malthaf/anaconda3/etc/profile.d/conda.sh
conda activate myenv_py312

# Run your script
python3 main.py