#!/bin/bash
#SBATCH --partition=gpu                    # Partition [compute|memory|gpu]
#SBATCH --nodes=1                          # Number of nodes
#SBATCH --ntasks-per-node=1                # Tasks per node
#SBATCH --cpus-per-task=64                 # CPUs per task
#SBATCH --gpus-per-task=4                  # GPUs per task
#SBATCH --time=5-00:00:00                  # Time limit (day-hour:minute:second)
#SBATCH --account=lt200063                 # Project name
#SBATCH --job-name=hp_search               # Job name
#SBATCH --output=R-%x.out                  # Output file
#SBATCH --error=R-%x.out                   # Error file

ml Miniconda3
conda activate /project/lt200063-idcd/envs

export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TORCHELASTIC_ERROR_FILE="pytorch_error.json"

python hp_search.py
