#!/bin/bash
#SBATCH --job-name=hello
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x_%j.out

# Phase 0 connectivity check — confirms the env + GPU are visible to SLURM.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mmsearch

srun python -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
