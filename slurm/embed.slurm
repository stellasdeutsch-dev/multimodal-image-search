#!/bin/bash
#SBATCH --job-name=embed
#SBATCH --gres=gpu:1            # use gpu:2 to claim both GPUs on gpu6 / gpu4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

# Phase 2 — batch CLIP embedding on the cluster.
# Confirm partition / gres names on your cluster first:  sinfo
source ~/miniforge3/etc/profile.d/conda.sh
conda activate mmsearch

# Optional: stage the working set onto fast local NVMe (NOT backed up).
# rsync -a /shares/datasets/<your-set>/ /fastlocal/$USER/images/
# then point configs/data.yaml:images_dir at /fastlocal/$USER/images

srun python -m src.embed --config configs/embed.yaml --fp16
