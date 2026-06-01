#!/bin/bash
#SBATCH --job-name=exp3-baseline
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=slurm_logs/exp3-baseline-%j.out

# Sync Gen1 files to root workspace to execute Gen1 training
cp -r g1-resnet/models/* models/
cp -r g1-resnet/utils/* utils/
cp -r g1-resnet/prophesee_utils ./prophesee_utils

# Run training
uv run train_g1.py \
  --data data/gen1.yaml \
  --img 320 \
  --batch-size 128 \
  --epochs 50 \
  --lr 0.01 \
  --name exp3_baseline
