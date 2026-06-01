#!/bin/bash
#SBATCH --job-name=exp3-bs-32
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/exp3-bs-32-%j.out

# Load software modules
module load 2024r1
module load cuda
module load python

# Add virtual environment packages to Python path
export PYTHONPATH="/home/ebulboaca/EMS-YOLO/.venv/lib/python3.10/site-packages:$PYTHONPATH"

# Navigate to self-contained g1-resnet workspace
cd g1-resnet

# Train Gen1 model with batch-size = 32 (T=5, 50 epochs, 1 GPU)
python3 train_g1.py \
  --data data/gen1.yaml \
  --img 320 \
  --batch-size 32 \
  --epochs 50 \
  --weights "" \
  --device 0 \
  --project /scratch/$USER/runs/stage2 \
  --name bs_32 \
  -T 5
