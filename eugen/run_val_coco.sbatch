#!/bin/bash
#SBATCH --job-name=val-coco
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --partition=gpu-a100
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --output=slurm_logs/val-coco-%j.out

# Load software modules
module load 2024r1
module load cuda
module load python

# Add virtual environment packages to Python path
export PYTHONPATH="/home/ebulboaca/EMS-YOLO/.venv/lib/python3.10/site-packages:$PYTHONPATH"

# Navigate to self-contained g1-resnet workspace
cd g1-resnet

# Run COCO validation
python3 val.py \
  --data data/coco.yaml \
  --weights ../runs/train/exp/weights/best.pt \
  --img 640 \
  --batch-size 32 \
  --project /scratch/$USER/runs/stage1 \
  --name coco_val
