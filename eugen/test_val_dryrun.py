#!/usr/bin/env python3
"""
test_val_dryrun.py
Checks that the model can be loaded and executed on the CPU (dry-run).
This does not require any dataset downloads or GPU resources, so it can
be run safely and quickly on the login node.
"""

import os
import sys
from pathlib import Path
import torch

# Define workspace directories
ROOT_DIR = Path(__file__).resolve().parent.parent
g1_resnet_dir = ROOT_DIR / 'g1-resnet'

# Add g1-resnet to Python path to allow imports to work correctly
sys.path.insert(0, str(g1_resnet_dir))

from models.common import DetectMultiBackend
import models.yolo
import models.common

def test_dryrun():
    print("=== Initiating Model Dry-Run Verification ===")
    
    # We will test loading the model on the CPU
    device = torch.device("cpu")
    weights_path = ROOT_DIR / "runs/train/exp/weights/best.pt"
    
    if not weights_path.exists():
        print(f"Error: Weights file not found at {weights_path}")
        print("Please ensure the weights are downloaded and placed correctly.")
        sys.exit(1)
        
    print(f"Loading weights from: {weights_path}...")
    
    # Configure time-step parameter
    T = 5
    models.yolo.time_window = T
    models.common.time_window = T
    
    try:
        model = DetectMultiBackend(weights_path, device=device)
        print("Model loaded successfully.")
        
        # SNN models expect input size (batch, T, channel, height, width)
        imgsz = 640
        print(f"Warmup / Dry-run forward pass with dummy tensor of shape (1, {T}, 3, {imgsz}, {imgsz})...")
        
        # Perform dry run pass
        dummy_input = torch.zeros(1, T, 3, imgsz, imgsz).to(device)
        model(dummy_input)
        
        print("Forward pass completed successfully!")
        print("=== Dry-run completed: NO ERRORS FOUND ===")
        
    except Exception as e:
        print("Verification failed with the following exception:")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_dryrun()
