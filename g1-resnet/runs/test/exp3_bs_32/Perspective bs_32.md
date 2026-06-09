# Critical Perspective - Batch Size 32 Sweep (T=5)
* **Optimization and Scaling:** Increasing the batch size to 32 (with default learning rate **0.01**) resulted in a highly stable optimization path. It achieved a peak validation mAP@.5 of **0.622** at Epoch 20.
* **Convergence Behavior:** The model reached its peak performance slightly later than smaller batch sizes (Epoch 20 vs Epoch 17 for BS=8 and Epoch 10 for BS=16). Once peaked, it remained extremely stable, hovering within a very narrow range of **0.605** to **0.622**.
* **Batch Size Suite Analysis (Initial LR = 0.01):**
  - **BS=8**: [Batch Size 8 Sweep](file:///Users/eugenix/Downloads/Books/Business/Documente/University/Master/Q4/Fundamental%20ML/EMS-YOLO/g1-resnet/runs/test/exp3_bs_8/bs_8_results.txt) peaked at **0.626 mAP@.5** (Memory: 6.92 GB).
  - **BS=16**: [Batch Size 16 Sweep](file:///Users/eugenix/Downloads/Books/Business/Documente/University/Master/Q4/Fundamental%20ML/EMS-YOLO/g1-resnet/runs/test/exp3_bs_16/bs_16_results.txt) peaked at **0.623 mAP@.5** (Memory: 13.2 GB).
  - **BS=32**: Peaked at **0.622 mAP@.5** (Memory: 25.8 GB).
  - **BS=64**: [Batch Size 64 Sweep](file:///Users/eugenix/Downloads/Books/Business/Documente/University/Master/Q4/Fundamental%20ML/EMS-YOLO/g1-resnet/runs/test/exp3_bs_64/bs_64_results.txt) peaked at **0.620 mAP@.5** (Memory: 50.7 GB).
  - *Finding:* Increasing batch size scales up memory usage exponentially but actually yields a slight, steady decrease in validation mAP@.5 (from 0.626 down to 0.620). This indicates that smaller batch sizes (specifically BS=8) optimize better for Spiking Neural Networks on sparse event frames, likely because smaller batches introduce beneficial stochastic gradient noise.
* **Slurm Timeout Safety:** The job timed out during the training phase of Epoch 47 (completing the first step). Since the model had converged to its performance ceiling by Epoch 20, the 47 completed epochs are fully representative of the converged state.
