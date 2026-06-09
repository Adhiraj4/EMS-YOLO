# Critical Perspective - Learning Rate 0.001 Sweep (T=5)
* **Optimization Underfitting:** Decreasing the initial learning rate to **0.001** (10x lower than the default LR = 0.01) resulted in a much slower optimization rate, leading to persistent underfitting. The model achieved a peak validation mAP@.5 of **0.567** at Epoch 25 (Precision = 0.678, Recall = 0.526).
* **Flat Suboptimal Plateau:** Unlike the baseline or the high-LR run, this configuration exhibited flat, sluggish learning dynamics. After Epoch 21, the validation mAP@.5 remained stuck in a narrow, suboptimal plateau between **0.556** and **0.567**, never rising further or showing signs of recovery.
* **Learning Rate Comparisons (BS = 16):**
  - **LR=0.1**: Highly unstable. Peaked early at **0.591 mAP@.5** (Epoch 6) before collapsing to **0.381 mAP@.5**, showing severe parameter divergence.
  - **LR=0.01 (Baseline/BS-16)**: Optimal configuration. Peaked at **0.623 mAP@.5** (Epoch 17), maintaining a high-quality plateau (**0.605–0.623**).
  - **LR=0.001**: Stable but severely underfitting. Peaked at **0.567 mAP@.5** (Epoch 25), plateauing around **0.562**.
  - *Finding:* A learning rate of 0.001 is too low to effectively update the SNN's spiking thresholds and membrane potentials within the 50-epoch budget. SNN threshold dynamics are highly sensitive to step size, and small gradients fail to drive synaptic weight updates past the firing thresholds, trapping the model in a poor local minimum.
* **Slurm Timeout Safety:** The job timed out during Epoch 46. Because the model had been in a flat validation metric plateau since Epoch 21, the 45 completed epochs are more than sufficient to scientifically diagnose underfitting and rule out LR=0.001 as a viable configuration.
