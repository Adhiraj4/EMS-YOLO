# Critical Perspective - Learning Rate 0.1 Sweep (T=5)
* **Optimization Instability:** Increasing the initial learning rate to **0.1** (10x higher than the default LR = 0.01) resulted in severe training instability and parameter divergence. The model achieved a peak validation mAP@.5 of **0.591** at Epoch 6.
* **Severe Performance Degradation:** After reaching its peak at Epoch 6, the model's metrics degraded significantly. By Epoch 29, the validation mAP@.5 fell to a low of **0.381** (representing a massive **-33%** drop from the peak). The model recovered slightly toward the end, finishing at **0.506 mAP@.5** at Epoch 42, but was highly unstable.
* **Learning Rate Comparisons (BS = 16):**
  - **LR=0.1**: Peaked at **0.591 mAP@.5** (Epoch 6), followed by severe degradation/oscillation.
  - **LR=0.01 (Baseline/BS-16)**: Peaked at **0.623 mAP@.5** (Epoch 10/17) and stabilized in a narrow plateau (**0.605–0.623**).
  - *Finding:* A learning rate of 0.1 is far too high for this SNN architecture on event frames. Spiking neural networks are highly sensitive to gradient step sizes, and large steps destabilize the threshold-based firing states in the spiking layers, leading to catastrophic forgetting/divergence.
* **Slurm Timeout Safety:** The job timed out during the training phase of Epoch 43. Since the model had already peaked at Epoch 6 and entered a degraded/oscillating state, the 42 completed epochs are more than sufficient to diagnose learning rate instability and rule out LR=0.1 as a viable configuration.
