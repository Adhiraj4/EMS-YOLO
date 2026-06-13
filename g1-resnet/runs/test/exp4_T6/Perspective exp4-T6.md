# Critical Perspective - Timestep T=6 Ablation Sweep

### 1. Performance and Convergence Analysis
* **Incomplete Convergence due to Cluster Time Limits:** The T=6 ablation run was terminated at **Epoch 36** after hitting the 24-hour walltime limit on the DelftBlue cluster. This highlights the severe computational overhead of scaling timesteps in SNNs, as each epoch took approximately **40 minutes** on an A100 GPU.
* **Peak Metrics Achieved:**
  * **Precision:** `0.703` (Epoch 17)
  * **Recall:** `0.598` (Epoch 17)
  * **mAP@.5:** **`0.625`** (Epoch 17)
  * **mAP@.5:.95:** **`0.348`** (Epoch 21)

---

### 2. Timestep Comparative Analysis ($T \in \{1, 2, 4, 5, 6\}$)

| Timestep | Peak Epoch | Precision | Recall | mAP@.5 | mAP@.5:.95 | Training Speed (per epoch) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **T=1** | 25 | 0.704 | 0.523 | **0.569** | **0.305** | ~7 mins | Completed (Ep 33) |
| **T=2** | 38 | 0.711 | 0.546 | **0.588** | **0.327** | ~13 mins | Completed (Ep 50) |
| **T=4** | 18 | 0.720 | 0.565 | **0.620** | **0.346** | ~27 mins | Timed out (Ep 46) |
| **T=5 (Baseline)** | 17 | 0.737 | 0.576 | **0.623** | **0.320** | ~30 mins | Timed out (Ep 46) |
| **T=6** | 17 | 0.703 | 0.598 | **0.625** | **0.348** | ~40 mins | Timed out (Ep 36) |

#### Key Insights from the Ablation Study:
1. **Monotonic Scaling of mAP@.5:**
   - As the temporal resolution increases ($T=1 \rightarrow 2 \rightarrow 4 \rightarrow 5 \rightarrow 6$), the model's ability to capture fine-grained event features improves steadily: **`0.569`** $\rightarrow$ **`0.588`** $\rightarrow$ **`0.620`** $\rightarrow$ **`0.623`** $\rightarrow$ **`0.625`**.
   - Denser event integration windows allow the spiking neurons to sustain membrane potentials and accumulate spatial context, which is crucial for identifying smaller, fast-moving objects.

2. **The mAP@.5:.95 Localization Jolt:**
   - There is a noticeable drop in fine localization accuracy for **T=5 (0.320 mAP@.5:.95)** compared to **T=4 (0.346)** and **T=6 (0.348)**.
   - This suggests that even divisions of the 100ms event frame window (e.g., slicing into 4 or 6 intervals) align better with the SNN's firing threshold and decay dynamics. Odd divisions like T=5 may introduce temporal "jitter" or ghosting artifacts that blur object boundaries, degrading performance at high IoU thresholds.

3. **Diminishing Returns vs. Computational Cost:**
   - Going from T=4 to T=6 yields only a marginal **+0.5% mAP@.5** and **+0.2% mAP@.5:.95** improvement.
   - However, training time per epoch increases from 27 minutes ($T=4$) to 40 minutes ($T=6$), representing a **+48% increase in compute cost**.
   - Thus, **T=4** represents the optimal sweet spot for balancing SNN temporal accuracy and cluster resource utilization.
