# Critical Perspective - Timestep T=2 Ablation Sweep
* **Complete Converged Run:** Unlike the baseline or larger timestep sweeps that timed out, the $T=2$ ablation sweep successfully completed all 50 epochs in **13.04 hours** due to its lower computational load.
* **Accuracy Ceiling:** The model converged rapidly, achieving a peak training validation mAP@.5 of **0.595** at Epoch 18. The final best-model validation yielded **0.588 mAP@.5** and **0.327 mAP@.5:.95**.
* **Class-Specific Disparity:** 
  - **Car**: Peak mAP@.5 of **0.777** (mAP@.5:.95 = 0.510).
  - **Pedestrian**: Peak mAP@.5 of **0.398** (mAP@.5:.95 = 0.144).
  - *Finding:* Pedestrian detection performance is significantly lower. This is caused by severe class imbalance (2,820 pedestrian labels vs. 37,913 car labels) and the smaller spatial footprint of pedestrians in event data, which makes accumulation over a short $T=2$ window insufficient to trigger high-confidence spikes.
* **Temporal Context Trade-off:**
  - **T=1**: Best mAP@.5 of **0.569** (mAP@.5:.95 = 0.305).
  - **T=2**: Best mAP@.5 of **0.588** (mAP@.5:.95 = 0.327).
  - **T=5 (Gen1 Baseline)**: Best mAP@.5 of **0.623** (mAP@.5:.95 = 0.320).
  - *Finding:* Increasing the temporal steps from 1 to 2 provides a substantial relative gain of **+3.3% mAP@.5** and **+7.2% mAP@.5:.95**. SNNs depend on temporal membrane potential integration to resolve moving objects; a window of $T=2$ allows for rudimentary optical-flow-like accumulation, though $T=5$ remains necessary to match the full paper-equivalent representation.
