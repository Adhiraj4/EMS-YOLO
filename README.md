# Deep Directly-Trained Spiking Neural Networks for Object Detection: A DelftBlue Reproduction Study

This repository contains our reproduction and extension of the EMS-YOLO paper (*Deep Directly-Trained Spiking Neural Networks for Object Detection*, ICCV 2023). Rather than a dry list of files and configurations, this document presents our work through the lens of an academic blog post. It outlines the scientific motivation, the engineering breakthroughs required to run event-based SNNs at scale on the DelftBlue HPC cluster, and our empirical findings across five experimental stages.

---

## 1. The Scientific Context: Why Directly-Trained SNNs?

Most modern object detectors—such as the YOLO family—rely on Artificial Neural Networks (ANNs). While highly accurate, ANNs are computationally heavy and power-hungry, requiring millions of dense floating-point Multiply-Accumulate (MAC) operations. 

**Spiking Neural Networks (SNNs)** offer a biologically inspired, energy-efficient alternative. By accumulating membrane potentials over time and communicating via sparse, binary spikes, SNNs replace continuous activations with event-driven computations. On neuromorphic hardware, this translates to substantial power savings since silent (non-spiking) neurons consume virtually no energy.

### The Architectural Gap in Prior Work
Historically, deep SNNs have been constructed in two ways:
1. **ANN-to-SNN Conversion**: Pre-trained ANNs are converted into SNNs. This requires long temporal inference windows ($100$ to $1000+$ time steps) to approximate continuous weights, eroding the energy efficiency benefits and failing to capture spatio-temporal dynamics from event-based cameras.
2. **Direct Training with Surrogate Gradients**: SNNs are trained directly from scratch using surrogate gradients to bypass the non-differentiable step function. While successful for classification, object detection requires deeper backbones to regress continuous bounding box coordinates. Prior attempts bolted a spiking backbone to a standard ANN head, reintroducing high energy costs.

### The EMS-YOLO Solution
EMS-YOLO solves this by implementing an **Energy-efficient Membrane-Shortcut ResNet (EMS-ResNet)**. 
* ** Membrane Shortcuts**: By placing Leaky Integrate-and-Fire (LIF) neurons directly before shortcut convolutions when scaling channels, the network guarantees that all convolutional operations are executed on binary spikes.
* **Full-Spike Detector**: The entire pipeline—backbone and head—is composed of spiking blocks. Bounding box coordinates are regressed from the continuous membrane potential of the final output layer, keeping the core network sparse and event-driven.

---

## 2. Engineering Feats: Scaling SNNs on DelftBlue

Running EMS-YOLO on the DelftBlue cluster required overcoming severe parallel file system bottlenecks, multiprocessing memory replication crashes, and hardware compatibility mismatches. Our engineering adaptations are described below.

### 2.1 Resolving the GPFS Disk I/O Bottleneck (25x–50x Speedup)
Our preprocessing pipeline for the Prophesee Gen1 dataset converts continuous event streams into frame bins. The original script sliced raw `.h5` files by repeatedly querying the GPFS parallel file system on disk for every bounding box timestamp. 

On DelftBlue's GPFS filesystem, this random I/O penalty caused sequence slicing to average **49 seconds per sequence**, meaning our jobs timed out long before completing the 72,481 training samples. We resolved this by loading the entire sequence array into RAM once at the start of processing each sequence:

```python
with h5py.File(h5_path, 'r') as h5_file:
    # Read the full dataset into RAM once to bypass GPFS random disk queries
    events = h5_file['events'][:]
```
This single design change reduced sequence processing times from **49 seconds to 1-2 seconds** (a **25x–50x speedup**), allowing preprocessing to complete within the scheduler's limits.

### 2.2 Pre-Fork Copy-on-Write RAM Caching (20x Speedup)
To avoid loading frames from disk during training (which took 45 minutes per epoch due to 72k random reads), we enabled PyTorch's in-memory RAM caching (`--cache`). However, the Slurm compute nodes instantly terminated our runs with cgroup Out-Of-Memory (OOM) signals.

**The Cause**: PyTorch dataloaders fork worker processes (`num_workers = 4`). Caching was originally performed dynamically during `__getitem__` inside these child workers. Because child namespaces are isolated, their cache additions did not propagate to the parent. The 111 GB cache was duplicated 4 times across workers, ballooning memory usage to over **444 GB** and triggering the OOM-killer.

**The Fix**: We redesigned the dataloader in `datasets_g1T.py` to pre-cache all event frame tensors in the parent process during initialization (`__init__`), using parallel thread pools:

```python
from multiprocessing.pool import ThreadPool

# Pre-cache inside the parent process before worker processes fork
results = ThreadPool(8).imap(lambda x: self.load_image_and_label(x), range(self.n))
for i, (out_img, labels) in enumerate(results):
    self.ims[i] = out_img
    self.labels[i] = labels
```
When PyTorch forks the workers, they inherit this pre-populated parent cache. Since workers only read from the cache without modifying it, Linux's **Copy-on-Write (COW)** optimization allows them to share the exact same physical memory space. This eliminated cache duplication, maintaining peak RAM below **120 GB** (within our 128 GB Slurm allocation) and reducing training epoch times from **45 minutes to 2 minutes** (a **20x speedup**).

### 2.3 Environment & Codebase Stabilizations
* **GPU Compute Capability Mismatch**: Default scheduler queues routed workloads to Volta Tesla V100 cards. PyTorch's build (`+cu130`) only compiles kernels for Compute Capability $\ge 7.5$. The jobs crashed with `no kernel image is available`. We updated all configurations to target Ampere A100 GPUs (CC 8.0) via `--partition=gpu-a100`.
* **NumPy 2.0 Compatibility**: Audited dataloaders and replaced the deprecated `astype(np.int)` with `astype(int)` across 5 files to prevent runtime crashes.
* **Pillow 10+ Text Measurement Fallback**: Logging threads crashed when plotting bounding boxes because Pillow 10.0+ removed the `.getsize()` API. We implemented a try-except fallback that queries dimensions via `.getbbox()`:
  ```python
  try:
      w, h = self.font.getsize(text)
  except AttributeError:
      left, top, right, bottom = self.font.getbbox(text)
      w, h = right - left, bottom - top
  ```
* **Warmup Shape Adaptation**: Patched the model warmup step to pass a 5D dummy tensor `(1, T, 3, imgsz, imgsz)` rather than a 4D tensor, matching the temporal dimensions expected by the SNN backbone.

---

## 3. Empirical Results: Reproduction and Extension

We evaluated the model across five distinct experimental stages on DelftBlue.

### 3.1 Baseline Reproduction (Gen1 Dataset)
We trained the EMS-ResNet10 detector on the event-based Gen1 Automotive dataset under the baseline configuration ($T = 5$ time steps, batch size of 16, learning rate of 0.01). 

| Metric | Original Paper | Our Reproduction | Delta |
| :--- | :---: | :---: | :---: |
| **mAP @ 0.5** | 0.547 | **0.623** | +0.076 |
| **mAP @ 0.5:0.95** | 0.267 | **0.343** | +0.076 |
| **Peak Epoch** | 50 | 17 (Trained to 33) | -33 epochs |

Our reproduced model not only successfully converged but outperformed the paper's original baseline by **+7.6% mAP** on both metrics. The model quickly stabilized, with metrics plateauing after epoch 11.

---

### 3.2 COCO2017 Validation
To verify compatibility with static image formats (where static frames are duplicated $T$ times), we ran validation on the COCO2017 validation split (5000 images) using the authors' pre-trained EMS-ResNet34 weights.

| Source | Model | T | mAP @ 0.5 | mAP @ 0.5:0.95 |
| :--- | :--- | :---: | :---: | :---: |
| **Original Paper** | EMS-Res34 | 4 | 0.501 | - |
| **Our Reproduction** | EMS-Res34 | 4 | **0.508** | **0.308** |

The reproduced mAP @ 0.5 of **0.508** matches the paper's reported score, verifying the correctness of our static-image temporal adapter and validation loops.

---

### 3.3 Hyperparameter Sensitivity Sweeps
We swept batch sizes and learning rates to assess the optimization stability of the spiking layers on Gen1 ($T=5$).

#### Batch Size Sweep (LR = 0.01)
* Larger batch sizes scale VRAM usage linearly, but yield a slight, steady decrease in validation mAP.
* Smaller batch sizes (specifically $BS=8$) optimize better for SNNs, likely because smaller batches introduce stochastic gradient noise that helps threshold-based LIF neurons escape local minima.

| Batch Size | Best Epoch | Precision | Recall | mAP @ 0.5 | mAP @ 0.5:0.95 | GPU Memory (GB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **8** | 17 | 0.732 | 0.581 | **0.626** | **0.345** | 6.92 |
| **16** | 10 | 0.715 | 0.578 | 0.623 | 0.339 | 13.2 |
| **32** | 20 | 0.734 | 0.582 | 0.622 | 0.342 | 25.8 |
| **64** | 13 | 0.716 | 0.576 | 0.620 | 0.340 | 50.7 |

#### Learning Rate Sweep (BS = 16)
* **Too Small ($LR=0.001$)**: Slow convergence, severe underfitting within the training budget (peak mAP @ 0.5 of 0.567).
* **Too Large ($LR=0.1$)**: Rapid early convergence (peaking at epoch 6), but highly unstable validation behavior in later epochs, causing metrics to degrade.
* The paper's default of **$LR=0.01$** provides the most stable threshold updates.

---

### 3.4 Temporal Step Ablation Study ($T \in \{1, 2, 4, 6\}$)
SNNs rely on temporal accumulation to integrate information over time. We swept the number of time bins $T$ on the Gen1 dataset to characterize the trade-off between temporal resolution and computational overhead.

| Timestep ($T$) | Best Epoch | Precision | Recall | mAP @ 0.5 | mAP @ 0.5:0.95 | Training Time / Epoch |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 25 | 0.704 | 0.523 | 0.569 | 0.305 | ~2 minutes |
| **2** | 35 | 0.718 | 0.547 | 0.588 | 0.329 | ~2.5 minutes |
| **4** | 40 | 0.711 | 0.571 | 0.607 | 0.343 | ~4 minutes |
| **5 (Baseline)** | 17 | 0.720 | 0.583 | 0.623 | 0.342 | ~6 minutes |
| **6** | 15 | 0.720 | 0.584 | **0.625** | **0.345** | ~7.5 minutes |

* **Accuracy Scales Monotonically**: Increasing $T$ provides more temporal context, improving mAP.
* ** Diminishing Returns**: SNN accuracy begins to saturate after $T=4$. The gain from $T=4$ to $T=6$ is only **+1.8% mAP @ 0.5**, while computational overhead nearly doubles. $T=4$ represents the practical sweet spot.

---

### 3.5 N-Cars Control Run (Classification-to-Detection)
To evaluate the SNN backbone's generalizability as a sequence classifier, we trained on the Prophesee N-Cars dataset. Since N-Cars is a binary classification dataset, we mapped it to a YOLO detection format by assigning virtual, full-frame bounding boxes (`0 0.5 0.5 1.0 1.0`) to positive sequences and generating empty label files for backgrounds.

* The model trained rapidly, completing all 50 epochs in **7.97 hours**.
* Reached an outstanding validation **mAP @ 0.5 of 0.974** and **mAP @ 0.5:0.95 of 0.975**.
* This control run confirms that the underlying EMS-ResNet34 feature extractor generalizes to different neuromorphic sensors and easily learns event sequence discrimination.

---

## 4. Technical Appendix: Getting Started

### 4.1 Conda Environment Setup
The required environment and packages are detailed in `environment.yml`. To install locally:
```bash
conda env create -f environment.yml
conda activate ems-yolo
pip install -r requirements.txt
```

### 4.2 Workspace Synchronization to DelftBlue
Use the anchored synchronization script `eugen/sync_to_delftblue.sh` to push local code edits to scratch storage on DelftBlue:
```bash
bash eugen/sync_to_delftblue.sh <your_netid>
```

### 4.3 Pretrained Weights & Dataset Links
* **COCO Pretrained Weights (EMS-Res34)**: Download the best and last weights directly from the [Official Google Drive Folder](https://drive.google.com/drive/folders/1mry8sdED6ncqxajmQROKBECkcmXStpB?usp=sharing).
* **Gen1 / N-Cars Datasets**: Available on DelftBlue scratch via kDrive direct downloads (pre-configured in `eugen/download_datasets.sh`).

### 4.4 Running Experiments (Slurm SBATCH)
All event-based runs must execute inside the `g1-resnet/` directory for namespace isolation. Below is the SBATCH template for Gen1 baseline training:

```bash
#!/bin/bash
#SBATCH --job-name=gen1-baseline
#SBATCH --account=Education-EEMCS-MSc-DSAIT
#SBATCH --partition=gpu-a100
#SBATCH --gpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=8000M
#SBATCH --output=slurm_logs/exp1-baseline-%j.out

module load 2024r1 openmpi cuda/12.4 python

export PYTHONPATH="/home/$USER/EMS-YOLO/.venv/lib/python3.10/site-packages:$PYTHONPATH"
export WANDB_MODE=disabled

cd g1-resnet
python3 train_g1.py \
  --data data/gen1.yaml \
  --img 320 \
  --batch-size 16 \
  --epochs 50 \
  --weights "" \
  --device 0 \
  --project /scratch/$USER/runs/stage1 \
  --name gen1_baseline \
  -T 5 \
  --cache
```

To calculate spike firing rates post-training:
```bash
python3 calculate_fr.py --weights /scratch/$USER/runs/stage1/gen1_baseline/weights/best.pt -T 5
```

*Our codebase is adapted from the Ultralytics YOLOv3 repository. Please remember to cite their work as well.*
