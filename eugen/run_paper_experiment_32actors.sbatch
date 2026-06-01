#!/bin/bash
#
# DelftBlue SLURM batch script — 32 actors / gpu-v100 / priority-optimized.
# Submit from the repo root with:   sbatch run_paper_experiment_32actors.sbatch
#
# Make sure uv is installed on your home directory.
# Make sure the .venv is created (use uv sync)
#
# If not from DSAIT check the correct account partition (I think it might
# apply to Kasper but I am not sure since access is requested for a dsait
# course: https://doc.dhpc.tudelft.nl/delftblue/Accounting-and-shares/ )
# Run `sacctmgr list user $USER withassoc format='user%-20,account%-45'`
# to see your account
#
# Sizing rationale (gpu-v100: 48 cores / 250 GB / 4x V100 32GB per node):
#   - 32 actor processes spawned via torch.multiprocessing (--num_actors 32
#     in run_paper_experiment_32actors.sh). OMP_NUM_THREADS=1 is forced at
#     the top of monobeast.py so each actor is single-threaded => 32 cores
#     for actors.
#   - 1 main process running 2 learner threads + the stats/checkpoint loop
#     => 3 cores. Round up to 4.
#   - 1 GPU for the learner model.
#   Total: 36 cores + 1 GPU + 108 GB RAM requested.
#   SIZING DONE WITH THE HELP OF CLAUDE - NEEDS VERIFYING
#
# Sizing chosen for maximum scheduler priority when the cluster
# is heavily loaded:
#   - 12 cores free on the v100 node (same backfill window as the
#     48-actor / a100 variant on its 64-core node).
#   - 142 GB RAM free on the node (mem-per-cpu=3G * 36 = 108 GB used).
#     Realistic peak usage is ~15-25 GB; 3 GB/cpu gives ~4x margin so the
#     scheduler sees us as a small RAM ask without risking OOM.
#
# Throughput penalty vs the 48-actor / a100 variant: roughly 25-35% slower
# wall-clock for 50M steps (combination of fewer actors + slower v100 +
# sub-linear IMPALA scaling). Worth it when the a100 queue is hours long.
#
#SBATCH --job-name="tmw-v100-32actors"
#SBATCH --partition=gpu-a100-v100
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --time=23:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

mkdir -p slurm_logs

# --- Modules ---------------------------------------------------------------
# 2024r1 / 2025 give us a recent toolchain; cuda is needed so torch finds the
# driver libs at runtime on the gpu-v100 nodes.
module load 2024r1
module load cuda

# --- Environment -----------------------------------------------------------
# monobeast.py also sets OMP_NUM_THREADS=1 inside Python before importing
# torch, but exporting it here too keeps any subprocess that imports torch
# (e.g. wandb's spawn workers) from oversubscribing the actor cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Make sure uv is on PATH. The standard install location is ~/.local/bin
# (uv's installer puts it there)

export PATH="$HOME/.local/bin:$PATH"

# Force-quiet wandb's console spam; logs go to W&B + slurm_logs/*.out anyway.
export WANDB_CONSOLE=off

# --- Sanity checks ---------------------------------------------------------
# Fail-fast guards. Each one prevents a known silent-failure mode that would
# otherwise waste hours of allocation before being noticed. Order matters:
# cheapest checks first, GPU smoke test last (it imports torch).
echo "Node:           $(hostname)"
echo "Job ID:         ${SLURM_JOB_ID:-N/A}"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-?}"
echo "Allocated GPUs: ${SLURM_GPUS_PER_TASK:-?}  (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
echo "Working dir:    $(pwd)"
nvidia-smi || true

# 1. uv on PATH (last run died here when uv wasn't installed yet).
command -v uv >/dev/null || { echo "FAIL: uv not found on PATH"; exit 1; }

# 2. .env present and contains WANDB_API_KEY. Without this, wandb falls back
#    to anonymous mode or prompts for login => 12h of unlogged training.
[ -f .env ] || { echo "FAIL: .env missing — scp it from your laptop"; exit 1; }
set -a; source .env; set +a
[ -n "${WANDB_API_KEY:-}" ] || { echo "FAIL: WANDB_API_KEY not set after sourcing .env"; exit 1; }

# 3. .venv exists. Compute nodes generally have no internet, so if uv tries
#    to recreate it inside the job it'll hang or fail. Build it on the login
#    node with `uv sync` before submitting.
[ -d .venv ] || { echo "FAIL: .venv missing — run 'uv sync' on the login node"; exit 1; }

# 4. Torch can actually see the GPU. nvidia-smi succeeding doesn't prove
#    torch will — if the cuda module is mismatched with what torch was built
#    against, the run silently falls back to CPU and burns the allocation.
uv run python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
    || { echo "FAIL: torch.cuda.is_available() is False — check cuda module / torch build"; exit 1; }
echo "Sanity checks passed."

# --- Run -------------------------------------------------------------------
# run_paper_experiment_32actors.sh is the canonical entrypoint for this
# variant; we just hand control to it so its CLI flags stay the single
# source of truth. --unbuffered makes log lines stream to slurm_logs/*.out
# in real time instead of in 8KB chunks (useful for `tail -f`).
srun bash run_paper_experiment_32actors.sh