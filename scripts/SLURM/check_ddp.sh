#!/bin/bash
# ==============================================================================
# UMA-Fold DDP Pre-Flight Check
# ==============================================================================
# Runs on the medium (CPU) partition — no GPU queue wait.
# Catches: import failures, model instantiation errors, config problems,
#          bad checkpoint paths, data directory issues, DDP config errors.
#
# For GPU-specific errors (OOM, bf16 overflow), use the pilot targets instead.
#
# Usage:
#   make check-multi          → submit this job
#   Check the .out log        → if it exits 0, safe to train
# ==============================================================================
#SBATCH --job-name=uma-preflight
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/logs/SLURM_out/preflight_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/logs/SLURM_err/preflight_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=medium
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
source /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/.venv/bin/activate

cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold

echo "=== UMA-Fold Pre-Flight Check (CPU / medium partition) ==="

python scripts/preflight.py \
    ++training.devices=4

echo "=== Pre-Flight PASSED — safe to run make train-multi ==="
