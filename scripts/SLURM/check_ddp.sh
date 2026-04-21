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

# Iterate monitored checkpoints in ascending train_loss order and validate each
# via preflight --ckpt-only. First clean candidate becomes the resume target
# for the upcoming GPU run. Any poisoned checkpoints are reported but skipped.
echo "--- Scanning monitored checkpoints for a clean resume candidate ---"
BEST_CKPT=""
for ckpt in $(ls checkpoints/uma-fold-epoch=*.ckpt 2>/dev/null \
        | sed -E 's/.*train_loss=([0-9.]+)\.ckpt$/\1 &/' \
        | sort -g \
        | cut -d' ' -f2-); do
    if python scripts/preflight.py --check-ckpt "$ckpt" --ckpt-only 2>&1 \
            | grep -q '^\[ OK \]'; then
        BEST_CKPT="$ckpt"
        echo "✓ Clean resume candidate: $BEST_CKPT"
        break
    else
        echo "✗ NaN-poisoned (skipped): $ckpt"
    fi
done

if [ -z "$BEST_CKPT" ]; then
    echo "No prior checkpoints found — would start training from scratch."
fi

CHECK_ARGS=""
if [ -n "${BEST_CKPT}" ]; then
    CHECK_ARGS="--check-ckpt ${BEST_CKPT}"
fi

python scripts/preflight.py \
    ${CHECK_ARGS} \
    ++training.devices=4

echo "=== Pre-Flight PASSED — safe to run make train-multi ==="
