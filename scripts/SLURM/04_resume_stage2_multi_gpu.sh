#!/bin/bash
# ==============================================================================
# UMA-Fold Multi-GPU — Resume from Stage 2
# ==============================================================================
# Purpose: resume the curriculum at Stage 2 (crop30) using the best finite
# checkpoint from a prior Stage 1 run. Skips Stage 1 entirely. Runs on 4 GPUs.
#
# Why this exists: the overnight run of 03_train_model.sh (single-GPU) cold-
# started Stage 2 because its `ls -t checkpoints/last*.ckpt` resume glob went
# stale after we removed `save_last=True` in P0.2. Stage 1's weights
# (uma-fold-epoch=14-train_loss=0.7179.ckpt, verified finite) were discarded at
# the stage boundary. This script picks them back up via best_ckpt() and
# continues to Stage 2 and Stage 3 on 4 GPUs with proper DDP launch semantics.
#
# DDP launch: --ntasks-per-node=4 + `srun python scripts/train.py ...` spawns
# one process per GPU so Lightning's SLURMEnvironment sees WORLD_SIZE=4.
# ==============================================================================
#SBATCH --job-name=uma-resume-stage2-multi
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/logs/SLURM_out/train_multi_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/logs/SLURM_err/train_multi_%j.err
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4         # one SLURM task per GPU (required for DDP)
#SBATCH --cpus-per-task=4           # 4 data-loader workers per task
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
source /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/.venv/bin/activate

cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold

export WANDB_MODE="offline"

DEVICES=4

# Iterate monitored checkpoints in ascending train_loss order, validate each
# via preflight --ckpt-only, and return the first whose weights are all finite.
# Same helper used by 03_train_multi_gpu.sh — factored inline so this script is
# self-contained.
best_ckpt() {
    local candidates ckpt
    candidates=$(ls checkpoints/uma-fold-epoch=*.ckpt 2>/dev/null \
        | sed -E 's/.*train_loss=([0-9.]+)\.ckpt$/\1 &/' \
        | sort -g \
        | cut -d' ' -f2-)

    while IFS= read -r ckpt; do
        [ -z "$ckpt" ] && continue
        if python scripts/preflight.py --check-ckpt "$ckpt" --ckpt-only >/dev/null 2>&1; then
            echo "$ckpt"
            return 0
        else
            echo "[best_ckpt] skipping NaN-poisoned: $ckpt" >&2
        fi
    done <<< "$candidates"

    echo "[best_ckpt] ERROR: no finite checkpoint available" >&2
    return 1
}

echo "Resuming Multi-GPU UMA-Fold Training from Stage 2 (${DEVICES}× A5500)..."

# =========================================================================
# STAGE 2: INTERMEDIATE CROPPING
# =========================================================================
LATEST_CKPT=$(best_ckpt)
echo ">> [STAGE 2] ${DEVICES}-GPU, up to Epoch 40, max_neighborhood=30, resume=${LATEST_CKPT}"
srun python scripts/train.py \
    run_name="pairmixer-multi${DEVICES}gpu-resume-stage2-crop30" \
    ++training.epochs=40 \
    ++training.devices=${DEVICES} \
    ++training.accumulate_grad_batches=1 \
    ++training.log_feature_nans=true \
    ++data.num_workers=4 \
    ++data.datasets.0.cropper.max_neighborhood=30 \
    ++training.ckpt_path="'${LATEST_CKPT}'"

# =========================================================================
# STAGE 3: FULL CONTEXT (FINETUNING)
# =========================================================================
LATEST_CKPT=$(best_ckpt)
echo ">> [STAGE 3] ${DEVICES}-GPU, up to Epoch 100, max_neighborhood=40, resume=${LATEST_CKPT}"
srun python scripts/train.py \
    run_name="pairmixer-multi${DEVICES}gpu-resume-stage3-crop40" \
    ++training.epochs=100 \
    ++training.devices=${DEVICES} \
    ++training.accumulate_grad_batches=1 \
    ++training.log_feature_nans=true \
    ++data.num_workers=4 \
    ++data.datasets.0.cropper.max_neighborhood=40 \
    ++training.ckpt_path="'${LATEST_CKPT}'"

echo "Multi-GPU Resume Campaign Complete!"
