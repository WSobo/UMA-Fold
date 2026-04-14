#!/bin/bash
# ==============================================================================
# UMA-Fold Multi-GPU Curriculum Training
# ==============================================================================
# Runs the full 3-stage curriculum on 4× A5500 GPUs (96 GB total VRAM).
# Effective batch size = devices × batch_size_per_gpu = 4 × 1 = 4, same as
# single-GPU with accumulate_grad_batches=4 — so learning dynamics are identical
# but training is ~4× faster.
#
# To use 8 GPUs instead:
#   Change --gres=gpu:A5500:8, --cpus-per-task=32, and ++training.devices=8
# ==============================================================================
#SBATCH --job-name=uma-train-multi
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/logs/SLURM_out/train_multi_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/logs/SLURM_err/train_multi_%j.err
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # 4 data-loader workers × 4 GPUs
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
source /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/.venv/bin/activate

cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold

export WANDB_MODE="offline"

# Number of GPUs — keep in sync with --gres above
DEVICES=4

echo "Starting Multi-GPU UMA-Fold Training Campaign (${DEVICES}× A5500)..."

# =========================================================================
# STAGE 1: AGGRESSIVE CROPPING
# =========================================================================
echo ">> [STAGE 1] ${DEVICES}-GPU, 15 Epochs, max_neighborhood=15"
python scripts/train.py \
    run_name="pairmixer-multi${DEVICES}gpu-stage1-crop15" \
    ++training.epochs=15 \
    ++training.devices=${DEVICES} \
    ++training.accumulate_grad_batches=1 \
    ++data.num_workers=4 \
    ++data.datasets.0.cropper.max_neighborhood=15

# =========================================================================
# STAGE 2: INTERMEDIATE CROPPING
# =========================================================================
echo ">> [STAGE 2] ${DEVICES}-GPU, up to Epoch 40, max_neighborhood=30"
LATEST_CKPT=$(ls -t checkpoints/last*.ckpt | head -n 1)
python scripts/train.py \
    run_name="pairmixer-multi${DEVICES}gpu-stage2-crop30" \
    ++training.epochs=40 \
    ++training.devices=${DEVICES} \
    ++training.accumulate_grad_batches=1 \
    ++data.num_workers=4 \
    ++data.datasets.0.cropper.max_neighborhood=30 \
    ++training.ckpt_path="${LATEST_CKPT}"

# =========================================================================
# STAGE 3: FULL CONTEXT (FINETUNING)
# =========================================================================
echo ">> [STAGE 3] ${DEVICES}-GPU, up to Epoch 100, max_neighborhood=40"
LATEST_CKPT=$(ls -t checkpoints/last*.ckpt | head -n 1)
python scripts/train.py \
    run_name="pairmixer-multi${DEVICES}gpu-stage3-crop40" \
    ++training.epochs=100 \
    ++training.devices=${DEVICES} \
    ++training.accumulate_grad_batches=1 \
    ++data.num_workers=4 \
    ++data.datasets.0.cropper.max_neighborhood=40 \
    ++training.ckpt_path="${LATEST_CKPT}"

echo "Multi-GPU Curriculum Campaign Complete!"
