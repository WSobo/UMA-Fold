#!/bin/bash
#SBATCH --job-name=uma-train
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/logs/SLURM_out/train_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/logs/SLURM_err/train_%j.err
#SBATCH --time=96:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
eval "$(micromamba shell hook --shell bash)"
micromamba activate uma-fold

cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold

# WANDB Setup: Provide your W&B API key to sync metrics to the cloud
# (You can find it at https://wandb.ai/authorize)
#export WANDB_API_KEY=""
# Do not leak your API key in public repos or shared scripts! Consider using environment variables or .netrc for secure storage.
# OR uncomment the following line to run strictly offline (no account needed):
export WANDB_MODE="offline"

python -c "import torch; torch.set_float32_matmul_precision('high');"

echo "Starting Full UMA-Fold Training Campaign (Curriculum Pipeline)..."

# =========================================================================
# STAGE 1: AGGRESSIVE CROPPING
# =========================================================================
# Start with extremely condensed structural environments (neighborhood=15)
# This forces the model to rapidly learn local chemistry geometry very cheaply.
echo ">> [STAGE 1] Running 15 Epochs on max_neighborhood=15"
python scripts/train.py \
    run_name="pairmixer-stage1-crop15" \
    ++training.epochs=15 \
    ++data.datasets.0.cropper.max_neighborhood=15

# =========================================================================
# STAGE 2: INTERMEDIATE CROPPING
# =========================================================================
# Widen the perspective (neighborhood=30) and continue learning.
# We pass ckpt_path="checkpoints/last.ckpt", meaning PyTorch Lightning will
# automatically resume step/epoch count from where Stage 1 left off.
echo ">> [STAGE 2] Running up to Epoch 40 on max_neighborhood=30"
python scripts/train.py \
    run_name="pairmixer-stage2-crop30" \
    ++training.epochs=40 \
    ++data.datasets.0.cropper.max_neighborhood=30 \
    ++training.ckpt_path="checkpoints/last.ckpt"

# =========================================================================
# STAGE 3: FULL CONTEXT (FINETUNING)
# =========================================================================
# End with the standard neighborhood context radius (40)
# to learn global folding features and far-domain contacts.
echo ">> [STAGE 3] Running up to Epoch 100 on max_neighborhood=40"
python scripts/train.py \
    run_name="pairmixer-stage3-crop40" \
    ++training.epochs=100 \
    ++data.datasets.0.cropper.max_neighborhood=40 \
    ++training.ckpt_path="checkpoints/last.ckpt"

echo "Full Curriculum Campaign Complete!"
