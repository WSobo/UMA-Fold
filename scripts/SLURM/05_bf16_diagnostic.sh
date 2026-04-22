#!/bin/bash
# ==============================================================================
# UMA-Fold bf16 Overflow Diagnostic — Phase A
# ==============================================================================
# Purpose: 1-epoch single-GPU run with activation-magnitude instrumentation ON
# to localize the bf16 backward-overflow source inside AtomAttentionEncoder.
#
# Context: Job 32478361 completed with 82% grad-guard fire rate on 117 params,
# all inside input_embedder.atom_attention_encoder. Forward loss is finite and
# input features are finite — overflow is purely in the backward pass on this
# subgraph. This diagnostic adds per-call stderr logs of |max| activation at
# suspected sites (d, d_sq_sum, d_norm, p, p_mlp) so we can identify which
# intermediate exceeds bf16 safe range.
#
# Single GPU so the stderr stream is linear and readable; compile_model=false
# so fused graphs don't reorder the prints.
# ==============================================================================
#SBATCH --job-name=uma-bf16-diag
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/logs/SLURM_out/bf16_diag_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/logs/SLURM_err/bf16_diag_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
source /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/.venv/bin/activate

cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold

export WANDB_MODE="offline"

# Stage-3 best checkpoint — overflow pattern is stable by this point.
CKPT="checkpoints/uma-fold-epoch=95-train_loss=0.6294.ckpt"

# Validate the resume target is finite before we launch.
python scripts/preflight.py --check-ckpt "${CKPT}" --ckpt-only >/dev/null 2>&1 \
    || { echo "ERROR: ${CKPT} failed weight-finiteness check"; exit 1; }

echo ">> [DIAG] single-GPU, +1 epoch, max_neighborhood=40, resume=${CKPT}"
echo ">> [DIAG] log_activation_magnitudes=true, compile_model=false"

# epochs=96 → one additional epoch of training beyond epoch 95.
# compile_model=false → keep the graph eager so [act_mag] prints appear
# in the order they're emitted (torch.compile can hoist/fuse and reorder).
srun python scripts/train.py \
    run_name="pairmixer-bf16-diag" \
    ++training.epochs=96 \
    ++training.devices=1 \
    ++training.compile_model=false \
    ++training.accumulate_grad_batches=1 \
    ++training.log_activation_magnitudes=true \
    ++data.num_workers=4 \
    ++data.datasets.0.cropper.max_neighborhood=40 \
    ++training.ckpt_path="'${CKPT}'"

echo "bf16 diagnostic complete — grep logs/SLURM_err/bf16_diag_${SLURM_JOB_ID}.err for [act_mag] lines"
