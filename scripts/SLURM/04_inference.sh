#!/bin/bash
#SBATCH --job-name=uma-infer
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/logs/SLURM_out/infer_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/logs/SLURM_err/infer_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
eval "$(micromamba shell hook --shell bash)"
micromamba activate uma-fold

cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold

python -c "import torch; torch.set_float32_matmul_precision('high');"

echo "Running Fast UMA-Fold Inference..."
# Update the paths below according to your exact needs!
python scripts/inference.py \
    --yaml boltz/examples/multimer.yaml \
    --config configs/config.yaml \
    --ckpt checkpoints/last.ckpt
