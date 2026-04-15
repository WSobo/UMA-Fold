#!/bin/bash
#SBATCH --job-name=uma_shape_test
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/logs/SLURM_out/shape_test_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/logs/SLURM_err/shape_test_%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A5500:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wsobolew@ucsc.edu

set -e
eval "$(micromamba shell hook --shell bash)"
micromamba activate uma-fold

# Navigate to the active repository
cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold

# Ensure Tensor Cores on the A5500 are fully utilized
python -c "import torch; torch.set_float32_matmul_precision('high');"

echo "Running PairMixer cuBLAS dimension and VRAM test..."
python test_matmul_shapes.py

echo "Shape test complete! Check the .out file for VRAM footprint and tensor dimensions."