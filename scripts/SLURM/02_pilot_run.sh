#!/bin/bash
#SBATCH --job-name=uma-pilot
#SBATCH --output=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/logs/SLURM_out/pilot_%j.out
#SBATCH --error=/private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/logs/SLURM_err/pilot_%j.err
#SBATCH --time=01:00:00
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
# OR if you prefer standard conda:
# source ~/.bashrc 
# conda activate uma-fold

cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold

python -c "import torch; torch.set_float32_matmul_precision('medium');"

echo "Running 1-Batch Sanity Check (Pilot Run)..."
python scripts/pilot_run.py
