#!/bin/bash
#SBATCH --job-name=umapilot
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
source /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold/.venv/bin/activate

cd /private/groups/yehlab/wsobolew/02_projects/computational/UMA-Fold

python -c "import torch; torch.set_float32_matmul_precision('high');"

echo "Running 1-Batch Sanity Check (Pilot Run) at max_neighborhood=15..."
python scripts/pilot_run.py ++data.datasets.0.cropper.max_neighborhood=15

echo "Running 1-Batch Sanity Check (Pilot Run) at max_neighborhood=30..."
python scripts/pilot_run.py ++data.datasets.0.cropper.max_neighborhood=30

echo "Running 1-Batch Sanity Check (Pilot Run) at max_neighborhood=40..."
python scripts/pilot_run.py ++data.datasets.0.cropper.max_neighborhood=40

echo "All Pilot runs completed successfully. Pipeline is confirmed for all curriculum stages!"
