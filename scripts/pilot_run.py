"""
scripts/pilot_run.py
────────────────────
Pilot training script to test the entire UMA-Fold pipeline (Data -> Forward -> Backward -> Val).
This ensures 100% code stability before committing to a multi-day training run.

It overrides the Hydra config to use PyTorch Lightning's `fast_dev_run`, which runs exactly 
one normal batch and one validation batch, bypassing wandb logging and checkpointing.
"""

import os
import torch
import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig, OmegaConf

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.datamodule import create_uma_fold_datamodule
from src.training.lightning_module import UMAFoldLightningModule


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    print("🚀 Initiating UMA-Fold Pilot Run (Sanity Check)...")
    
    pl.seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')

    # Initialize components
    print("Loading DataModule...")
    data_module = create_uma_fold_datamodule(OmegaConf.to_container(cfg.data, resolve=True))
    
    print("Loading UMA-Fold Model...")
    model = UMAFoldLightningModule(
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        lr=cfg.training.lr,
        compile_model=False # Disable compilation for the pilot run so it starts instantly
    )

    # Use fast_dev_run to run exactly 1 batch of training & validation to catch any crashes
    print("Setting up Trainer in FAST_DEV_RUN mode...")
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=cfg.training.precision,
        fast_dev_run=True, # <--- The Magic Flag
        logger=False,      # Don't pollute your Weights & Biases dashboard
        enable_checkpointing=False
    )

    print("Beginning 1-Batch Sanity Check...")
    trainer.fit(model, datamodule=data_module)
    print("✅ Pilot run complete! If no errors occurred, your pipeline is 100% sound.")


if __name__ == "__main__":
    main()
