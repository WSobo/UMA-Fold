"""
scripts/train.py
────────────────
Top-level PyTorch Lightning training script for UMA-Fold.
Uses Hydra for configuration management and Weights & Biases for experiment tracking.
Optimized for single-node (specifically single A5500/RTX 4090) execution.
"""

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
import hydra
from omegaconf import DictConfig, OmegaConf

# Adjust paths organically so we don't trip over module imports from root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.datamodule import create_uma_fold_datamodule
from src.training.lightning_module import UMAFoldLightningModule


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # 1. Setup environments and reproducability
    pl.seed_everything(cfg.get("seed", 42), workers=True)
    
    # Enable TF32 for Ampere/Ada architectures as standard
    torch.set_float32_matmul_precision('high')

    # 2. Instantiate DataModule (Wraps Boltz-1 pipeline)
    print("Initializing DataModule...")
    # Use hydra instantiate so Boltz dataset objects are fully created from YAML definitions
    instantiated_data_cfg = hydra.utils.instantiate(cfg.data)
    data_module = create_uma_fold_datamodule(instantiated_data_cfg)
    
    # 3. Instantiate Lightning Model (Our UMAFold Wrapper)
    print("Initializing UMA-Fold Model...")
    model = UMAFoldLightningModule(
        model_config=OmegaConf.to_container(cfg.model, resolve=True),
        lr=cfg.training.lr,
        compile_model=cfg.training.compile_model,
        log_feature_nans=cfg.training.get("log_feature_nans", False),
        log_activation_magnitudes=cfg.training.get("log_activation_magnitudes", False),
    )

    # 4. Setup Logger (W&B)
    logger = WandbLogger(
        project=cfg.get("project_name", "UMA-Fold"),
        name=cfg.get("run_name", "pairmixer-run"),
        log_model=False, # We handle model saving via ModelCheckpoint to save disk space
        save_dir="logs/"
    )
    
    # 5. Setup Callbacks
    # save_last=False: last.ckpt was overwritten with NaN-poisoned weights across
    # an entire curriculum stage in the prior run, which silently corrupted every
    # subsequent stage. Relying on the top-k monitor (mode="min" over train_loss)
    # is NaN-safe because +inf sentinels logged on non-finite epochs are rejected.
    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints/",
            filename="uma-fold-{epoch:02d}-{train_loss:.4f}",
            monitor="train_loss",
            mode="min",
            save_top_k=3,
            save_last=False,
        ),
        LearningRateMonitor(logging_interval="step")
    ]

    # 6. Initialize Trainer
    devices = cfg.training.devices
    # DDP for multi-GPU; "auto" for single GPU (avoids unnecessary process spawning).
    # static_graph=True: required because fairscale's reentrant gradient
    # checkpointing replays backward through the same params twice, which
    # otherwise trips DDP's "marked ready twice" assertion. static_graph also
    # lets DDP handle unused params automatically, replacing find_unused_parameters.
    if devices > 1:
        strategy = DDPStrategy(static_graph=True)
    else:
        strategy = "auto"

    fast_dev_run = cfg.training.get("fast_dev_run", False)
    if fast_dev_run:
        print("[CHECK MODE] fast_dev_run=True — running 1 batch only to validate config/model.")

    print(f"Setting up Trainer: precision={cfg.training.precision}, devices={devices}, strategy={strategy}...")
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator="gpu",
        devices=devices,
        strategy=strategy,
        precision=cfg.training.precision,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.training.get("gradient_clip_val", 1.0),
        log_every_n_steps=cfg.training.log_every_n_steps,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        fast_dev_run=fast_dev_run,
    )

    # 7. Train!
    print("Beginning Training...")
    ckpt_path = cfg.training.get("ckpt_path", None)
    if ckpt_path is not None and not os.path.exists(ckpt_path):
        print(f"Warning: Checkpoint {ckpt_path} not found. Starting from scratch.")
        ckpt_path = None
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
