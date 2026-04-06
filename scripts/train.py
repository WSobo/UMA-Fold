#!/usr/bin/env python
"""
scripts/train.py
────────────────
Entry point for training UMA-Fold.

Usage
─────
    # Default config
    python scripts/train.py

    # Override individual settings via Hydra CLI
    python scripts/train.py model.pair_dim=256 trainer.max_epochs=50

    # W&B disabled
    python scripts/train.py wandb.enabled=false

    # CPU smoke-test (no GPU required)
    python scripts/train.py trainer.accelerator=cpu trainer.devices=1
"""

import logging
import sys
from pathlib import Path

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger

# Allow `python scripts/train.py` without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datamodule import ProteinDataModule
from src.training.lightning_module import UMAFoldLightningModule

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    pl.seed_everything(42, workers=True)

    # ── Logger (W&B) ─────────────────────────────────────────────────────────
    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            entity=cfg.wandb.entity or None,
            tags=list(cfg.wandb.tags),
            notes=cfg.wandb.notes,
            log_model=True,
        )

    # ── Callbacks ────────────────────────────────────────────────────────────
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
    ]

    if cfg.trainer.enable_checkpointing:
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.paths.checkpoint_dir,
                filename="uma_fold-{epoch:03d}-{val_loss:.4f}",
                monitor=cfg.trainer.checkpoint_monitor,
                mode=cfg.trainer.checkpoint_mode,
                save_top_k=cfg.trainer.checkpoint_save_top_k,
                save_last=True,
            )
        )

    if cfg.trainer.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.trainer.early_stopping.monitor,
                patience=cfg.trainer.early_stopping.patience,
                mode=cfg.trainer.early_stopping.mode,
            )
        )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
    )

    # ── Model & Data ─────────────────────────────────────────────────────────
    model = UMAFoldLightningModule(cfg)
    datamodule = ProteinDataModule(
        data_dir=cfg.paths.data_dir,
        train_split=cfg.data.train_split,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        max_seq_len=cfg.data.max_seq_len,
    )

    # ── Fit ───────────────────────────────────────────────────────────────────
    trainer.fit(model, datamodule=datamodule)

    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
