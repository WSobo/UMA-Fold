"""
src/training/lightning_module.py
─────────────────────────────────
PyTorch Lightning LightningModule for UMA-Fold.

Responsibilities
─────────────────
* Wraps UMAFold.
* Computes a coordinate regression loss (FAPE proxy: mean squared error on
  Cα positions — replace with full FAPE for production).
* Logs training/validation metrics to Weights & Biases via W&B logger.
* Configures the AdamW optimiser with cosine annealing scheduler.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

# Small constant to prevent division by zero in masked loss computation.
_LOSS_EPS = 1e-8
import pytorch_lightning as pl
from omegaconf import DictConfig

from src.models.uma_fold import UMAFold


class UMAFoldLightningModule(pl.LightningModule):
    """LightningModule encapsulating UMAFold training and validation.

    Args:
        cfg: Hydra configuration (OmegaConf DictConfig).  The module reads:
             - cfg.model.*   for architecture hyperparameters
             - cfg.trainer.* for optimisation settings
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        m = cfg.model
        self.model = UMAFold(
            num_tokens=m.num_tokens,
            single_dim=m.single_dim,
            pair_dim=m.pair_dim,
            num_blocks=m.num_blocks,
            ffn_expansion=m.ffn_expansion,
            dropout_rate=m.dropout_rate,
            low_norm_dropout_enabled=m.low_norm_dropout.enabled,
            low_norm_dropout_fraction=m.low_norm_dropout.keep_fraction,
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model(token_ids)

    # ── Loss ──────────────────────────────────────────────────────────────────

    def _compute_loss(
        self,
        batch: dict[str, torch.Tensor],
        prefix: str,
    ) -> torch.Tensor:
        """Compute MSE loss on masked Cα coordinates (fp32 for stability).

        Args:
            batch:  Dict with keys "token_ids" [B,N], "coords" [B,N,3],
                    "mask" [B,N].
            prefix: "train" or "val" — used as the metric prefix.

        Returns:
            Scalar loss tensor (float32).
        """
        pred = self.forward(batch["token_ids"])        # [B, N, 3]
        target = batch["coords"].float()               # [B, N, 3]
        mask = batch["mask"].unsqueeze(-1).float()     # [B, N, 1]

        # Mean squared error over valid (non-padded) positions
        # Numerically sensitive: kept in fp32.
        loss = (F.mse_loss(pred * mask, target * mask, reduction="sum")
                / (mask.sum() + _LOSS_EPS))

        self.log(f"{prefix}/loss", loss, prog_bar=True, on_step=(prefix == "train"),
                 on_epoch=True, sync_dist=True)
        return loss

    # ── Lightning hooks ───────────────────────────────────────────────────────

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._compute_loss(batch, "train")

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._compute_loss(batch, "val")

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        self._compute_loss(batch, "test")

    # ── Optimiser & scheduler ─────────────────────────────────────────────────

    def configure_optimizers(self) -> dict[str, Any]:
        """AdamW with cosine-annealing learning-rate schedule."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-4,
            weight_decay=1e-2,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.trainer.max_epochs,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val/loss",
            },
        }
