"""
src/training/lightning_module.py
────────────────────────────────
PyTorch Lightning module for UMA-Fold training.
Engineered using 2026 MLOps standards and PyTorch 2.x native CUDA optimizations.
"""

import math

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from typing import Any, Dict

from boltz.model.modules.encoders import AtomAttentionEncoder
from src.models.uma_fold import UMAFold


class UMAFoldLightningModule(pl.LightningModule):
    """
    Lightning Engine for UMA-Fold.
    Configured precisely for optimal throughput on Ada/Hopper architecture GPUs (RTX 4090/A5500).
    """

    def __init__(
        self,
        model_config: dict,
        lr: float = 1e-3,
        compile_model: bool = True,
        log_feature_nans: bool = False,
        log_activation_magnitudes: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize Architecture
        self.model = UMAFold(model_config)
        # Diagnostic toggle surfaced by scripts/train.py via cfg.training.log_feature_nans.
        # When True, UMAFold.forward logs input-feature keys that arrive non-finite.
        self.model._log_feature_nans = bool(log_feature_nans)
        # Activation-magnitude tracing inside AtomAttentionEncoder.forward, used to
        # localize the bf16 backward-overflow site. The gate is checked inside the
        # encoder itself, so the attribute must be set on every AtomAttentionEncoder
        # instance in the module tree (UMAFold wires two of them — one in
        # input_embedder, one inside structure_module/AtomDiffusion).
        for m in self.model.modules():
            if isinstance(m, AtomAttentionEncoder):
                m._log_activation_magnitudes = bool(log_activation_magnitudes)

        # --- CUDA Optimizations ---
        # 1. Enable TF32 for matrix multiplications (uses Ampere/Ada Tensor Cores natively)
        torch.set_float32_matmul_precision('high')

        # 2. PyTorch 2.x Graph Compilation
        if compile_model:
            try:
                # default mode is safer for dynamic shapes like varied protein lengths
                # dynamic=True is required to avoid Inductor stride/shape assertion failures on backward pass
                self.model = torch.compile(self.model, dynamic=True)
            except Exception as e:
                print(f"Warning: torch.compile failed. Proceeding eager mode. {e}")

    def forward(self, batch: dict) -> dict:
        return self.model(batch)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self(batch)

        # Loss math runs in fp32 to prevent bf16 overflow in the diffusion MSE
        # ((denoised - target)**2).sum() — squared coord error × weight multipliers
        # (up to 10× for ligand atoms) is the exact overflow site that produced NaN
        # gradients and poisoned checkpoints in the last run.
        with torch.autocast(device_type="cuda", enabled=False):
            out_fp32 = {
                k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in outputs.items()
            }
            loss_dict = self.model.compute_loss(feats=batch, out_dict=out_fp32)

        loss = loss_dict["loss"]

        # Always log something on the monitored key so ModelCheckpoint never sees a
        # missing metric. On NaN: log +inf so mode="min" refuses to select the epoch.
        if not torch.isfinite(loss):
            print(f"[WARNING] Non-finite loss ({loss.item()}) at batch {batch_idx}, skipping.")
            self.log(
                "train_loss",
                torch.tensor(float("inf"), device=loss.device),
                on_step=False, on_epoch=True, sync_dist=True,
            )
            self.log("nonfinite_loss_rate", 1.0, on_step=True, sync_dist=True)
            return None

        self.log("nonfinite_loss_rate", 0.0, on_step=True, sync_dist=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)

        # Per-component breakdown so we can see which term blows up first when it does.
        # Boltz's AtomDiffusion.compute_loss nests the per-term tensors inside a
        # loss_breakdown dict — flatten one level so mse_loss / smooth_lddt_loss /
        # etc. actually show up in wandb.
        for key, val in loss_dict.items():
            if key == "loss":
                continue
            if isinstance(val, dict):
                for subkey, subval in val.items():
                    if torch.is_tensor(subval) and subval.dim() == 0 and torch.isfinite(subval):
                        self.log(f"train_{subkey}", subval.detach(), on_step=True, sync_dist=True)
                continue
            if torch.is_tensor(val) and val.dim() == 0 and torch.isfinite(val):
                self.log(f"train_{key}", val.detach(), on_step=True, sync_dist=True)

        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        """
        Gradient-finiteness guard. Runs after backward, before optimizer.step() and
        before Lightning's gradient clipping. If a gradient is NaN/Inf, zero it so
        the optimizer step becomes a no-op for that parameter instead of propagating
        corruption into the weights. Fixes the silent catastrophe from the prior run
        where NaN gradients poisoned last.ckpt across an entire curriculum stage.
        """
        bad = []
        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue
            if not torch.isfinite(p.grad).all():
                bad.append(name)
                p.grad.zero_()

        if bad:
            self.log(
                "nonfinite_grads",
                float(len(bad)),
                on_step=True, sync_dist=True,
            )
            preview = bad[:10] + ([f"... +{len(bad) - 10} more"] if len(bad) > 10 else [])
            print(
                f"[WARNING] Zeroed non-finite grads on {len(bad)} params "
                f"at step {self.global_step}: {preview}"
            )
        else:
            self.log("nonfinite_grads", 0.0, on_step=True, sync_dist=True)

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        outputs = self(batch)
        with torch.autocast(device_type="cuda", enabled=False):
            out_fp32 = {
                k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                for k, v in outputs.items()
            }
            loss_dict = self.model.compute_loss(feats=batch, out_dict=out_fp32)
        val_loss = loss_dict["loss"]
        self.log("val_loss", val_loss, sync_dist=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        AdamW + warmup → cosine-decay scheduler sized to the full run.
        The previous `T_max=1000` produced a cosine of LR spikes (oscillating between
        lr_max and 0 every 2000 steps) because 100 epochs × 250 steps/epoch ≈ 25,000
        steps — 20× longer than the old T_max. Those spikes directly contributed to
        bf16 overflow at larger crops.
        """
        # fused=False is required when using gradient clipping alongside PyTorch Lightning AMP
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=0.01,
            fused=False,
        )

        # estimated_stepping_batches is valid here — Lightning wires up trainer/datamodule
        # before configure_optimizers() is called
        total_steps = max(int(self.trainer.estimated_stepping_batches), 2)
        warmup_steps = max(1, min(500, total_steps // 20))
        cosine_steps = max(1, total_steps - warmup_steps)

        warmup = LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps),
        )
        cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
