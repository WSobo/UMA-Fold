"""
src/training/lightning_module.py
────────────────────────────────
PyTorch Lightning module for UMA-Fold training.
Engineered using 2026 MLOps standards and PyTorch 2.x native CUDA optimizations.
"""

import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Any, Dict

from src.models.uma_fold import UMAFold

class UMAFoldLightningModule(pl.LightningModule):
    """
    Lightning Engine for UMA-Fold. 
    Configured precisely for optimal throughput on Ada/Hopper architecture GPUs (RTX 4090/A5500).
    """

    def __init__(self, model_config: dict, lr: float = 1e-3, compile_model: bool = True):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize Architecture
        self.model = UMAFold(model_config)
        
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
        
        # Calculate structure loss safely through the compiled model boundary
        loss_dict = self.model.compute_loss(
            feats=batch,
            out_dict=outputs
        )
        loss = loss_dict["loss"]
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss
        
    def validation_step(self, batch: dict, batch_idx: int) -> None:
        outputs = self(batch)
        loss_dict = self.model.compute_loss(
            feats=batch,
            out_dict=outputs
        )
        val_loss = loss_dict["loss"]
        self.log("val_loss", val_loss, sync_dist=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Fused optimizer implementation avoiding memory bandwidth bottlenecks.
        """
        # fused=False is required when using gradient clipping alongside PyTorch Lightning AMP
        optimizer = AdamW(
            self.model.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=0.01, 
            fused=False 
        )
        
        # Standard cosine annealing schedule matching modern LLM and Protein setup heuristics
        scheduler = CosineAnnealingLR(optimizer, T_max=1000)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }
