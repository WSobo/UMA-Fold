"""
src/models/uma_fold.py
──────────────────────
Main UMA-Fold model architecture.
Implements the PairMixer backbone with gradient checkpointing optimized for 
24GB VRAM consumer GPUs (e.g., A5500, RTX 4090).
"""

import torch
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint
from typing import Optional, Dict

from .pairmixer_block import PairMixerBlock

class PairMixerBackbone(nn.Module):
    """
    The hyper-lightweight PairMixer backbone.
    Stack of PairMixer blocks replacing the original Pairformer.
    Includes built-in Gradient Checkpointing for massive VRAM savings.
    """
    def __init__(
        self,
        num_blocks: int = 12, # 12 layers = Small model from paper, perfect for single GPU
        c_z: int = 128,
        c_hidden_mul: int = 128,
        drop_rate: float = 0.0,
        gradient_checkpointing: bool = True
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.blocks = nn.ModuleList([
            PairMixerBlock(c_z=c_z, c_hidden_mul=c_hidden_mul, drop_rate=drop_rate)
            for _ in range(num_blocks)
        ])

    def forward(self, z: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for block in self.blocks:
            # VRAM optimization: Trade compute for memory by checkpointing activations
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                # use_reentrant=False is the PyTorch 2.x recommended standard
                z = checkpoint(block, z, mask, use_reentrant=False)
            else:
                z = block(z, mask)
        return z


class UMAFold(nn.Module):
    """
    Main UMA-Fold architecture shell linking Embedder, PairMixer Backbone, and Diffusion.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.c_z = config.get("c_z", 128)
        
        # 1. (Placeholder) Input / MSA Embedder
        # We will wrap Boltz's data dictionary into initial s and z representations here.
        # self.embedder = BoltzInputEmbedder(...)
        
        # 2. Our highly-optimized PairMixer Backbone
        self.backbone = PairMixerBackbone(
            num_blocks=config.get("num_blocks", 12),
            c_z=self.c_z,
            c_hidden_mul=config.get("c_hidden_mul", 128),
            drop_rate=config.get("drop_rate", 0.0),
            gradient_checkpointing=config.get("gradient_checkpointing", True)
        )
        
        # 3. (Placeholder) Diffusion Module
        # self.diffusion = BoltzDiffusionModule(...)

    def forward(self, batch: dict) -> dict:
        """
        Forward pass mimicking the structure prediction pipeline.
        """
        # --- Step 1: Embedding ---
        # s_init, z_init = self.embedder(batch)
        
        # NOTE: Placeholder inputs used for structural integrity testing
        # B = batch size, L = sequence length
        B, L = 1, 256 
        device = next(self.parameters()).device
        s_init = torch.zeros((B, L, self.c_z), device=device) # Dummy sequences
        z_init = torch.zeros((B, L, L, self.c_z), device=device) # Dummy pairs
        mask = torch.ones((B, L, L), device=device)
        
        # --- Step 2: Backbone (Only updates z!) ---
        # Notice how s_init bypasses the backbone entirely: massive VRAM memory save!
        z_backbone = self.backbone(z_init, mask)
        
        # --- Step 3: Diffusion ---
        # coords = self.diffusion(s_init, z_backbone)
        
        return {"z_out": z_backbone}  # Returning z_out until diffusion is wired
