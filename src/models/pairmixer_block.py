"""
src/models/pairmixer_block.py
─────────────────────────────
Implementation of the PairMixer Block as described in:
"Triangle Multiplication is All You Need for Biomolecular Structure Representations"

This module strips out the computationally expensive Sequence Updates and 
Triangle Attention layers found in normal AlphaFold3/Boltz Pairformer blocks.
It relies purely on Triangle Multiplication for spatial reasoning.
"""

import torch
from torch import nn, Tensor

# We can leverage Boltz's well-tested low-level primitives instead of writing 
# our own Einsum-heavy triangle multiplications.
from boltz.model.layers.triangular_mult import TriangleMultiplication
from boltz.model.layers.transition import Transition


class PairMixerBlock(nn.Module):
    """
    A single layer of the PairMixer backbone.
    Updates only the pair representation (z) using Triangle Multiplication.
    The sequence representation (s) passes through untouched.
    """
    
    def __init__(self, c_z: int, c_hidden_mul: int = 128, drop_rate: float = 0.0):
        """
        Args:
            c_z: Dimension of the pair representation.
            c_hidden_mul: Hidden dimension inside the triangle multiplication.
            drop_rate: Dropout probability.
        """
        super().__init__()
        
        # 1. Triangle Multiplication (Incoming Edges)
        self.tri_mul_in = TriangleMultiplication(
            c_z=c_z, 
            c_hidden=c_hidden_mul, 
            outgoing=False
        )
        
        # 2. Triangle Multiplication (Outgoing Edges)
        self.tri_mul_out = TriangleMultiplication(
            c_z=c_z, 
            c_hidden=c_hidden_mul, 
            outgoing=True
        )
        
        # 3. Pair Transition (Feed-Forward Network applied across all pairs)
        # Note: Boltz uses a Transition module which acts as a standard FFN layer with LayerNorm
        self.transition = Transition(
            d_in=c_z, 
            n_scale=4, 
            drop_rate=drop_rate
        )

    def forward(self, z: Tensor, mask: Tensor | None = None) -> Tensor:
        """
        Forward pass for a PairMixer block.
        
        Args:
            z: Pair representation tensor of shape [B, L, L, C_z]
            mask: Optional pair mask of shape [B, L, L] (1=valid, 0=padding)
            
        Returns:
            z_out: Updated pair representation of shape [B, L, L, C_z]
        """
        # --- 1. Triangle Multiplication (Incoming) ---
        z = z + self.tri_mul_in(z, mask=mask)
        
        # --- 2. Triangle Multiplication (Outgoing) ---
        z = z + self.tri_mul_out(z, mask=mask)
        
        # --- 3. Pair Transition (FFN) ---
        z = z + self.transition(z)

        return z
