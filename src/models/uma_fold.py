"""
src/models/uma_fold.py
──────────────────────
Top-level UMA-Fold model.

Architecture overview
─────────────────────
1. Input embedding  : token IDs → single representation (d_single)
                      outer-product mix → initial pair representation (d_pair)
2. Pairmixer trunk  : N stacked PairmixerBlocks (attention-free)
3. Structure head   : pair representation → per-residue 3-D coordinates
                      (simplified linear head; replace with IPA or diffusion
                      head for a full production model)

All trunk computations run in bfloat16 via the cast_to_trunk_dtype context
provided in src/utils/precision.py.  Numerically sensitive operations
(softmax, loss) should be kept in fp32 by the caller.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.models.pairmixer_block import PairmixerBlock
from src.utils.precision import cast_to_trunk_dtype


class InputEmbedding(nn.Module):
    """Map token IDs to single + pair representations.

    Args:
        num_tokens: Vocabulary size (default 21: 20 AAs + unknown).
        single_dim: Dimension of per-residue single representation.
        pair_dim:   Dimension of pair representation.
    """

    def __init__(self, num_tokens: int = 21, single_dim: int = 64, pair_dim: int = 128) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(num_tokens, single_dim)
        # Pair initialisation: outer-product of single embeddings projected to pair_dim
        self.pair_proj = nn.Linear(single_dim * single_dim, pair_dim, bias=False)

        nn.init.xavier_uniform_(self.pair_proj.weight)

    def forward(self, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            token_ids: [B, N] integer token IDs.

        Returns:
            single: [B, N, d_single]
            pair:   [B, N, N, d_pair]
        """
        # Single representation
        single = self.token_embed(token_ids)          # [B, N, d_s]

        # Pair initialisation via outer product
        # outer_{b,i,j,d_s*d_s} = single_{b,i,:} ⊗ single_{b,j,:}
        s_i = single.unsqueeze(2)                     # [B, N, 1, d_s]
        s_j = single.unsqueeze(1)                     # [B, 1, N, d_s]
        outer = (s_i * s_j).reshape(
            single.shape[0], single.shape[1], single.shape[1], -1
        )                                             # [B, N, N, d_s^2]
        pair = self.pair_proj(outer)                  # [B, N, N, d_pair]

        return single, pair


class StructureHead(nn.Module):
    """Lightweight head: pair representation → 3-D Cα coordinates.

    This is a placeholder.  A production head should implement an Invariant
    Point Attention (IPA) or diffusion-based coordinate decoder.

    Args:
        pair_dim:   Input pair representation dimension.
        single_dim: Input single representation dimension.
    """

    def __init__(self, pair_dim: int = 128, single_dim: int = 64) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(single_dim + pair_dim)
        self.coord_head = nn.Linear(single_dim + pair_dim, 3, bias=True)        nn.init.zeros_(self.coord_head.weight)
        nn.init.zeros_(self.coord_head.bias)

    def forward(self, single: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            single: [B, N, d_single]
            pair:   [B, N, N, d_pair]   (summed over j to collapse pair dim)

        Returns:
            coords: [B, N, 3] predicted Cα coordinates
        """
        # Aggregate pair information per residue i (mean over j)
        pair_agg = pair.mean(dim=2)                   # [B, N, d_pair]
        combined = torch.cat([single, pair_agg], dim=-1)  # [B, N, d_s+d_p]
        # Cast to fp32: LayerNorm computes running mean/variance which can
        # overflow or underflow in bfloat16, causing numerical instability.
        combined = self.norm(combined.float())
        return self.coord_head(combined)              # [B, N, 3]


class UMAFold(nn.Module):
    """UMA-Fold: Ultra-lightweight, attention-free biomolecular structure predictor.

    Args:
        num_tokens:                    Vocabulary size.
        single_dim:                    Single representation dimension.
        pair_dim:                      Pair representation dimension.
        num_blocks:                    Number of PairmixerBlocks.
        ffn_expansion:                 FFN expansion factor.
        dropout_rate:                  Standard dropout probability.
        low_norm_dropout_enabled:      Enable low-norm dropout in triangle ops.
        low_norm_dropout_fraction:     Keep-fraction for low-norm dropout.
        trunk_dtype:                   dtype for trunk computations (bfloat16).
    """

    def __init__(
        self,
        num_tokens: int = 21,
        single_dim: int = 64,
        pair_dim: int = 128,
        num_blocks: int = 8,
        ffn_expansion: int = 4,
        dropout_rate: float = 0.1,
        low_norm_dropout_enabled: bool = True,
        low_norm_dropout_fraction: float = 0.75,
        trunk_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.trunk_dtype = trunk_dtype

        self.input_embedding = InputEmbedding(
            num_tokens=num_tokens,
            single_dim=single_dim,
            pair_dim=pair_dim,
        )

        self.blocks = nn.ModuleList(
            [
                PairmixerBlock(
                    pair_dim=pair_dim,
                    ffn_expansion=ffn_expansion,
                    dropout_rate=dropout_rate,
                    low_norm_dropout_enabled=low_norm_dropout_enabled,
                    low_norm_dropout_fraction=low_norm_dropout_fraction,
                )
                for _ in range(num_blocks)
            ]
        )

        self.structure_head = StructureHead(pair_dim=pair_dim, single_dim=single_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """End-to-end forward pass.

        Args:
            token_ids: [B, N] integer token IDs.

        Returns:
            coords: [B, N, 3] predicted Cα coordinates (float32).
        """
        single, pair = self.input_embedding(token_ids)

        # Run Pairmixer trunk in bfloat16 for memory efficiency
        with cast_to_trunk_dtype(self.trunk_dtype):
            single = single.to(self.trunk_dtype)
            pair = pair.to(self.trunk_dtype)
            for block in self.blocks:
                pair = block(pair)

        # Structure head operates in fp32 for numerical stability
        coords = self.structure_head(single.float(), pair.float())
        return coords
