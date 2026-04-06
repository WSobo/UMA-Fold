"""
src/models/pairmixer_block.py
─────────────────────────────────────────────────────────────────────────────
UMA-Fold Pairmixer Block
========================
An *attention-free* pair-representation update block.

Design axioms
─────────────
* NO nn.MultiheadAttention, sequence attention, or triangle attention.
* Geometric reasoning is preserved entirely through **triangle multiplications**
  implemented with torch.einsum, followed by pair feed-forward networks (FFN).
* A "low-norm dropout" mechanism drops the lowest-magnitude features inside
  triangle operations to reduce memory footprint on 24 GB VRAM GPUs.

Triangle multiplication recap
──────────────────────────────
Given a pair tensor  z  of shape [B, N, N, d]:

  Outgoing triangle update (Alg. 11 in AF2 supplement):
      m_{ij} = LayerNorm(z_{ij})
      a_{ij} = sigmoid(gate_a) ⊙ linear_a(m_{ij})     # "left projection"
      b_{ij} = sigmoid(gate_b) ⊙ linear_b(m_{ij})     # "right projection"
      # key einsum: sum over shared index k
      t_{ij} = einsum("b i k d, b j k d -> b i j d", a, b)
      z_{ij} += sigmoid(gate_out) ⊙ LayerNorm(t_{ij})

  Incoming triangle update (Alg. 12):
      # key einsum: sum over shared index k (transposed role)
      t_{ij} = einsum("b k i d, b k j d -> b i j d", a, b)

Low-norm dropout
────────────────
Before the einsum, features whose L2-norm across the channel dimension falls
below the `keep_fraction` quantile are *zeroed out* (masked to zero).
This is a structured form of feature-level dropout that preferentially removes
low-information features and is differentiable at inference (mask is
deterministically computed; at training it adds a stochastic element via
`torch.rand`).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─── helpers ──────────────────────────────────────────────────────────────────

def _low_norm_dropout(
    x: torch.Tensor,
    keep_fraction: float = 0.75,
    training: bool = True,
) -> torch.Tensor:
    """Zero out the lowest-magnitude features to save memory in triangle ops.

    Args:
        x: Tensor of shape [B, N, N, d] or [B, N, d].
        keep_fraction: Fraction of features to *keep* (0 < f ≤ 1).
            Features with L2 norm below the (1-keep_fraction) quantile are
            dropped.
        training: When False, returns x unchanged (no stochastic behaviour at
            inference).

    Returns:
        Tensor with the same shape as x; low-norm positions are set to zero.
    """
    if keep_fraction >= 1.0 or not training:
        return x

    # Compute per-feature norms over the channel (last) dimension.
    # shape: [..., 1]
    norms = x.norm(dim=-1, keepdim=True)          # [B, N, N, 1]

    # Derive the quantile threshold at the (1-keep_fraction) level.
    flat_norms = norms.reshape(-1)
    threshold = torch.quantile(flat_norms, 1.0 - keep_fraction)

    # Binary mask: 1 → keep, 0 → drop.
    mask = (norms >= threshold).to(x.dtype)       # [B, N, N, 1] broadcast

    # Scale kept values to preserve expected activation magnitude.
    return x * mask / (keep_fraction + 1e-8)


def _gate_linear(in_dim: int, out_dim: int) -> tuple[nn.Linear, nn.Linear]:
    """Return a (projection, gate) pair for a gated linear unit."""
    proj = nn.Linear(in_dim, out_dim, bias=False)
    gate = nn.Linear(in_dim, out_dim, bias=False)
    return proj, gate


# ─── Triangle multiplication ──────────────────────────────────────────────────

class TriangleMultiplication(nn.Module):
    """Single outgoing *or* incoming triangle multiplication update.

    References
    ──────────
    Jumper et al. (2021) AlphaFold2, Supplementary Algorithm 11 / 12.

    Notation
    ─────────
    B  – batch size
    N  – sequence length
    d  – pair_dim (hidden channel dimension)
    """

    def __init__(
        self,
        pair_dim: int,
        direction: str = "outgoing",  # "outgoing" | "incoming"
        low_norm_dropout_fraction: float = 0.75,
        low_norm_dropout_enabled: bool = True,
    ) -> None:
        super().__init__()
        if direction not in {"outgoing", "incoming"}:
            raise ValueError(f"direction must be 'outgoing' or 'incoming', got {direction!r}")

        self.direction = direction
        self.pair_dim = pair_dim
        self.low_norm_dropout_fraction = low_norm_dropout_fraction
        self.low_norm_dropout_enabled = low_norm_dropout_enabled

        # Input layer norm
        self.layer_norm_in = nn.LayerNorm(pair_dim)

        # Left / right projections with gating (Gated Linear Units)
        self.proj_a, self.gate_a = _gate_linear(pair_dim, pair_dim)
        self.proj_b, self.gate_b = _gate_linear(pair_dim, pair_dim)

        # Output gate + layer norm
        self.gate_out = nn.Linear(pair_dim, pair_dim, bias=False)
        self.layer_norm_out = nn.LayerNorm(pair_dim)

        # Final output projection (maps d → d after triangle mixing)
        self.proj_out = nn.Linear(pair_dim, pair_dim, bias=False)

        self._init_weights()

    # ── initialisation ────────────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

        # Zero-init the gate_out weight so the output gate starts closed,
        # letting the residual stream dominate at the beginning of training.
        nn.init.zeros_(self.gate_out.weight)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute one triangle multiplication update.

        Args:
            z: Pair representation of shape [B, N, N, d].

        Returns:
            Updated pair representation of the same shape.
        """
        # z: [B, N, N, d]
        residual = z
        m = self.layer_norm_in(z)                     # [B, N, N, d]

        # ── Gated left and right projections ─────────────────────────────────
        # a_{ij} = sigmoid(gate_a(m_{ij})) ⊙ proj_a(m_{ij})
        a = torch.sigmoid(self.gate_a(m)) * self.proj_a(m)   # [B, N, N, d]
        b = torch.sigmoid(self.gate_b(m)) * self.proj_b(m)   # [B, N, N, d]

        # ── Low-norm dropout (structured memory-saving regularisation) ────────
        if self.low_norm_dropout_enabled:
            a = _low_norm_dropout(a, self.low_norm_dropout_fraction, self.training)
            b = _low_norm_dropout(b, self.low_norm_dropout_fraction, self.training)

        # ── Triangle mixing via einsum ────────────────────────────────────────
        if self.direction == "outgoing":
            # Outgoing: for each pair (i,j) sum over shared neighbour k.
            # Einsum axes:  B=batch, i=row-index, j=col-index,
            #               k=shared-index, d=channel
            # t_{b,i,j,d} = Σ_k  a_{b,i,k,d} · b_{b,j,k,d}
            t = torch.einsum("b i k d, b j k d -> b i j d", a, b)
        else:
            # Incoming: roles of i and k are transposed.
            # t_{b,i,j,d} = Σ_k  a_{b,k,i,d} · b_{b,k,j,d}
            t = torch.einsum("b k i d, b k j d -> b i j d", a, b)

        # ── Output gate + residual ────────────────────────────────────────────
        t = self.layer_norm_out(t)                    # [B, N, N, d]
        gate = torch.sigmoid(self.gate_out(m))        # [B, N, N, d]
        out = gate * self.proj_out(t)                 # [B, N, N, d]

        return residual + out


# ─── Pair Feed-Forward Network ────────────────────────────────────────────────

class PairFFN(nn.Module):
    """Position-wise feed-forward network applied independently to each (i, j).

    Architecture: LayerNorm → Linear → SiLU → Linear → residual add.
    """

    def __init__(self, pair_dim: int, expansion: int = 4) -> None:
        super().__init__()
        hidden = pair_dim * expansion
        self.norm = nn.LayerNorm(pair_dim)
        self.fc1 = nn.Linear(pair_dim, hidden)
        self.fc2 = nn.Linear(hidden, pair_dim)

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, N, N, d]
        Returns:
            [B, N, N, d]
        """
        return z + self.fc2(F.silu(self.fc1(self.norm(z))))


# ─── PairmixerBlock ───────────────────────────────────────────────────────────

class PairmixerBlock(nn.Module):
    """One complete Pairmixer block: the core attention-free pair update.

    Processing order (per block)
    ─────────────────────────────
    1. Outgoing triangle multiplication  (Σ over shared right-neighbour k)
    2. Incoming triangle multiplication  (Σ over shared left-neighbour k)
    3. Pair feed-forward network

    No sequence attention or triangle attention is used anywhere.

    Args:
        pair_dim:               Hidden dimension of the pair representation.
        ffn_expansion:          Expansion factor for the pair FFN hidden layer.
        dropout_rate:           Standard dropout after each sub-layer.
        low_norm_dropout_enabled:
                                Whether to apply low-norm dropout inside
                                triangle multiplications.
        low_norm_dropout_fraction:
                                Fraction of features to *keep* during
                                low-norm dropout (remainder is zeroed out).
    """

    def __init__(
        self,
        pair_dim: int = 128,
        ffn_expansion: int = 4,
        dropout_rate: float = 0.1,
        low_norm_dropout_enabled: bool = True,
        low_norm_dropout_fraction: float = 0.75,
    ) -> None:
        super().__init__()

        tri_kwargs = dict(
            pair_dim=pair_dim,
            low_norm_dropout_fraction=low_norm_dropout_fraction,
            low_norm_dropout_enabled=low_norm_dropout_enabled,
        )

        # Sub-layers (no attention of any kind)
        self.tri_out = TriangleMultiplication(direction="outgoing", **tri_kwargs)
        self.tri_in = TriangleMultiplication(direction="incoming", **tri_kwargs)
        self.ffn = PairFFN(pair_dim=pair_dim, expansion=ffn_expansion)

        # Standard dropout applied after each sub-layer
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Update the pair representation for one Pairmixer block.

        Args:
            z: Pair representation tensor, shape [B, N, N, d_pair].
               Heavy operations here run in bfloat16 (see precision.py).

        Returns:
            Updated pair tensor of the same shape [B, N, N, d_pair].
        """
        # 1. Outgoing triangle multiplication
        #    Captures "i and j share a common right-neighbour k"
        z = z + self.dropout(self.tri_out(z))

        # 2. Incoming triangle multiplication
        #    Captures "i and j share a common left-neighbour k"
        z = z + self.dropout(self.tri_in(z))

        # 3. Pair feed-forward network
        #    Mixes information across channels for each (i, j) position
        z = self.ffn(z)

        return z
