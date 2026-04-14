"""
src/models/layers.py
─────────────────────
Core reusable layer primitives for UMA-Fold.

These replace the corresponding boltz.model.layers.* imports so the project
no longer requires a local boltz clone for basic building blocks.
"""

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional


class LinearNoBias(nn.Linear):
    """nn.Linear with bias permanently disabled. Drop-in for boltz's LinearNoBias."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)


class Transition(nn.Module):
    """SiLU-gated 2-layer MLP with pre-LayerNorm (SwiGLU variant).

    Replaces boltz.model.layers.transition.Transition.

    Architecture: LayerNorm → [SiLU(fc1(x)) * fc2(x)] → fc3
    - fc3 zero-initialized (AlphaFold/AF3 convention — stable early training)
    - No bias on linear layers (consistent with rest of architecture)
    """

    def __init__(
        self,
        dim: int,
        hidden: Optional[int] = None,
        out_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        hidden = hidden if hidden is not None else dim * 4
        out_dim = out_dim if out_dim is not None else dim

        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fc1 = LinearNoBias(dim, hidden)
        self.fc2 = LinearNoBias(dim, hidden)
        self.fc3 = LinearNoBias(hidden, out_dim)

        # Zero-init output projection: keeps residual stream unchanged at init,
        # allowing the block to contribute gradually as training stabilises.
        nn.init.zeros_(self.fc3.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        return self.fc3(F.silu(self.fc1(x)) * self.fc2(x))
