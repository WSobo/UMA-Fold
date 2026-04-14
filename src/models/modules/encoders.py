"""
src/models/modules/encoders.py
────────────────────────────────
Positional encoding modules for UMA-Fold.

Replaces boltz.model.modules.encoders.RelativePositionEncoder so the project
no longer requires a local boltz clone for this component.
"""

import torch
from torch import nn, Tensor
from torch.nn.functional import one_hot

from src.models.layers import LinearNoBias


class RelativePositionEncoder(nn.Module):
    """Encodes relative positions between token pairs into pair embeddings.

    Ports boltz's RelativePositionEncoder 1-to-1; the only external dependency
    removed is boltz's LinearNoBias (replaced with ours) and boltz.data.const
    (no constants needed here — r_max/s_max are constructor params).

    The forward pass reads these keys from the feature dict:
        asym_id       [B, L]   chain identity per token
        residue_index [B, L]   residue position within chain
        entity_id     [B, L]   entity (unique sequence) identity
        cyclic_period [B, L]   >0 for cyclic peptides, else 0
        token_index   [B, L]   token position within residue (for multi-token residues)
        sym_id        [B, L]   symmetry copy index

    Output: pair embedding [B, L, L, token_z]
    """

    def __init__(self, token_z: int, r_max: int = 32, s_max: int = 2) -> None:
        """
        Parameters
        ----------
        token_z : int
            Output pair-representation dimension.
        r_max : int
            Maximum residue-index distance to one-hot encode (clipped beyond ±r_max).
        s_max : int
            Maximum symmetry-copy distance to one-hot encode (clipped beyond ±s_max).
        """
        super().__init__()
        self.r_max = r_max
        self.s_max = s_max
        # Input feature size:
        #   a_rel_pos   : 2*r_max + 2  (residue distance one-hot, +1 bucket for cross-chain)
        #   a_rel_token : 2*r_max + 2  (token distance one-hot within same residue)
        #   b_same_entity: 1
        #   a_rel_chain : 2*s_max + 2  (symmetry-copy distance one-hot)
        in_features = 4 * (r_max + 1) + 2 * (s_max + 1) + 1
        self.linear = LinearNoBias(in_features, token_z)

    def forward(self, feats: dict) -> Tensor:
        r_max = self.r_max
        s_max = self.s_max

        b_same_chain = torch.eq(
            feats["asym_id"][:, :, None], feats["asym_id"][:, None, :]
        )
        b_same_residue = torch.eq(
            feats["residue_index"][:, :, None], feats["residue_index"][:, None, :]
        )
        b_same_entity = torch.eq(
            feats["entity_id"][:, :, None], feats["entity_id"][:, None, :]
        )

        rel_pos = (
            feats["residue_index"][:, :, None] - feats["residue_index"][:, None, :]
        )

        # Handle cyclic peptides: wrap relative position by the cyclic period
        if torch.any(feats["cyclic_period"] != 0):
            period = torch.where(
                feats["cyclic_period"] > 0,
                feats["cyclic_period"],
                torch.zeros_like(feats["cyclic_period"]) + 10000,
            ).unsqueeze(1)
            rel_pos = (rel_pos - period * torch.round(rel_pos / period)).long()

        # Residue-level relative position (cross-chain tokens get an overflow bucket)
        d_residue = torch.clip(rel_pos + r_max, 0, 2 * r_max)
        d_residue = torch.where(
            b_same_chain, d_residue, torch.zeros_like(d_residue) + 2 * r_max + 1
        )
        a_rel_pos = one_hot(d_residue, 2 * r_max + 2)

        # Token-level relative position within the same residue
        d_token = torch.clip(
            feats["token_index"][:, :, None] - feats["token_index"][:, None, :] + r_max,
            0,
            2 * r_max,
        )
        d_token = torch.where(
            b_same_chain & b_same_residue,
            d_token,
            torch.zeros_like(d_token) + 2 * r_max + 1,
        )
        a_rel_token = one_hot(d_token, 2 * r_max + 2)

        # Symmetry-copy relative position
        d_chain = torch.clip(
            feats["sym_id"][:, :, None] - feats["sym_id"][:, None, :] + s_max, 0, 2 * s_max
        )
        d_chain = torch.where(
            b_same_chain, torch.zeros_like(d_chain) + 2 * s_max + 1, d_chain
        )
        a_rel_chain = one_hot(d_chain, 2 * s_max + 2)

        pair_feat = torch.cat(
            [
                a_rel_pos.float(),
                a_rel_token.float(),
                b_same_entity.unsqueeze(-1).float(),
                a_rel_chain.float(),
            ],
            dim=-1,
        )
        return self.linear(pair_feat)
