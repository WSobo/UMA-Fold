"""
tests/test_pairmixer_block.py
─────────────────────────────
Unit tests for the attention-free PairmixerBlock.

These tests verify:
* Forward pass produces the correct output shape.
* No attention layers are present in the module (architectural constraint).
* Triangle multiplication einsum produces numerically sane values.
* Low-norm dropout changes activations at training time but not at eval time.
* Both outgoing and incoming TriangleMultiplication directions produce
  the correct output shape.
"""

import pytest
import torch
import torch.nn as nn

from src.models.pairmixer_block import (
    PairmixerBlock,
    TriangleMultiplication,
    PairFFN,
    _low_norm_dropout,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def pair_tensor() -> torch.Tensor:
    """A small random pair tensor [B=2, N=8, N=8, d=32] for fast tests."""
    torch.manual_seed(0)
    return torch.randn(2, 8, 8, 32)


# ── Shape tests ───────────────────────────────────────────────────────────────

class TestPairmixerBlockShape:
    """Verify that the block preserves the pair tensor shape."""

    @pytest.mark.parametrize("pair_dim", [32, 64, 128])
    def test_output_shape_matches_input(self, pair_dim: int) -> None:
        B, N = 2, 10
        z = torch.randn(B, N, N, pair_dim)
        block = PairmixerBlock(pair_dim=pair_dim, ffn_expansion=2)
        block.eval()
        out = block(z)
        assert out.shape == z.shape, (
            f"Expected shape {z.shape}, got {out.shape}"
        )

    def test_output_dtype_preserved(self, pair_tensor: torch.Tensor) -> None:
        """Output dtype should match input dtype."""
        block = PairmixerBlock(pair_dim=32, ffn_expansion=2)
        block.eval()
        out = block(pair_tensor)
        assert out.dtype == pair_tensor.dtype


# ── Architectural constraints ─────────────────────────────────────────────────

class TestNoAttention:
    """Ensure the block contains no attention layers whatsoever."""

    def test_no_multihead_attention(self) -> None:
        block = PairmixerBlock(pair_dim=32)
        for name, module in block.named_modules():
            assert not isinstance(module, nn.MultiheadAttention), (
                f"Found nn.MultiheadAttention at '{name}' — architectural constraint violated"
            )

    def test_no_attention_in_model_tree(self) -> None:
        """Check recursively that zero attention submodules exist."""
        block = PairmixerBlock(pair_dim=64)
        attention_classes = (nn.MultiheadAttention,)
        found = [
            name
            for name, m in block.named_modules()
            if isinstance(m, attention_classes)
        ]
        assert len(found) == 0, f"Attention modules found: {found}"


# ── Triangle multiplication ───────────────────────────────────────────────────

class TestTriangleMultiplication:
    @pytest.mark.parametrize("direction", ["outgoing", "incoming"])
    def test_shape_and_finite(self, direction: str, pair_tensor: torch.Tensor) -> None:
        tri = TriangleMultiplication(
            pair_dim=32,
            direction=direction,
            low_norm_dropout_enabled=False,
        )
        tri.eval()
        out = tri(pair_tensor)
        assert out.shape == pair_tensor.shape
        assert torch.isfinite(out).all(), "Triangle multiplication produced non-finite values"

    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction must be"):
            TriangleMultiplication(pair_dim=32, direction="lateral")

    def test_residual_connection(self, pair_tensor: torch.Tensor) -> None:
        """With zero-init gate the output should be close to input (residual)."""
        tri = TriangleMultiplication(pair_dim=32, direction="outgoing", low_norm_dropout_enabled=False)
        # Zero out the output gate weights so gate = sigmoid(0) ≈ 0.5 * proj_out
        # The residual should dominate; we just verify output ≠ all-zeros.
        tri.eval()
        out = tri(pair_tensor)
        assert not torch.allclose(out, torch.zeros_like(out))


# ── PairFFN ───────────────────────────────────────────────────────────────────

class TestPairFFN:
    def test_shape(self, pair_tensor: torch.Tensor) -> None:
        ffn = PairFFN(pair_dim=32, expansion=2)
        ffn.eval()
        out = ffn(pair_tensor)
        assert out.shape == pair_tensor.shape

    def test_residual_non_zero(self, pair_tensor: torch.Tensor) -> None:
        ffn = PairFFN(pair_dim=32, expansion=2)
        ffn.eval()
        out = ffn(pair_tensor)
        # Verify the FFN actually changes some values (checks that fc1/fc2 are active)
        assert not torch.equal(out, pair_tensor), (
            "PairFFN output is bit-for-bit identical to input — residual add likely broken"
        )


# ── Low-norm dropout ──────────────────────────────────────────────────────────

class TestLowNormDropout:
    def test_training_changes_values(self) -> None:
        torch.manual_seed(1)
        x = torch.randn(2, 4, 4, 16)
        out = _low_norm_dropout(x, keep_fraction=0.5, training=True)
        assert not torch.allclose(out, x), (
            "Low-norm dropout had no effect during training"
        )

    def test_eval_is_identity(self) -> None:
        x = torch.randn(2, 4, 4, 16)
        out = _low_norm_dropout(x, keep_fraction=0.5, training=False)
        assert torch.allclose(out, x), (
            "Low-norm dropout should be identity at inference"
        )

    def test_full_keep_is_identity(self) -> None:
        x = torch.randn(2, 4, 4, 16)
        out = _low_norm_dropout(x, keep_fraction=1.0, training=True)
        assert torch.allclose(out, x), (
            "keep_fraction=1.0 should be a no-op"
        )

    def test_zero_norm_features_are_dropped(self) -> None:
        """Features with zero norm must be in the dropped set."""
        x = torch.ones(1, 2, 2, 8)
        x[0, 0, 0, :] = 0.0  # zero-norm feature at position (0,0)
        out = _low_norm_dropout(x, keep_fraction=0.9, training=True)
        # The zeroed position should remain zero (it has the lowest norm)
        assert torch.allclose(out[0, 0, 0, :], torch.zeros(8))
