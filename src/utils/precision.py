"""
src/utils/precision.py
──────────────────────
Mixed-precision helpers for UMA-Fold.

Strategy
─────────
* Heavy trunk operations (triangle multiplications, pair FFN) → bfloat16.
  This halves memory bandwidth on Ampere/Hopper GPUs without the overflow
  risk of float16.
* Numerically sensitive operations (softmax, cross-entropy loss, coordinate
  prediction) → float32.

Usage
─────
    from src.utils.precision import cast_to_trunk_dtype

    with cast_to_trunk_dtype(torch.bfloat16):
        z = block(z)   # runs in bfloat16

    loss = F.cross_entropy(logits.float(), targets)  # back in fp32
"""

from __future__ import annotations

import contextlib
from typing import Generator

import torch


@contextlib.contextmanager
def cast_to_trunk_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """Context manager that sets the default autocast dtype for the trunk.

    When dtype is torch.bfloat16 this is equivalent to
    ``torch.autocast(device_type="cuda", dtype=torch.bfloat16)``.
    When dtype is torch.float32 (e.g. in CPU-only test environments) the
    context manager is a no-op.

    Args:
        dtype: The dtype to use inside the context (typically torch.bfloat16).

    Yields:
        None
    """
    if dtype == torch.float32:
        yield
        return

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.autocast(device_type=device_type, dtype=dtype):
        yield


def to_fp32(*tensors: torch.Tensor) -> list[torch.Tensor]:
    """Cast an arbitrary number of tensors to float32.

    Useful for ensuring numerically sensitive operations (softmax, loss)
    are computed in full precision regardless of the current autocast context.

    Args:
        *tensors: Any number of torch.Tensor objects.

    Returns:
        List of the same tensors cast to torch.float32.
    """
    return [t.float() for t in tensors]
