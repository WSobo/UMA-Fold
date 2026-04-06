#!/usr/bin/env python
"""
scripts/inference.py
────────────────────
Run UMA-Fold inference on a single amino-acid sequence and save predicted
Cα coordinates to a .npy file.

Usage
─────
    python scripts/inference.py \
        --checkpoint checkpoints/uma_fold-last.ckpt \
        --sequence ACDEFGHIKLMNPQRSTVWY \
        --output outputs/pred_coords.npy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datamodule import sequence_to_tensor
from src.training.lightning_module import UMAFoldLightningModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UMA-Fold inference")
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument("--sequence", required=True, help="Amino-acid sequence string")
    parser.add_argument("--output", default="pred_coords.npy",
                        help="Path to save predicted coordinates (.npy)")
    parser.add_argument("--device", default="cpu", help="Inference device (cpu | cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)

    # ── Load model from checkpoint ────────────────────────────────────────────
    model = UMAFoldLightningModule.load_from_checkpoint(
        args.checkpoint, map_location=device
    )
    model.eval()
    model.to(device)

    # ── Tokenise ──────────────────────────────────────────────────────────────
    token_ids = sequence_to_tensor(args.sequence).unsqueeze(0).to(device)  # [1, N]

    # ── Forward pass ──────────────────────────────────────────────────────────
    with torch.no_grad():
        coords = model(token_ids)  # [1, N, 3]

    coords_np = coords.squeeze(0).cpu().numpy()  # [N, 3]

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, coords_np)
    print(f"Saved predicted Cα coordinates to {out_path}  (shape {coords_np.shape})")


if __name__ == "__main__":
    main()
