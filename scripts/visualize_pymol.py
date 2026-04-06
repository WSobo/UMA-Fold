#!/usr/bin/env python
"""
scripts/visualize_pymol.py
──────────────────────────
PyMOL automation script: load UMA-Fold predicted Cα coordinates and render
a structural visualisation.

Prerequisites
─────────────
* PyMOL must be installed and on the Python path:
      conda install -c conda-forge pymol-open-source
  or  pip install pymol-open-source  (if a compatible wheel exists)
* The predicted coordinates should be a .npy file of shape [N, 3] produced
  by scripts/inference.py.

Usage
─────
    python scripts/visualize_pymol.py \
        --coords outputs/pred_coords.npy \
        --sequence ACDEFGHIKLMNPQRSTVWY \
        --output outputs/uma_fold_structure.png

    # Launch interactive PyMOL session instead of saving a PNG
    python scripts/visualize_pymol.py \
        --coords outputs/pred_coords.npy \
        --sequence ACDEFGHIKLMNPQRSTVWY \
        --interactive
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise UMA-Fold predictions with PyMOL")
    parser.add_argument("--coords", required=True, help="Path to predicted coords .npy file")
    parser.add_argument("--sequence", required=True, help="Amino-acid sequence string")
    parser.add_argument("--output", default="uma_fold_structure.png",
                        help="Output PNG path (ignored when --interactive is set)")
    parser.add_argument("--interactive", action="store_true",
                        help="Launch interactive PyMOL GUI instead of saving PNG")
    return parser.parse_args()


# ── PDB builder ───────────────────────────────────────────────────────────────

_AA_3LETTER: dict[str, str] = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
    "X": "UNK",
}


def coords_to_pdb(sequence: str, coords: np.ndarray) -> str:
    """Build a minimal PDB string with Cα atoms only.

    Args:
        sequence: Amino-acid string of length N.
        coords:   Float array of shape [N, 3] (Å).

    Returns:
        Multi-line PDB-format string.
    """
    lines = []
    for i, (aa, (x, y, z)) in enumerate(zip(sequence.upper(), coords), start=1):
        resname = _AA_3LETTER.get(aa, "UNK")
        line = (
            f"ATOM  {i:5d}  CA  {resname} A{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
        lines.append(line)
    lines.append("END")
    return "\n".join(lines)


# ── PyMOL visualisation ───────────────────────────────────────────────────────

def visualize(
    sequence: str,
    coords: np.ndarray,
    output_png: str | None,
    interactive: bool,
) -> None:
    """Load coordinates into PyMOL and optionally save a PNG.

    Args:
        sequence:    Amino-acid sequence.
        coords:      Cα coordinates [N, 3].
        output_png:  Path for the output PNG (None in interactive mode).
        interactive: If True, launch the PyMOL GUI.
    """
    try:
        import pymol  # type: ignore
        from pymol import cmd  # type: ignore
    except ImportError as exc:
        sys.exit(
            "PyMOL is not installed.  Install via:\n"
            "  conda install -c conda-forge pymol-open-source\n"
            f"Original error: {exc}"
        )

    pdb_str = coords_to_pdb(sequence, coords)

    # Write to a temporary PDB file and load into PyMOL
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as tmp:
        tmp.write(pdb_str)
        tmp_path = tmp.name

    try:
        if interactive:
            pymol.finish_launching()
            cmd.load(tmp_path, "uma_fold")
            cmd.show("cartoon", "uma_fold")
            cmd.color("spectrum", "uma_fold")
            cmd.zoom("uma_fold")
            cmd.enable("all")
            # Hand control to PyMOL's own event loop; returns when the window closes.
            pymol.gui.startup()
        else:
            pymol.finish_launching(["pymol", "-cq"])  # quiet, no GUI
            cmd.load(tmp_path, "uma_fold")
            cmd.show("cartoon", "uma_fold")
            cmd.color("spectrum", "uma_fold")
            cmd.zoom("uma_fold")
            if output_png:
                Path(output_png).parent.mkdir(parents=True, exist_ok=True)
                cmd.png(output_png, width=1200, height=900, dpi=150, ray=1)
                print(f"Structure image saved to {output_png}")
    finally:
        os.unlink(tmp_path)


def main() -> None:
    args = parse_args()
    coords = np.load(args.coords)              # [N, 3]
    output_png = None if args.interactive else args.output
    visualize(args.sequence, coords, output_png, args.interactive)


if __name__ == "__main__":
    main()
