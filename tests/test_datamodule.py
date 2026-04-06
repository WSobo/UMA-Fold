"""
tests/test_datamodule.py
────────────────────────
Unit tests for the ProteinDataModule and supporting helpers.

These tests verify:
* sequence_to_tensor maps known AAs to correct indices.
* Unknown characters map to UNKNOWN_IDX (20).
* _pad_collate produces correctly padded batches.
* ProteinDataset __getitem__ returns the expected keys and shapes.
* ProteinDataModule setup / DataLoader integration (using a tmp CSV).
"""

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.datamodule import (
    UNKNOWN_IDX,
    ProteinDataset,
    ProteinDataModule,
    _pad_collate,
    sequence_to_tensor,
)


# ── sequence_to_tensor ────────────────────────────────────────────────────────

class TestSequenceToTensor:
    def test_known_residues(self) -> None:
        t = sequence_to_tensor("ACD")
        # A=0, C=1, D=2 in _AA_VOCAB="ACDEFGHIKLMNPQRSTVWY"
        assert t.tolist() == [0, 1, 2]

    def test_unknown_residue(self) -> None:
        t = sequence_to_tensor("X")
        assert t.item() == UNKNOWN_IDX

    def test_lowercase_is_normalised(self) -> None:
        t_lower = sequence_to_tensor("acd")
        t_upper = sequence_to_tensor("ACD")
        assert torch.equal(t_lower, t_upper)

    def test_output_dtype(self) -> None:
        t = sequence_to_tensor("ACDE")
        assert t.dtype == torch.long


# ── _pad_collate ──────────────────────────────────────────────────────────────

class TestPadCollate:
    def _make_item(self, n: int) -> dict:
        return {
            "token_ids": torch.zeros(n, dtype=torch.long),
            "coords": torch.zeros(n, 3),
        }

    def test_shapes_after_padding(self) -> None:
        batch = [self._make_item(3), self._make_item(5), self._make_item(7)]
        out = _pad_collate(batch)
        assert out["token_ids"].shape == (3, 7)
        assert out["coords"].shape == (3, 7, 3)
        assert out["mask"].shape == (3, 7)

    def test_mask_is_correct(self) -> None:
        batch = [self._make_item(2), self._make_item(4)]
        out = _pad_collate(batch)
        # First item: first 2 positions True, last 2 False
        assert out["mask"][0].tolist() == [True, True, False, False]
        # Second item: all four True
        assert out["mask"][1].tolist() == [True, True, True, True]


# ── ProteinDataset ────────────────────────────────────────────────────────────

def _write_csv_and_coords(tmp_dir: Path, n_samples: int = 3) -> Path:
    """Write a minimal CSV + numpy coord files for testing."""
    csv_path = tmp_dir / "test.csv"
    rows = []
    for i in range(n_samples):
        seq = "ACDEFGHIKL"[:5 + i]        # lengths 5, 6, 7
        coords_path = tmp_dir / f"coords_{i}.npy"
        np.save(coords_path, np.zeros((len(seq), 3), dtype=np.float32))
        rows.append({"sequence": seq, "coords_path": str(coords_path)})

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sequence", "coords_path"])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


class TestProteinDataset:
    def test_len(self, tmp_path: Path) -> None:
        csv_path = _write_csv_and_coords(tmp_path, n_samples=4)
        ds = ProteinDataset(csv_path, max_seq_len=512)
        assert len(ds) == 4

    def test_getitem_keys(self, tmp_path: Path) -> None:
        csv_path = _write_csv_and_coords(tmp_path, n_samples=2)
        ds = ProteinDataset(csv_path, max_seq_len=512)
        item = ds[0]
        assert "token_ids" in item
        assert "coords" in item

    def test_getitem_shapes_consistent(self, tmp_path: Path) -> None:
        csv_path = _write_csv_and_coords(tmp_path, n_samples=1)
        ds = ProteinDataset(csv_path, max_seq_len=512)
        item = ds[0]
        n = item["token_ids"].shape[0]
        assert item["coords"].shape == (n, 3)

    def test_crop_applied(self, tmp_path: Path) -> None:
        """Sequences longer than max_seq_len should be cropped."""
        seq = "ACDEF" * 20       # length 100
        coords_path = tmp_path / "long.npy"
        np.save(coords_path, np.zeros((len(seq), 3), dtype=np.float32))
        csv_path = tmp_path / "long.csv"
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["sequence", "coords_path"])
            w.writeheader()
            w.writerow({"sequence": seq, "coords_path": str(coords_path)})

        ds = ProteinDataset(csv_path, max_seq_len=32, crop=True)
        item = ds[0]
        assert item["token_ids"].shape[0] <= 32


# ── ProteinDataModule ─────────────────────────────────────────────────────────

class TestProteinDataModule:
    def test_train_loader_returns_batch(self, tmp_path: Path) -> None:
        csv_path = _write_csv_and_coords(tmp_path, n_samples=6)
        dm = ProteinDataModule(
            data_dir=str(tmp_path),
            train_split=csv_path.name,
            val_split=csv_path.name,
            test_split=csv_path.name,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            max_seq_len=512,
        )
        dm.setup(stage="fit")
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert "token_ids" in batch
        assert batch["token_ids"].shape[0] == 2   # batch_size=2
