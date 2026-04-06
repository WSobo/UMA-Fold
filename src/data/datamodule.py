"""
src/data/datamodule.py
──────────────────────
PyTorch Lightning DataModule for UMA-Fold.

This is a *boilerplate* implementation.  Replace the CSV-based ProteinDataset
with your actual data source (e.g. mmCIF parsing via BioPython, lmdb, etc.).

Expected CSV format (one row per protein)
──────────────────────────────────────────
    sequence,coords_path
    ACDEFGHIKLMNPQRSTVWY...,/path/to/coords.npy
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl


# ── Amino-acid vocabulary ──────────────────────────────────────────────────────

_AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
_CHAR_TO_IDX: dict[str, int] = {aa: i for i, aa in enumerate(_AA_VOCAB)}
UNKNOWN_IDX = len(_AA_VOCAB)  # index 20


def sequence_to_tensor(seq: str) -> torch.Tensor:
    """Convert an amino-acid string to an integer token tensor.

    Unknown characters are mapped to UNKNOWN_IDX (20).

    Args:
        seq: Amino-acid sequence string (uppercase).

    Returns:
        1-D LongTensor of length len(seq).
    """
    return torch.tensor(
        [_CHAR_TO_IDX.get(aa, UNKNOWN_IDX) for aa in seq.upper()],
        dtype=torch.long,
    )


# ── Dataset ───────────────────────────────────────────────────────────────────

class ProteinDataset(Dataset):
    """Minimal protein structure dataset backed by a CSV index file.

    Each row in the CSV must provide:
      - ``sequence``:    amino-acid string
      - ``coords_path``: path to a numpy ``.npy`` file of shape [N, 3]

    Args:
        csv_path:    Path to the CSV index file.
        max_seq_len: Maximum sequence length.  Longer sequences are randomly
                     cropped during training.
        crop:        Whether to randomly crop sequences that exceed max_seq_len.
    """

    def __init__(self, csv_path: str | Path, max_seq_len: int = 512, crop: bool = True) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.crop = crop

        self.records: list[dict] = []
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self.records.append(row)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.records[idx]
        seq = rec["sequence"]
        coords = np.load(rec["coords_path"])          # [N, 3]

        n = len(seq)

        # Random crop for sequences longer than max_seq_len
        if n > self.max_seq_len and self.crop:
            start = torch.randint(0, n - self.max_seq_len + 1, (1,)).item()
            seq = seq[start : start + self.max_seq_len]
            coords = coords[start : start + self.max_seq_len]

        token_ids = sequence_to_tensor(seq)            # [N']
        coords_t = torch.from_numpy(coords).float()   # [N', 3]

        return {"token_ids": token_ids, "coords": coords_t}


def _pad_collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad variable-length sequences within a batch to the same length."""
    max_n = max(item["token_ids"].shape[0] for item in batch)

    token_ids_list = []
    coords_list = []
    masks_list = []

    for item in batch:
        n = item["token_ids"].shape[0]
        pad = max_n - n

        token_ids_list.append(
            torch.cat([item["token_ids"], torch.zeros(pad, dtype=torch.long)])
        )
        coords_list.append(
            torch.cat([item["coords"], torch.zeros(pad, 3)])
        )
        mask = torch.cat([torch.ones(n, dtype=torch.bool), torch.zeros(pad, dtype=torch.bool)])
        masks_list.append(mask)

    return {
        "token_ids": torch.stack(token_ids_list),   # [B, N]
        "coords": torch.stack(coords_list),         # [B, N, 3]
        "mask": torch.stack(masks_list),            # [B, N]
    }


# ── DataModule ────────────────────────────────────────────────────────────────

class ProteinDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapping the protein structure dataset.

    Args:
        data_dir:    Root data directory.
        train_split: Relative path to the training CSV.
        val_split:   Relative path to the validation CSV.
        test_split:  Relative path to the test CSV.
        batch_size:  Samples per GPU per step.
        num_workers: DataLoader worker processes.
        pin_memory:  Whether to pin CUDA memory.
        max_seq_len: Maximum sequence length (longer sequences are cropped).
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_split: str = "raw/train.csv",
        val_split: str = "raw/val.csv",
        test_split: str = "raw/test.csv",
        batch_size: int = 4,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(data_dir)
        self.train_csv = self.data_dir / train_split
        self.val_csv = self.data_dir / val_split
        self.test_csv = self.data_dir / test_split

        self._train_ds: Optional[ProteinDataset] = None
        self._val_ds: Optional[ProteinDataset] = None
        self._test_ds: Optional[ProteinDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self._train_ds = ProteinDataset(self.train_csv, self.hparams.max_seq_len, crop=True)
            self._val_ds = ProteinDataset(self.val_csv, self.hparams.max_seq_len, crop=False)
        if stage in ("test", None):
            self._test_ds = ProteinDataset(self.test_csv, self.hparams.max_seq_len, crop=False)

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=_pad_collate,
            drop_last=shuffle,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self._val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self._test_ds, shuffle=False)
