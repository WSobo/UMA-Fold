# UMA-Fold
**Ultra-lightweight, attention-free biomolecular structure predictor**

UMA-Fold replaces the standard AlphaFold3 / Boltz-1 Pairformer backbone with a
**Pairmixer** architecture that relies entirely on triangle multiplications
(via `torch.einsum`) and pair feed-forward networks — **zero attention layers**.

Designed to train on a single 24 GB VRAM GPU.

---

## Repository layout

```
UMA-Fold/
├── configs/                  # Hydra configuration files
│   ├── config.yaml           #   root config (W&B, paths, precision)
│   ├── model/pairmixer.yaml  #   model hyperparameters
│   ├── data/default.yaml     #   data pipeline settings
│   └── trainer/default.yaml  #   Lightning Trainer settings
├── src/
│   ├── models/
│   │   ├── pairmixer_block.py  #   attention-free pair update (core)
│   │   └── uma_fold.py         #   top-level model
│   ├── data/
│   │   └── datamodule.py       #   LightningDataModule
│   ├── training/
│   │   └── lightning_module.py #   LightningModule
│   └── utils/
│       └── precision.py        #   mixed-precision helpers
├── data/
│   ├── raw/                  # raw input data (gitignored)
│   └── processed/            # processed data (gitignored)
├── notebooks/
│   └── exploration.ipynb     # exploratory analysis
├── tests/
│   ├── test_pairmixer_block.py
│   └── test_datamodule.py
├── scripts/
│   ├── train.py              # Hydra-powered training entry point
│   ├── inference.py          # single-sequence inference
│   └── visualize_pymol.py    # PyMOL visualisation automation
├── requirements.txt
└── setup.py
```

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) install the package in editable mode
pip install -e .

# 4. Train (Hydra discovers configs/ automatically)
python scripts/train.py

# 5. Override any setting from the CLI
python scripts/train.py model.pair_dim=256 trainer.max_epochs=20

# 6. CPU smoke-test (no GPU required)
python scripts/train.py trainer.accelerator=cpu trainer.devices=1 wandb.enabled=false

# 7. Run tests
pytest tests/ -v
```

---

## Architecture

### Pairmixer block (no attention)

```
z [B, N, N, d_pair]
  │
  ├─► TriangleMultiplication (outgoing)
  │     einsum: "b i k d, b j k d -> b i j d"
  │
  ├─► TriangleMultiplication (incoming)
  │     einsum: "b k i d, b k j d -> b i j d"
  │
  └─► PairFFN (LayerNorm → Linear → SiLU → Linear)
```

### Low-norm dropout

Inside each triangle multiplication, features whose L2-norm falls below the
`(1 - keep_fraction)` quantile are zeroed out before the einsum.
This preferentially discards low-information features and reduces memory
bandwidth — a structured alternative to random dropout.

### Mixed precision

| Component | dtype |
|---|---|
| Trunk (triangle ops, FFN) | **bfloat16** |
| Softmax, loss, coordinate head | **float32** |

---

## Configuration (Hydra)

All hyperparameters live in `configs/`.  Override any value at the CLI:

```bash
python scripts/train.py \
    model.pair_dim=256 \
    model.num_blocks=12 \
    model.low_norm_dropout.keep_fraction=0.6 \
    trainer.max_epochs=200
```

---

## Experiment tracking (W&B)

Set your entity in `configs/config.yaml` or override at the CLI:

```bash
python scripts/train.py wandb.entity=my-team wandb.name=exp-001
```

Disable W&B entirely:

```bash
python scripts/train.py wandb.enabled=false
```

---

## References

The Pairmixer architecture is inspired by:

> Ouyang-Zhang, J., Murugan, P., Diaz, D. J., Scarpellini, G., Bowen, R. S., Gruver, N., Klivans, A., Krähenbühl, P., Faust, A., & Al-Shedivat, M. (2025). *Triangle Multiplication Is All You Need For Biomolecular Structure Representations*. arXiv:2510.18870 [q-bio.QM]. https://arxiv.org/abs/2510.18870
