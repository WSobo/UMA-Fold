# UMA-Fold

A personal reimplementation of the **PairMixer** backbone for biomolecular structure prediction, trained on the Boltz-1 dataset pipeline. Built as a portfolio piece exploring whether triangle attention can be fully eliminated from the Pairformer without catastrophic quality loss.

> **Disclaimer:** Not affiliated with Genesis Molecular AI or UT Austin. Full credit for PairMixer, PEARL, and the underlying theory goes to the original authors. This is an independent reproduction and engineering exercise.

---

## What it is

Standard structure predictors (AlphaFold3, Boltz-1) use a **Pairformer** backbone containing triangle attention, triangle multiplication, and sequence updates. Triangle attention is the dominant memory bottleneck — $O(L^3)$ in sequence length.

UMA-Fold replaces the Pairformer with **PairMixer**: triangle multiplication only, no attention in the backbone at all. The hypothesis from ["Triangle Multiplication is All You Need"](https://arxiv.org/abs/2506.01085) is that the multiplicative update alone captures sufficient geometric signal, making the attention layers redundant for the pair track.

The rest of the pipeline — atom-level embedding, MSA processing, and the diffusion structure module — comes from Boltz-1 (installed from PyPI, no local clone needed).

### Architecture comparison

| Component | Boltz-1 | UMA-Fold |
|---|---|---|
| Backbone | Pairformer (tri-attn + tri-mult + seq update) | PairMixer (tri-mult only) |
| Diffusion geometry | SE(3) — random translations | SO(3) — translations zeroed out |
| MSA triangle attention | Yes | Removed |
| Backbone parameters | ~200M | ~50M |
| Min. training VRAM | ~80 GB | 24 GB (A5500 / RTX 4090) |
| Multi-GPU training | Yes | Yes (DDP, 4–8 GPUs) |

### Key engineering decisions

**Explicit `torch.matmul` for triangle multiplication** — replaces einsum to route through cuBLAS kernels directly, avoiding VRAM spikes from intermediate einsum buffers.

**float32 upcast inside the triangle matmul** — in bf16, summing over `L` terms overflows at crop30/40 sequence lengths, producing NaN. The matmul operands are upcast to fp32 and cast back, mirroring the SVD patch already applied to the rigid alignment in the diffusion head.

**SO(3) diffusion** — the PEARL technical report identifies translation augmentation as unnecessary overhead for the score network. Zeroing `s_trans` forces the model into SO(3) geometry at no quality cost.

**Curriculum training** — three stages: crop15 (local chemistry) → crop30 (intermediate context) → crop40 (full context), each resuming from the previous checkpoint. This keeps early training cheap while letting the model progressively learn longer-range contacts.

**NaN guard in training step** — returns `None` on non-finite loss so PyTorch Lightning skips the batch cleanly rather than silently propagating NaN gradients through the optimizer state.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/). Install it once per user:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the project root:

```bash
uv sync --extra dev
```

This creates `.venv/`, installs PyTorch with the correct CUDA 12.4 wheels, and pulls Boltz 2.2.1 from PyPI. No manual `git clone` of boltz needed.

Activate for interactive use:

```bash
source .venv/bin/activate
```

---

## Data

Download the pre-processed RCSB targets and MSAs from the Boltz-1 S3 bucket:

```bash
make download
```

Populates `data/raw/rcsb_processed_targets/` (`.npz` per PDB entry) and `data/raw/rcsb_processed_msa/`.

---

## Training

### Sanity check first

```bash
make pilot-15   # 1-batch test at crop15
make pilot-30   # 1-batch test at crop30
make pilot-40   # 1-batch test at crop40
```

### Single GPU (A5500 / RTX 4090, 24 GB)

```bash
make train
```

Runs three sequential stages via `scripts/SLURM/03_train_model.sh`:

| Stage | Epochs | max\_neighborhood | Notes |
|---|---|---|---|
| 1 | 0–15 | 15 | Local chemistry, cheapest batches |
| 2 | 15–40 | 30 | Intermediate context, resumes Stage 1 |
| 3 | 40–100 | 40 | Full context, resumes Stage 2 |

### Multi-GPU (4–8× A5500, ~4× faster)

```bash
make train-multi
```

Uses `scripts/SLURM/03_train_multi_gpu.sh` with 4× A5500 and DDP. Edit `DEVICES=` and `--gres=gpu:A5500:N` in that file to switch to 8 GPUs (also update `--cpus-per-task=32`). `accumulate_grad_batches=1` with 4 GPUs gives the same effective batch of 4 as single-GPU with `accumulate_grad_batches=4`.

### Resume from a checkpoint

```bash
sbatch scripts/SLURM/03_resume_stage2.sh
```

Or override directly:

```bash
python scripts/train.py \
    run_name="resume-stage2" \
    ++training.epochs=40 \
    ++training.devices=4 \
    ++training.accumulate_grad_batches=1 \
    ++data.datasets.0.cropper.max_neighborhood=30 \
    ++training.ckpt_path="checkpoints/last.ckpt"
```

### W&B

Remove `export WANDB_MODE="offline"` from any SLURM script and set `WANDB_API_KEY` to stream metrics live.

---

## Inference

```bash
make inference                        # runs on boltz/examples/multimer.yaml
make infer-yaml YAML=path/to/my.yaml  # custom target
```

---

## Project structure

```
UMA-Fold/
├── src/
│   ├── models/
│   │   ├── layers.py               # LinearNoBias, Transition (owned)
│   │   ├── modules/
│   │   │   └── encoders.py         # RelativePositionEncoder (owned)
│   │   ├── pairmixer_block.py      # PairMixerBlock: tri-mul × 2 + FFN
│   │   └── uma_fold.py             # Full model: Embed → MSA → PairMixer → Diffuse
│   ├── data/
│   │   ├── constants.py            # Token vocab, num_tokens (owned)
│   │   └── datamodule.py           # Thin wrapper around BoltzTrainingDataModule
│   └── training/
│       └── lightning_module.py     # Training/validation step + NaN guard
├── scripts/
│   ├── train.py                    # Hydra entry point, DDP auto-select
│   ├── pilot_run.py                # 1-batch sanity check
│   ├── inference.py                # Structure prediction
│   └── SLURM/
│       ├── 01_download_data.sh
│       ├── 02_pilot_run.sh
│       ├── 03_train_model.sh           # Single-GPU curriculum
│       ├── 03_train_multi_gpu.sh       # Multi-GPU DDP curriculum (4–8×)
│       ├── 03_resume_stage2.sh         # Resume from Stage 2 checkpoint
│       └── 04_inference.sh
├── configs/
│   └── config.yaml                 # Hydra config (model, data, training)
├── pyproject.toml                  # uv project — torch cu124, boltz PyPI
├── MAKEFILE                        # make pilot / train / train-multi / inference / test
└── test_matmul_shapes.py           # Shape + NaN sanity check for PairMixerBlock
```

---

## Boltz dependency status

UMA-Fold owns the simple building blocks and uses Boltz 2.2.1 (PyPI) for the heavy internals:

| Component | Owned | Boltz PyPI | Roadmap |
|---|---|---|---|
| `Transition` (SwiGLU MLP) | `src/models/layers.py` | — | Done |
| `LinearNoBias` | `src/models/layers.py` | — | Done |
| `RelativePositionEncoder` | `src/models/modules/encoders.py` | — | Done |
| Token vocabulary / constants | `src/data/constants.py` | — | Done |
| Diffusion loss + rigid align | patched in `uma_fold.py` | — | Done |
| `InputEmbedder` | — | `boltz.model.modules.trunk` | Phase 2 |
| `MSAModule` | — | `boltz.model.modules.trunk` | Phase 2 |
| `AtomDiffusion` | — | `boltz.model.modules.diffusion` | Phase 3 |
| Data featurizer / pipeline | — | `boltz.data.*` | Phase 4 |

---

## References

- **PairMixer:** "Triangle Multiplication is All You Need for Biomolecular Structure Representations" — Genesis Molecular AI & UT Austin
- **PEARL:** Genesis Research Team technical report — SO(3) diffusion and curriculum training
- **Boltz-1:** Wohlwend et al. — open-source structure predictor; data pipeline and diffusion head
