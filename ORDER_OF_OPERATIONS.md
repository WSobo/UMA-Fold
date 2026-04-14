# UMA-Fold: Order of Operations

End-to-end setup from a blank machine to a trained model.

---

## Phase 1: Environment Setup

- [ ] **Install uv** (once per user):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

- [ ] **Clone the repo and create the environment:**
  ```bash
  git clone <repo-url>
  cd UMA-Fold
  uv sync --extra dev
  ```
  This installs PyTorch (CUDA 12.4 wheels), Boltz 2.2.1 from PyPI, and all other dependencies into `.venv/`. No separate boltz clone needed.

- [ ] **Verify the environment:**
  ```bash
  make env-check   # confirms torch version, CUDA availability, bf16 support
  ```

---

## Phase 2: Data Acquisition

- [ ] **Download the RCSB dataset:**
  ```bash
  make download
  ```
  Fetches pre-processed `.npz` structure targets and `.a3m` MSAs from the Boltz-1 S3 bucket into `data/raw/`. Expect ~100–200 GB.

---

## Phase 3: Sanity Testing

Before scheduling a multi-day job, run a 1-batch pilot to catch OOM errors and shape mismatches:

- [ ] **Run pilot at all three crop sizes:**
  ```bash
  make pilot-all     # submits scripts/SLURM/02_pilot_run.sh (sbatch)
  # or interactively:
  make pilot-15
  make pilot-30
  make pilot-40
  ```

- [ ] **Run the matmul shape/NaN test:**
  ```bash
  python test_matmul_shapes.py
  ```
  Should print `SUCCESS` for shape and `No NaN/Inf values` for numerical stability.

- [ ] **Troubleshoot if needed:** Check `logs/SLURM_err/` for stack traces. Most common issues are VRAM OOM (reduce `max_tokens`/`max_atoms` in `configs/config.yaml`) or shape mismatches from config drift.

---

## Phase 4: Training

### Single GPU (24 GB — A5500 / RTX 4090)

- [ ] **Submit the curriculum:**
  ```bash
  make train
  ```
  Runs three stages sequentially via `scripts/SLURM/03_train_model.sh`:
  - Stage 1: 15 epochs, crop15 (cheapest)
  - Stage 2: epochs 15–40, crop30 (resume from Stage 1)
  - Stage 3: epochs 40–100, crop40 (resume from Stage 2)

### Multi-GPU (4–8× A5500, ~4× faster)

- [ ] **Submit multi-GPU curriculum:**
  ```bash
  make train-multi
  ```
  Uses `scripts/SLURM/03_train_multi_gpu.sh`. Edit `DEVICES=` and `--gres` in that file to switch between 4 and 8 GPUs.

### Resuming a crashed run

- [ ] **Resume from the latest checkpoint:**
  ```bash
  sbatch scripts/SLURM/03_resume_stage2.sh
  ```
  The script finds the latest `checkpoints/last*.ckpt` automatically.

### Monitoring

- [ ] **W&B:** Remove `export WANDB_MODE="offline"` and set `WANDB_API_KEY` in the SLURM script to stream metrics live.
- [ ] **Watch `train_loss`:** Should decrease steadily. Non-finite losses are now silently skipped (logged as a `[WARNING]` line in stdout) — if you see many warnings, check for data issues or too-large a learning rate.

---

## Phase 5: Inference

- [ ] **Verify a checkpoint exists:**
  ```bash
  ls checkpoints/
  ```

- [ ] **Run inference on an example target:**
  ```bash
  make inference         # uses boltz/examples/multimer.yaml + checkpoints/last.ckpt
  ```

- [ ] **Run on your own target:**
  ```bash
  make infer-yaml YAML=path/to/target.yaml
  ```

---

## Utilities

```bash
make test       # run pytest suite (CPU, no GPU needed)
make lint       # ruff check
make lint-fix   # ruff check --fix
make jobs       # squeue -u $USER
make env-check  # verify torch + CUDA inside .venv
```
