# UMA-Fold: Order of Operations (A-Z)

This checklist walks you through the entire end-to-end pipeline, starting from a blank machine all the way to a fully trained model with inference support on your single-GPU setup.

### Phase 1: Environment & Dependency Setup
- [ ] **Create the Environment:** 
  ```bash
  conda create -n uma-fold python=3.11 -y
  conda activate uma-fold
  ```
- [ ] **Install Boltz (The Engine):** 
  ```bash
  git clone https://github.com/jwohlwend/boltz.git
  pip install -e ./boltz
  ```
- [ ] **Install PyTorch & Ecosystem:** 
  ```bash
  # Ensure your CUDA version matches (e.g. cu121 for CUDA 12.1)
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  pip install -r requirements.txt
  ```

### Phase 2: Data Acquisition
- [ ] **Download the Curated RCSB Dataset:**
  Run the data download script. It will securely fetch the pre-processed `.npz` targets, `.a3m` MSAs, and symmetry dictionary directly from the Boltz S3 buckets into your local `data/raw/` directory.
  ```bash
  bash download_trainingdata.txt
  ```

### Phase 3: Configuration Finalization
- [ ] **Finalize `configs/config.yaml` Data Loaders:** 
  Currently, `configs/config.yaml` has placeholders for `datasets:` and `featurizer:`. You will need to explicitly point these lists to your specific downloaded `data/raw/` folders matching the physical structure of your dataset once extracted. (You can reference `boltz/configs/data/default.yaml` for exact syntax shapes).
- [ ] **Weights & Biases Login:**
  ```bash
  wandb login
  ```
  Ensure your run metrics are appropriately streaming to your cloud dashboard.

### Phase 4: Sanity Testing (The Pilot Run)
- [ ] **Execute Pilot Run:** 
  Before scheduling a multi-day training session, check for OOM (Out Of Memory) limits and mismatched tensor loops by passing exactly 1 batch through the entire module pipeline.
  ```bash
  python scripts/pilot_run.py
  ```
- [ ] **Troubleshoot:** 
  If it fails, evaluate the stack trace. Typically, it will be a dimension mismatch (e.g., config `token_s` widths not explicitly matching the featurizer lengths). Fix in `config.yaml`.

### Phase 5: The Training Campaign
- [ ] **Branch Out:** 
  Leave `main` untouched as the stable architecture representation.
  ```bash
  git checkout -b training-campaign-1
  ```
- [ ] **Initiate the Full Run:** 
  Launch the primary training wrapper. Expect 12-24 hours for a full pass over the RCSB datasets.
  ```bash
  python scripts/train.py
  ```
- [ ] **Monitor:** Check W&B to ensure `val_loss` is decreasing and gradients aren't exploding. 

### Phase 6: Fast Inference & Downstream Tasks
- [ ] **Test Native Inference:**
  Once you have a mature `.ckpt` file in your `/checkpoints/` directory, simulate a complex cofolding procedure. Start by writing a quick target `.yaml` mimicking Boltz's examples (e.g., a multimer + ligand).
  ```bash
  python scripts/inference.py --yaml examples/target.yaml --ckpt checkpoints/last.ckpt
  ```
- [ ] **Output Connectivity:** Route the coordinates outputted from `inference.py` to a standard `.cif` or `.pdb` writer to load straight into PyMOL.
