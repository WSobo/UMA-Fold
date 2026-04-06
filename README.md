# UMA-Fold 🧬

**UMA-Fold** is a hyper-lightweight, single-GPU trainable biomolecular structure prediction model based on the architecture proposed in *"Triangle Multiplication is All You Need for Biomolecular Structure Representations"* (PairMixer).

## 🚀 The Strategy: Wrap, Don't Rewrite
To maintain a lean codebase, UMA-Fold completely outsources the heavy lifting of chemical validation, multi-sequence alignment (MSA) parsing, and coordinate handling. 

We directly wrap the data ecosystem from **Boltz-1** via PyTorch Lightning datamodules. This guarantees our inputs are 100% compatible with their pre-processed training datasets (PDB & MSAs), allowing us to focus entirely on the model architecture: **replacing the heavy `Pairformer` with our highly optimized `PairMixer`.**

## ⚡ What makes PairMixer different?
Standard structure predictors rely on $O(L^3)$ Triangle Attention, which causes massive memory bottlenecks on long sequences. The **PairMixer backbone** completely removes Triangle Attention and Sequence Updates, relying purely on highly optimized **Triangle Multiplication** and Feed-Forward Networks. 

This enables:
* **~34% reduction in training compute**
* **4x faster inference on long sequences**
* **Single-GPU trainability** from scratch for smaller scale variants.

## 📦 Setup & Installation
Please review the top of `requirements.txt` for step-by-step instructions on setting up your Conda environment, cloning dependencies, and installing the required packages.

## 🛠️ Data Downloading
To download the pre-processed RCSB training data and MSAs used in the Boltz-1 paper, simply run:
```bash
bash download_trainingdata.txt
```
