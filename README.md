# UMA-Fold 🧬 (Unofficial PairMixer & PEARL-inspired Architecture)

**UMA-Fold** is a personal, educational project aimed at independently implementing the architecture described in the preprint [**"Triangle Multiplication is All You Need for Biomolecular Structure Representations" (PairMixer)**] by researchers at Genesis Molecular AI and UT Austin, while incorporating key methodology from the [**PEARL**] technical report by the Genesis Research Team.

⚠️ **Disclaimer:** I am not affiliated with Genesis Molecular AI or UT Austin, nor am I the inventor of these architectures. This repository is purely a personal learning exercise to understand and reproduce their impressive hyper-lightweight biomolecular structure prediction backbone and diffusion pipeline optimizations. Full credit for the PairMixer architecture, the PEARL foundation model, theoretical breakthroughs, and structural insights goes to Genesis Molecular AI and the original authors. 

## 🚀 Project Goal: Wrap, Don't Rewrite
The primary goal of this project is to build a single-GPU trainable model by faithfully implementing the PairMixer backbone. To achieve this while keeping the codebase lean, UMA-Fold completely outsources the heavy lifting of chemical validation, multi-sequence alignment (MSA) parsing, and coordinate handling.

Following the methodology described in the paper, this implementation is built on top of **Boltz-1**. It directly wraps the data ecosystem from the open-source Boltz repository via PyTorch Lightning datamodules. This guarantees compatibility with pre-processed training datasets (PDB & MSAs), allowing me to focus entirely on the core experiment: replacing the heavy `Pairformer` with the highly optimized `PairMixer`.

## ⚡ The PairMixer Concept
As outlined by the Genesis team, standard structure predictors rely on $O(L^3)$ Triangle Attention, which causes massive memory bottlenecks on long sequences. The **PairMixer backbone** completely removes Triangle Attention and Sequence Updates, relying purely on highly optimized **Triangle Multiplication** and Feed-Forward Networks. 

By reproducing this, this project aims to leverage:
* **Single-GPU trainability** from scratch for smaller scale variants (specifically targeting 24GB consumer GPUs like the RTX 4090/A5500).
* **Faster inference and reduced training compute**, exploring the paper's findings of significant compute reduction and faster inference speeds.

## � PEARL Insights Integrated
To further optimize this project for local hardware, UMA-Fold explicitly borrows from Genesis Molecular AI's **PEARL** foundation model technical report:
* **SO(3)-Equivariant Diffusion Head:** Routes the sequence-free PairMixer representations directly into an SO(3) equivariant transformer to inherently respect 3D rotational symmetries without bloating parameters.
* **Curriculum Training:** Utilizes progressive spatial cropping via PyTorch Lightning to efficiently train on local physics and geometry extremely cheaply before scaling to full multi-chain resolutions.
* **Conservative Mixed-Precision:** Heavy triangle metrics execute natively in VRAM-saving `bf16`, but unstable losses and projection matrices are safeguarded in precise `fp32`.

## �📦 Setup & Installation
Please review the top of `requirements.txt` for step-by-step instructions on setting up your Conda environment, cloning dependencies, and installing the required packages.

## 🛠️ Data Downloading
To download the pre-processed RCSB training data and MSAs used in the Boltz-1 paper, simply run:
```bash
bash download_trainingdata.txt
```
