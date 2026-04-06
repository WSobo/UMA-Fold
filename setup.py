"""Minimal setup for editable installs (pip install -e .)."""

from setuptools import find_packages, setup

setup(
    name="uma_fold",
    version="0.1.0",
    description="UMA-Fold: Ultra-lightweight, attention-free biomolecular structure predictor",
    author="WSobo",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.2.0",
        "pytorch-lightning>=2.2.0",
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "wandb>=0.16.0",
        "einops>=0.7.0",
        "biopython>=1.83",
        "numpy>=1.26.0",
    ],
)
