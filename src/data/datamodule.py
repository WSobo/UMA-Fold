"""
src/data/datamodule.py
──────────────────────
DataModule wrapper for UMA-Fold using the Boltz-1 data pipeline.

Instead of writing custom parsing logic, we directly leverage the existing
Boltz data infrastructure. This ensures 100% compatibility with the downloaded
processed datasets (RCSB targets and MSAs) and handles complex featurization
(cropping, symmetries, etc.) out-of-the-box.
"""
from typing import Optional, Dict
import pytorch_lightning as pl

# Import directly from the boltz repository
from boltz.data.module.training import BoltzTrainingDataModule, DataConfig

def create_uma_fold_datamodule(config) -> pl.LightningDataModule:
    """
    Creates and returns the BoltzTrainingDataModule using a configuration
    dictionary that matches Boltz's DataConfig expectations.
    
    This allows us to seamlessly pass the RCSB targets and MSAs straight 
    into our custom Pairmixer backbone.
    """
    # Transform instantiated hydra dictionary mapping into the DataConfig dataclass
    if hasattr(config, "_items"): # If it's still a DictConfig somehow
        # convert it into a naive python dict to unpack safely into the DataConfig
        config = dict(config)
    elif not isinstance(config, dict):
        config = dict(config)
        
    cfg = DataConfig(**config)
    return BoltzTrainingDataModule(cfg)
