import sys, yaml, warnings, torch
warnings.filterwarnings('ignore')
from omegaconf import OmegaConf
config = OmegaConf.load('configs/data/default.yaml')
sys.path.append('src')
from data.datamodule import create_uma_fold_datamodule
dm = create_uma_fold_datamodule(config)
dm.setup('fit')
dl = dm.train_dataloader()
batch = next(iter(dl))

import boltz.data.const as const
from boltz.model.modules.trunk import InputEmbedder
from boltz.model.modules.encoders import RelativePositionEncoder

embedder = InputEmbedder(
    atom_s=128, atom_z=32, token_s=384, token_z=128,
    atoms_per_window_queries=32, atoms_per_window_keys=128,
    atom_feature_dim=128, atom_encoder_depth=3, atom_encoder_heads=4
)
print("Input embedder created")
try:
    s_inputs = embedder(batch)
    print("s_inputs shape:", s_inputs.shape)
except Exception as e:
    print("embedder err:", e)

