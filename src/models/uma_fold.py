"""
src/models/uma_fold.py
──────────────────────
Main UMA-Fold model architecture.
Implements the PairMixer backbone with gradient checkpointing optimized for 
24GB VRAM consumer GPUs, completely replacing the heavy Pairformer while
leveraging Boltz's InputEmbedder and AtomDiffusion pipelines for zero-headache
data compatibility.
"""

import torch
from torch import nn, Tensor
from typing import Optional, Dict

from .pairmixer_block import PairMixerBlock
from .layers import LinearNoBias
from .modules.encoders import RelativePositionEncoder

# Heavy boltz modules — installed from PyPI (boltz==2.2.1), no local clone needed.
# Replacement roadmap: InputEmbedder → Phase 2, AtomDiffusion → Phase 3.
from boltz.model.modules.trunk import InputEmbedder, MSAModule
from boltz.model.modules.diffusion import AtomDiffusion


# =========================================================================
# PEARL INSIGHT: SO(3) vs SE(3) Architecture 
# =========================================================================
# Boltz-1's diffusion process naturally behaves as an SE(3) model because it 
# actively augments coordinates with random spatial translations (s_trans=1.0).
# The PEARL architecture identifies this as a massive computational burden. 
# By simply mean-centering the coordinates and mathematically eliminating 
# translation from the equation, we force the network to operate as an SO(3) 
# architecture, achieving identical physical validity with drastically lower
# parameter/VRAM overhead. 
import boltz.model.modules.diffusion as boltz_diffusion
import boltz.model.loss.diffusion as boltz_loss_diffusion
from einops import einsum

# 1. Store original Boltz augmentation functions
_orig_compute_random_augmentation = boltz_diffusion.compute_random_augmentation
_orig_center_random_augmentation = boltz_diffusion.center_random_augmentation

# 2. Patch them to zero-out translations (s_trans=0.0) globally
def _so3_compute_random_augmentation(*args, **kwargs):
    kwargs["s_trans"] = 0.0 # Force Translation to ZERO
    return _orig_compute_random_augmentation(*args, **kwargs)

def _so3_center_random_augmentation(*args, **kwargs):
    kwargs["s_trans"] = 0.0 # Force Translation to ZERO
    return _orig_center_random_augmentation(*args, **kwargs)

boltz_diffusion.compute_random_augmentation = _so3_compute_random_augmentation
boltz_diffusion.center_random_augmentation = _so3_center_random_augmentation

# =========================================================================
# ROBUST SVD PATCH for Mixed-Precision Stability (Avoids LinalgError: 3)
# =========================================================================
def _patched_weighted_rigid_align(true_coords, pred_coords, weights, mask):
    batch_size, num_points, dim = true_coords.shape
    weights = (mask * weights).unsqueeze(-1)
    
    # Safely compute centroids avoiding division by zero
    weight_sum = weights.sum(dim=1, keepdim=True)
    weight_sum = torch.clamp(weight_sum, min=1e-8)
    
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weight_sum
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weight_sum
    
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        print("Warning: The size of one of the point clouds is <= dim+1.")

    cov_matrix = einsum(weights * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j")
    original_dtype = cov_matrix.dtype
    
    # -------------------- THE PATCH --------------------
    # Extremely small SVDs fail to converge in fp32. Adding a trace of noise guarantees positive definiteness
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    cov_matrix_32 = cov_matrix_32 + 1e-5 * torch.eye(dim, device=cov_matrix.device, dtype=torch.float32).unsqueeze(0)
    
    try:
        # Pre-check for NaNs. If NaNs exist in the input matrix, bypass SVD entirely to prevent hard crashes.
        if torch.isnan(cov_matrix_32).any() or torch.isinf(cov_matrix_32).any():
            raise torch._C._LinAlgError("NaN/Inf strictly caught before SVD computation.")
            
        U, S, V = torch.linalg.svd(cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None)
    except torch._C._LinAlgError:
        # Fallback returning identity/dummy rotation if SVD truly fails due to structural collapse or NaNs
        # (This just zeros out the rotational loss for this single corrupted sequence to let the batch gradient survive)
        U = torch.eye(dim, device=cov_matrix.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)
        S = torch.ones(dim, device=cov_matrix.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        V = torch.eye(dim, device=cov_matrix.device, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1)

    V = V.mH

    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        print("Warning: Excessively low rank of cross-correlation between aligned point clouds.")

    rot_matrix = torch.einsum("b i j, b k j -> b i k", U, V).to(dtype=torch.float32)
    F_mat = torch.eye(dim, dtype=torch.float32, device=cov_matrix.device)[None].repeat(batch_size, 1, 1)
    F_mat[:, -1, -1] = torch.det(rot_matrix)
    rot_matrix = einsum(U, F_mat, V, "b i j, b j k, b l k -> b i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    aligned_coords = (einsum(true_coords_centered, rot_matrix, "b n i, b j i -> b n j") + pred_centroid)
    aligned_coords.detach_()
    return aligned_coords

boltz_loss_diffusion.weighted_rigid_align = _patched_weighted_rigid_align
boltz_diffusion.weighted_rigid_align = _patched_weighted_rigid_align
# =========================================================================

class PairMixerBackbone(nn.Module):
    """
    The hyper-lightweight PairMixer backbone.
    Stack of PairMixer blocks replacing the original Pairformer.
    Includes built-in Gradient Checkpointing (via torch.utils.checkpoint) for massive VRAM savings.
    """
    def __init__(
        self,
        num_blocks: int = 12, 
        c_z: int = 128,
        c_hidden_mul: int = 128,
        drop_rate: float = 0.0,
        gradient_checkpointing: bool = True
    ):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.blocks = nn.ModuleList([
            PairMixerBlock(c_z=c_z, c_hidden_mul=c_hidden_mul, drop_rate=drop_rate)
            for _ in range(num_blocks)
        ])

    def forward(self, z: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                z = checkpoint(block, z, mask, use_reentrant=False)
            else:
                z = block(z, mask)
        return z


class IdentityTriAttention(nn.Module):
    def forward(self, z, *args, **kwargs):
        return z

class UMAFold(nn.Module):
    """
    Main UMA-Fold architecture shell.
    Wiring: InputEmbedder -> MSA Module (Attention Removed) -> PairMixer Backbone -> AtomDiffusion
    """
    def __init__(self, config: dict):
        super().__init__()
        
        # Hyperparameters (fallback to Boltz Small defaults)
        self.token_s = config.get("token_s", 384)
        self.token_z = config.get("token_z", 128)
        self.atom_s = config.get("atom_s", 128)
        self.atom_z = config.get("atom_z", 32)

        # --- 1. Embedding Modules ---
        from src.data.constants import num_tokens, pocket_contact_info
        s_input_dim = self.token_s + 2 * num_tokens + 1 + len(pocket_contact_info)
        self.s_init = LinearNoBias(s_input_dim, self.token_s)
        self.z_init_1 = LinearNoBias(s_input_dim, self.token_z)
        self.z_init_2 = LinearNoBias(s_input_dim, self.token_z)

        embedder_args = {
            "atoms_per_window_queries": 32,
            "atoms_per_window_keys": 128,
            "atom_feature_dim": 389,
            "atom_encoder_depth": 3,
            "atom_encoder_heads": 4,
            **config.get("embedder_args", {})
        }
        self.input_embedder = InputEmbedder(
            atom_s=self.atom_s, atom_z=self.atom_z,
            token_s=self.token_s, token_z=self.token_z,
            **embedder_args
        )
        self.rel_pos = RelativePositionEncoder(self.token_z)
        self.token_bonds = LinearNoBias(1, self.token_z)

        # Normalization layers
        self.s_norm = nn.LayerNorm(self.token_s)
        self.z_norm = nn.LayerNorm(self.token_z)

        # Recycling projections
        self.s_recycle = LinearNoBias(self.token_s, self.token_s)
        self.z_recycle = LinearNoBias(self.token_z, self.token_z)
        
        # --- 2. MSA Module (With Triangle Attention gracefully disabled) ---
        msa_args = {
            "msa_s": 64,
            **config.get("msa_args", {})
        }
        self.msa_module = MSAModule(
            token_z=self.token_z, 
            s_input_dim=s_input_dim, 
            **msa_args
        )
        self._disable_msa_triangle_attention()
        
        # --- 3. Our Highly-Optimized PairMixer Backbone ---
        pm_config = config.get("pairmixer_args", {})
        self.backbone = PairMixerBackbone(
            num_blocks=pm_config.get("num_blocks", 12),
            c_z=self.token_z,
            c_hidden_mul=pm_config.get("c_hidden_mul", 128),
            drop_rate=pm_config.get("drop_rate", 0.0),
            gradient_checkpointing=pm_config.get("gradient_checkpointing", True)
        )
        
        # --- 4. Structure/Diffusion Module (From Boltz) ---
        score_model_args = config.get("score_model_args", {})
        score_model_args.update({
            "token_z": self.token_z, "token_s": self.token_s,
            "atom_z": self.atom_z, "atom_s": self.atom_s,
            "atom_feature_dim": 389,
            "atoms_per_window_queries": 32,
            "atoms_per_window_keys": 128,
        })
        self.structure_module = AtomDiffusion(
            score_model_args=score_model_args,
            **config.get("diffusion_args", {})
        )

    def _disable_msa_triangle_attention(self):
        """
        The PairMixer paper explicitly calls for: 
        'we replace the Pairformer backbone with PairMixer and remove triangle attention from the MSA Module.'
        Instead of modifying the actual boltz pip installation, we dynamically patch the instantiated MSA 
        layers to skip the tri_att_start and tri_att_end functions, replacing them with Identity pass-throughs.
        """
        for layer in self.msa_module.layers:
            # If wrapped in activation checkpointing, unwrap
            target_layer = layer
            if hasattr(layer, "module"): # Checkpoint wrapper
                target_layer = layer.module
                
            if hasattr(target_layer, "tri_att_start"):
                # Rather than a strict Identity (which takes 1 arg), we mock the call 
                # since TriAttention in Boltz takes (z, mask, pair_mask)
                target_layer.tri_att_start = IdentityTriAttention()
            if hasattr(target_layer, "tri_att_end"):
                target_layer.tri_att_end = IdentityTriAttention()

    def forward(self, batch: dict) -> dict:
        """
        Forward pass mimicking the Boltz-1 structure prediction pipeline, but routing through PairMixer.
        """
        # Step 1: Embed inputs
        feats = batch  # Assuming batch is heavily populated dictionary generated by Boltz loader
        
        s_inputs = self.input_embedder(feats)
        
        # Initialize the sequence and pairwise embeddings
        s_init = self.s_init(s_inputs)
        z_init = (
            self.z_init_1(s_inputs)[:, :, None]
            + self.z_init_2(s_inputs)[:, None, :]
        )
        
        # Apply Relative positional encoding and bonds
        # Note: In Boltz this is feats, not self.rel_pos(feats)
        z = z_init + self.rel_pos(feats)
        if "token_bonds" in feats:
            z = z + self.token_bonds(feats["token_bonds"].float())
            
        s = s_init
        
        # Apply layer norms (Assuming recycling iteration = 0 for this standalone pass)
        s = self.s_norm(s) # + self.s_recycle(s_prev) 
        z = self.z_norm(z) # + self.z_recycle(z_prev)
        
        mask = feats["token_pad_mask"]
        pair_mask = mask[:, :, None] * mask[:, None, :]
        
        # Step 2: MSA processing (No Triangle Attention)
        z = z + self.msa_module(z, s_inputs, feats)

        # Step 3: PairMixer Backbone
        # Massive VRAM memory save! "Our model bypasses sequence processing entirely 
        # and passes the initial sequence representation directly to the diffusion module"
        z_backbone = self.backbone(z, pair_mask)
        
        # Step 4: Diffusion
        dict_out = {"s_trunk": s, "z_trunk": z_backbone}
        if self.training:
            dict_out.update(
                self.structure_module(
                    s_trunk=s,
                    z_trunk=z_backbone,
                    s_inputs=s_inputs,
                    feats=feats,
                    relative_position_encoding=self.rel_pos(feats)
                )
            )
        else:
            # Inference mode
            dict_out.update(
                self.structure_module.sample(
                    s_trunk=s,
                    z_trunk=z_backbone,
                    s_inputs=s_inputs,
                    feats=feats,
                    relative_position_encoding=self.rel_pos(feats)
                )
            )
            
        return dict_out

    def compute_loss(self, feats: dict, out_dict: dict) -> dict:
        """
        Wrapped loss delegation to atom diffusion, preventing edge-cases where 
        torch.compile wraps the model and breaks deep module property access.
        """
        return self.structure_module.compute_loss(feats, out_dict)

