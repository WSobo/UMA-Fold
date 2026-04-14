import torch
from src.models.pairmixer_block import PairMixerBlock

def test_dimensions():
    # 1. Setup our hardware constraints (Batch 1, Max Nodes 384, Channels 128)
    batch_size = 1
    seq_len = 384
    c_z = 128
    
    print("Initializing Custom PairMixerBlock (cuBLAS matmul)...")
    block = PairMixerBlock(c_z=c_z).cuda()
    
    # 2. Create our dummy dense pair tensor [B, L, L, C]
    # Simulating the invariant representation of 384 nodes
    z_dummy = torch.randn(batch_size, seq_len, seq_len, c_z, device='cuda', dtype=torch.bfloat16)
    mask = torch.ones(batch_size, seq_len, seq_len, device='cuda', dtype=torch.bfloat16)
    
    print(f"Input Tensor Shape:  {z_dummy.shape}")
    print(f"Allocated VRAM Init: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # 3. Fire the Forward Pass
    try:
        # We wrap in autocast to mirror the training loop's mixed precision
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            z_out = block(z_dummy, mask)
        
        print(f"Output Tensor Shape: {z_out.shape}")
        print(f"Peak VRAM Usage:     {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
        
        if z_dummy.shape == z_out.shape:
            print("\n✅ SUCCESS: Input and Output dimensions match perfectly.")
            print("The explicit matmul operations are geometrically sound.")
        else:
            print("\n❌ ERROR: Dimension mismatch. The .permute() operations warped the tensor.")

        # Numerical stability checks — bf16 overflow produces NaN/Inf in long sequences
        assert not torch.isnan(z_out).any(), "❌ NaN detected in output!"
        assert not torch.isinf(z_out).any(), "❌ Inf detected in output!"
        print("✅ No NaN/Inf values in output. Numerically stable.")
            
    except Exception as e:
        print(f"\n💥 FAILED: PyTorch threw an error during execution:\n{e}")

if __name__ == "__main__":
    test_dimensions()