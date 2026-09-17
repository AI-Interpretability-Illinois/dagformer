"""Pinpoint the Q/K norm divergence: is it the rearrange or the batched proj?"""
import torch
import torch.nn as nn
from transformers import Olmo2Config, Olmo2ForCausalLM
from einops import rearrange
import sys
sys.path.insert(0, '.')


def replace_rmsnorm(model):
    for name, module in list(model.named_modules()):
        if type(module).__name__ == "Olmo2RMSNorm":
            new_norm = nn.RMSNorm(module.weight.shape[0], eps=module.variance_epsilon).to(
                device=module.weight.device, dtype=module.weight.dtype
            )
            new_norm.weight = module.weight
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_norm)


def main():
    device = "cuda"
    dtype = torch.bfloat16
    B, T = 2, 128
    H, hd, D = 16, 64, 1024

    cfg = Olmo2Config(
        hidden_size=D, num_hidden_layers=2, num_attention_heads=H,
        intermediate_size=4096, vocab_size=100352, tie_word_embeddings=True,
        max_position_embeddings=4096,
    )
    base = Olmo2ForCausalLM(cfg).to(device, dtype=dtype)
    replace_rmsnorm(base)
    base.eval()

    attn = base.model.layers[1].self_attn

    # Create a test hidden state (simulating a layer output)
    X = torch.randn(B, T, D, device=device, dtype=dtype)

    with torch.no_grad():
        # ====== Path A: Direct (as in HF / manual test) ======
        q_direct = attn.q_proj(X)               # [B, T, H*hd=1024]
        q_direct_normed = attn.q_norm(q_direct)  # [B, T, 1024]
        q_direct_heads = q_direct_normed.view(B, T, H, hd).transpose(1, 2)  # [B, H, T, hd]

        # ====== Path B: Einsum path (as in FourWay) ======
        # Simulate: project, view to head dim, "select" via identity einsum,
        # then rearrange back to [B,T,H*hd] for norm
        q_proj_out = attn.q_proj(X)              # [B, T, H*hd=1024]
        q_per_src = q_proj_out.view(B, T, H, hd)  # [B, T, H, hd]

        # Simulate identity einsum: result = q_per_src transposed to [B,H,T,hd]
        q_einsum = q_per_src.permute(0, 2, 1, 3)  # [B, H, T, hd] (no actual einsum, just transpose)

        # Now rearrange back for norm (this is what FourWay does)
        q_concat = rearrange(q_einsum, 'b h t d -> b t (h d)')  # [B, T, H*hd]
        q_concat_normed = attn.q_norm(q_concat)
        q_path_b = rearrange(q_concat_normed, 'b t (h d) -> b h t d', h=H)  # [B, H, T, hd]

        # ====== Compare ======
        print("=== Step-by-step comparison ===")

        # Are the pre-norm tensors the same?
        diff_prenorm = (q_direct.float() - q_concat.float()).abs().max().item()
        print(f"Pre-norm (q_proj output flattened): max_diff = {diff_prenorm:.6e}")

        # Check element ordering
        print(f"\nDirect q_proj output [0,0,:10]:  {q_direct[0,0,:10].tolist()}")
        print(f"Einsum rearranged    [0,0,:10]:  {q_concat[0,0,:10].tolist()}")

        # Are they bitwise equal?
        bitwise_eq = (q_direct == q_concat).all().item()
        print(f"\nBitwise equal pre-norm: {bitwise_eq}")

        if not bitwise_eq:
            # Find first mismatch
            mismatch = (q_direct != q_concat)
            first_idx = mismatch.nonzero()[0]
            b, t, d = first_idx.tolist()
            print(f"First mismatch at [{b},{t},{d}]:")
            print(f"  Direct:   {q_direct[b,t,d].item()}")
            print(f"  Rearranged: {q_concat[b,t,d].item()}")

            # Check: is rearrange doing a copy?
            print(f"\nDirect is_contiguous: {q_direct.is_contiguous()}")
            print(f"q_concat is_contiguous: {q_concat.is_contiguous()}")
            print(f"q_einsum is_contiguous: {q_einsum.is_contiguous()}")

            # Check strides
            print(f"\nDirect strides: {q_direct.stride()}")
            print(f"q_concat strides: {q_concat.stride()}")

        # Post-norm comparison
        diff_postnorm = (q_direct_heads.float() - q_path_b.float()).abs().max().item()
        print(f"\nPost-norm max_diff: {diff_postnorm:.6e}")

        # ====== Path C: Batched projection (my optimization) ======
        # Simulate stacking 3 sources and projecting all at once
        X2 = torch.randn(B, T, D, device=device, dtype=dtype)
        X3 = torch.randn(B, T, D, device=device, dtype=dtype)

        stacked = torch.stack([X2, X3, X], dim=0)  # [3, B, T, D]
        flat = stacked.reshape(3 * B * T, D)
        q_batched_all = attn.q_proj(flat).view(3, B, T, H, hd)
        q_batched_last = q_batched_all[-1]  # [B, T, H, hd]

        # Compare with direct projection of X
        q_individual = attn.q_proj(X).view(B, T, H, hd)
        diff_batched = (q_batched_last.float() - q_individual.float()).abs().max().item()
        print(f"\nBatched proj vs individual proj: max_diff = {diff_batched:.6e}")


if __name__ == "__main__":
    main()
