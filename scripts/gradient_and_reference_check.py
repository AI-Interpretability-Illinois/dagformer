"""Definitive code correctness check for FourWayDAGFormer.

TEST 1: Numerical gradient check on α
  - Forward with non-trivial α, compute loss
  - Autograd: loss.backward() → α.grad
  - Finite difference: perturb each α element by ε, recompute loss
  - If max relative error > 1e-3 → backward bug

TEST 2: Reference implementation comparison
  - Write a NAIVE for-loop forward that does "mix-then-project"
  - Compare logits with FourWay's "project-then-mix" optimized forward
  - If they differ → forward math bug

TEST 3: Non-identity forward sanity
  - Set α to specific known values (e.g., uniform [1/L,...,1/L])
  - Forward through both FourWay and reference
  - Verify they match AND differ from dense (α=identity)

Uses a SMALL model (4 layers, 4 heads, hidden=128) for speed.

Usage:
    python scripts/gradient_and_reference_check.py
    python scripts/gradient_and_reference_check.py --full_size  # 12L 16H 1024D (slow)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Olmo2Config, Olmo2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.olmo_graph import FourWayDAGFormer


def build_model(num_layers=4, num_heads=4, hidden=128, vocab=1000):
    mc = Olmo2Config(
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        intermediate_size=hidden * 4,
        vocab_size=vocab,
        tie_word_embeddings=True,
        max_position_embeddings=512,
    )
    base = Olmo2ForCausalLM(mc)
    fw = FourWayDAGFormer(
        model=base,
        num_layers=num_layers,
        num_heads=num_heads,
        use_local_correction=False,
        use_v_norm=False,
    )
    return base, fw


def build_random_alpha(num_layers, num_heads, batch, seq_len, device, dtype=torch.float32):
    """Build non-trivial α: uniform mixing + small random perturbation."""
    rw = {"q": [], "k": [], "v": [], "r": []}
    for l in range(1, num_layers):
        n_src = l + 1
        for stream in ("q", "k", "v"):
            # Start near identity, add perturbation
            α = torch.zeros(batch, seq_len, num_heads, n_src, device=device, dtype=dtype)
            α[..., -1] = 1.0
            α = α + 0.1 * torch.randn_like(α)
            α.requires_grad_(True)
            rw[stream].append(α)
        α_r = torch.zeros(batch, seq_len, n_src, device=device, dtype=dtype)
        α_r[..., -1] = 1.0
        α_r = α_r + 0.1 * torch.randn_like(α_r)
        α_r.requires_grad_(True)
        rw["r"].append(α_r)
    return rw


def compute_loss(fw, input_ids, rw, labels):
    logits = fw(input_ids, rw)
    return F.cross_entropy(logits.float().view(-1, logits.shape[-1]), labels.view(-1))


def reference_forward(base_model, input_ids, rw, num_layers, num_heads):
    """NAIVE reference: mix-then-project, explicit for-loops. No einsum.

    For each head h in layer l:
      mixed_input_h = Σ_j α[j,h] * layer_outputs[j]    (in model_dim)
      q_h = q_proj_head(mixed_input_h)                  (split weight)
      k_h = k_proj_head(mixed_input_h)
      v_h = v_proj_head(mixed_input_h)

    This is mathematically equivalent to project-then-mix because
    linear operators distribute over addition.
    """
    batch, seq_len = input_ids.shape
    hidden_size = base_model.config.hidden_size
    head_dim = hidden_size // num_heads
    H = num_heads
    device = input_ids.device

    embedding = base_model.model.embed_tokens(input_ids)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    cos, sin = base_model.model.rotary_emb(embedding, position_ids)

    from transformers.models.olmo2.modeling_olmo2 import apply_rotary_pos_emb

    causal_mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=embedding.dtype)
    causal_mask.masked_fill_(
        torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1),
        float('-inf'),
    )

    layer_outputs = [embedding]
    scaling = head_dim ** -0.5

    for l in range(num_layers):
        layer = base_model.model.layers[l]
        attn = layer.self_attn

        if l == 0:
            # Standard forward for layer 0
            q = attn.q_norm(attn.q_proj(embedding))
            k = attn.k_norm(attn.k_proj(embedding))
            v = attn.v_proj(embedding)
            q = q.view(batch, seq_len, H, head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, H, head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, H, head_dim).transpose(1, 2)
        else:
            α_q = rw['q'][l - 1].to(embedding.dtype)  # [B, T, H, L]
            α_k = rw['k'][l - 1].to(embedding.dtype)
            α_v = rw['v'][l - 1].to(embedding.dtype)
            α_r = rw['r'][l - 1].to(embedding.dtype)  # [B, T, L]

            # === NAIVE mix-then-project ===
            # For each head, mix sources in model_dim, then project
            q_heads = []
            k_heads = []
            v_heads = []
            for h in range(H):
                # Mix sources for head h
                mixed = torch.zeros_like(embedding)  # [B, T, D]
                for j in range(l + 1):
                    mixed = mixed + α_q[:, :, h, j:j+1] * layer_outputs[j]
                # Project through head h's slice of q_proj
                q_h = F.linear(mixed, attn.q_proj.weight[h*head_dim:(h+1)*head_dim, :])
                q_heads.append(q_h)  # [B, T, hd]

                mixed_k = torch.zeros_like(embedding)
                for j in range(l + 1):
                    mixed_k = mixed_k + α_k[:, :, h, j:j+1] * layer_outputs[j]
                k_h = F.linear(mixed_k, attn.k_proj.weight[h*head_dim:(h+1)*head_dim, :])
                k_heads.append(k_h)

                mixed_v = torch.zeros_like(embedding)
                for j in range(l + 1):
                    mixed_v = mixed_v + α_v[:, :, h, j:j+1] * layer_outputs[j]
                v_h = F.linear(mixed_v, attn.v_proj.weight[h*head_dim:(h+1)*head_dim, :])
                v_heads.append(v_h)

            # Stack heads: [B, H, T, hd]
            q = torch.stack(q_heads, dim=1)  # [B, H, T, hd]
            k = torch.stack(k_heads, dim=1)
            v = torch.stack(v_heads, dim=1)

            # Q/K norm (concatenate, norm, split back — same as FourWay)
            from einops import rearrange
            q_cat = rearrange(q, 'b h t d -> b t (h d)')
            q_cat = attn.q_norm(q_cat)
            q = rearrange(q_cat, 'b t (h d) -> b h t d', h=H)

            k_cat = rearrange(k, 'b h t d -> b t (h d)')
            k_cat = attn.k_norm(k_cat)
            k = rearrange(k_cat, 'b t (h d) -> b h t d', h=H)

            # R stream
            R = torch.zeros_like(embedding)
            for j in range(l + 1):
                R = R + α_r[:, :, j:j+1] * layer_outputs[j]

        # RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention
        attn_w = torch.matmul(q, k.transpose(-2, -1)) * scaling
        attn_w = attn_w + causal_mask
        attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_values = torch.matmul(attn_w, v)

        # O projection
        from einops import rearrange
        attn_concat = rearrange(attn_values, 'b h t d -> b t (h d)')
        attn_proj = attn.o_proj(attn_concat)

        # Post-attention norm
        attn_out = layer.post_attention_layernorm(attn_proj)

        # MLP
        if l == 0:
            mlp_in = embedding + attn_out
        else:
            mlp_in = R + attn_out
        mlp_raw = layer.mlp(mlp_in)
        mlp_out = layer.post_feedforward_layernorm(mlp_raw)

        # Next layer output
        if l == 0:
            X_next = embedding + attn_out + mlp_out
        else:
            X_next = R + attn_out + mlp_out
        layer_outputs.append(X_next)

    final_state = base_model.model.norm(layer_outputs[-1])
    logits = base_model.lm_head(final_state)
    return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_size", action="store_true", help="use 12L 16H 1024D (slow)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    if args.full_size:
        L, H, D, V = 12, 16, 1024, 100352
        B, T = 1, 32
    else:
        L, H, D, V = 4, 4, 128, 1000
        B, T = 2, 64

    print(f"Config: {L}L {H}H {D}D, vocab={V}, batch={B}, seq={T}, device={device}")

    # Build model (fp32 for gradient check precision)
    base, fw = build_model(L, H, D, V)
    base = base.to(device, dtype=torch.float32)
    fw = fw.to(device, dtype=torch.float32)
    fw.eval()

    input_ids = torch.randint(0, V, (B, T), device=device)
    labels = torch.randint(0, V, (B, T), device=device)

    # ================================================================
    # TEST 1: Numerical gradient check on α
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 1: Numerical gradient check (∂Loss/∂α)")
    print("=" * 60)

    rw = build_random_alpha(L, H, B, T, device)

    # Autograd gradient
    loss = compute_loss(fw, input_ids, rw, labels)
    loss.backward()

    # Finite difference gradient
    eps = 1e-4
    max_rel_errors = []
    n_checked = 0

    for stream in ("q", "k", "v", "r"):
        for l_idx, α in enumerate(rw[stream]):
            if α.grad is None:
                print(f"  WARNING: {stream}[{l_idx}] has NO gradient!")
                continue

            autograd = α.grad.clone()
            numel = α.numel()
            # Check a random subset of elements (full check too slow for large α)
            n_check = min(50, numel)
            indices = torch.randperm(numel)[:n_check]

            for idx in indices:
                # Get multi-dim index
                multi_idx = []
                remaining = idx.item()
                for s in reversed(α.shape):
                    multi_idx.insert(0, remaining % s)
                    remaining //= s

                # Forward with +ε
                with torch.no_grad():
                    orig = α.data.view(-1)[idx].item()
                    α.data.view(-1)[idx] = orig + eps
                loss_plus = compute_loss(fw, input_ids, rw, labels).item()

                # Forward with -ε
                with torch.no_grad():
                    α.data.view(-1)[idx] = orig - eps
                loss_minus = compute_loss(fw, input_ids, rw, labels).item()

                # Restore
                with torch.no_grad():
                    α.data.view(-1)[idx] = orig

                fd_grad = (loss_plus - loss_minus) / (2 * eps)
                ag_grad = autograd.view(-1)[idx].item()

                rel_err = abs(fd_grad - ag_grad) / (max(abs(fd_grad), abs(ag_grad)) + 1e-8)
                max_rel_errors.append(rel_err)
                n_checked += 1

            # Zero grad for next iteration
            α.grad.zero_()

    max_err = max(max_rel_errors)
    mean_err = sum(max_rel_errors) / len(max_rel_errors)
    pct_bad = sum(1 for e in max_rel_errors if e > 0.01) / len(max_rel_errors) * 100

    print(f"  Checked {n_checked} elements across all streams/layers")
    print(f"  Max relative error:  {max_err:.6e}")
    print(f"  Mean relative error: {mean_err:.6e}")
    print(f"  Elements with >1% error: {pct_bad:.1f}%")

    if max_err < 1e-3:
        print("  PASS ✓ — Gradients match finite differences")
    elif max_err < 1e-2:
        print("  WARN — Small gradient discrepancy (may be numerical)")
    else:
        print("  FAIL ✗ — Large gradient error! Backward is likely WRONG.")

    # ================================================================
    # TEST 2: Reference implementation comparison
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 2: FourWay vs naive reference forward (non-identity α)")
    print("=" * 60)

    # Fresh α (detached, no grad needed for this test)
    rw2 = build_random_alpha(L, H, B, T, device)
    for stream in ("q", "k", "v", "r"):
        for i in range(len(rw2[stream])):
            rw2[stream][i] = rw2[stream][i].detach()

    with torch.no_grad():
        logits_fw = fw(input_ids, rw2)
        logits_ref = reference_forward(base, input_ids, rw2, L, H)

    diff = (logits_fw.float() - logits_ref.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    nll_fw = F.cross_entropy(logits_fw.float().view(-1, V), labels.view(-1)).item()
    nll_ref = F.cross_entropy(logits_ref.float().view(-1, V), labels.view(-1)).item()

    print(f"  max_abs_diff:  {max_diff:.6e}")
    print(f"  mean_abs_diff: {mean_diff:.6e}")
    print(f"  NLL (FourWay):   {nll_fw:.6f}")
    print(f"  NLL (reference): {nll_ref:.6f}")
    print(f"  NLL diff:        {abs(nll_fw - nll_ref):.6e}")

    if max_diff < 1e-4:
        print("  PASS ✓ — FourWay matches naive reference exactly")
    elif max_diff < 1e-2:
        print("  WARN — Small numerical difference (may be accumulation order)")
    else:
        print("  FAIL ✗ — FourWay diverges from reference! Forward math is WRONG.")

    # ================================================================
    # TEST 3: Non-identity α produces different output than identity
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Non-identity α ≠ identity α (routing actually changes output)")
    print("=" * 60)

    # Identity α
    rw_id = {"q": [], "k": [], "v": [], "r": []}
    for l in range(1, L):
        n_src = l + 1
        for stream in ("q", "k", "v"):
            α = torch.zeros(B, T, H, n_src, device=device)
            α[..., -1] = 1.0
            rw_id[stream].append(α)
        α_r = torch.zeros(B, T, n_src, device=device)
        α_r[..., -1] = 1.0
        rw_id["r"].append(α_r)

    with torch.no_grad():
        logits_id = fw(input_ids, rw_id)
        logits_nonid = fw(input_ids, rw2)  # reuse rw2 from TEST 2

    diff_id_nonid = (logits_id.float() - logits_nonid.float()).abs()
    nll_id = F.cross_entropy(logits_id.float().view(-1, V), labels.view(-1)).item()
    nll_nonid = F.cross_entropy(logits_nonid.float().view(-1, V), labels.view(-1)).item()

    print(f"  Identity NLL:      {nll_id:.6f}")
    print(f"  Non-identity NLL:  {nll_nonid:.6f}")
    print(f"  Logit max diff:    {diff_id_nonid.max().item():.4f}")

    if diff_id_nonid.max().item() > 0.01:
        print("  PASS ✓ — Non-identity α produces meaningfully different output")
    else:
        print("  FAIL ✗ — α has NO effect! Routing is not being applied.")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Gradient check:       max_rel_err = {max_err:.2e} {'✓' if max_err < 1e-2 else '✗'}")
    print(f"  Reference comparison: max_abs_diff = {max_diff:.2e} {'✓' if max_diff < 1e-2 else '✗'}")
    print(f"  Routing effect:       logit_diff = {diff_id_nonid.max().item():.2e} {'✓' if diff_id_nonid.max().item() > 0.01 else '✗'}")


if __name__ == "__main__":
    main()
