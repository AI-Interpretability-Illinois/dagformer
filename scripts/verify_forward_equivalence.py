"""Verify FourWay forward == HF native forward at identity routing.

If these diverge, there's a bug in the manual forward that causes
the base model to learn different features than the standard path.
"""
import torch
import torch.nn as nn
from transformers import Olmo2Config, Olmo2ForCausalLM

import sys
sys.path.insert(0, '.')
from src.model.olmo_graph import FourWayDAGFormer


def replace_rmsnorm(model):
    count = 0
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
            count += 1
    return count


def make_identity_routing(batch, seq_len, num_layers, num_heads, device, dtype):
    """Create identity routing weights: [0,...,0,1] for each stream."""
    rw = {'q': [], 'k': [], 'v': [], 'r': []}
    for l in range(1, num_layers):
        n_src = l + 1
        for stream in ('q', 'k', 'v'):
            α = torch.zeros(batch, seq_len, num_heads, n_src, device=device, dtype=dtype)
            α[..., -1] = 1.0
            rw[stream].append(α)
        α_r = torch.zeros(batch, seq_len, n_src, device=device, dtype=dtype)
        α_r[..., -1] = 1.0
        rw['r'].append(α_r)
    return rw


def test_forward_equivalence():
    device = "cuda"
    dtype = torch.bfloat16
    B, T = 2, 128
    num_layers = 12
    num_heads = 16

    cfg = Olmo2Config(
        hidden_size=1024,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=4096,
        vocab_size=100352,
        tie_word_embeddings=True,
        max_position_embeddings=4096,
    )

    base = Olmo2ForCausalLM(cfg).to(device, dtype=dtype)
    n = replace_rmsnorm(base)
    print(f"Replaced {n} Olmo2RMSNorm modules")

    fourway = FourWayDAGFormer(
        model=base,
        num_layers=num_layers,
        num_heads=num_heads,
        use_local_correction=False,
    )
    fourway.eval()  # match eval mode

    input_ids = torch.randint(0, 1000, (B, T), device=device)
    rw = make_identity_routing(B, T, num_layers, num_heads, device, dtype)

    with torch.no_grad():
        # FourWay forward with identity routing
        logits_fourway = fourway(input_ids, rw)

        # HF native forward
        out_hf = base(input_ids=input_ids)
        logits_hf = out_hf.logits

    # Compare
    max_diff = (logits_fourway - logits_hf).abs().max().item()
    mean_diff = (logits_fourway - logits_hf).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        logits_fourway.flatten().unsqueeze(0).float(),
        logits_hf.flatten().unsqueeze(0).float(),
    ).item()

    print(f"\n=== Forward Equivalence Test ===")
    print(f"Max  absolute diff: {max_diff:.6e}")
    print(f"Mean absolute diff: {mean_diff:.6e}")
    print(f"Cosine similarity:  {cos_sim:.10f}")
    print(f"FourWay logits range: [{logits_fourway.min():.4f}, {logits_fourway.max():.4f}]")
    print(f"HF      logits range: [{logits_hf.min():.4f}, {logits_hf.max():.4f}]")

    if max_diff < 1e-3:
        print("\n✓ PASS: Forward paths are numerically equivalent")
    else:
        print(f"\n✗ FAIL: Forward paths DIVERGE (max_diff={max_diff:.4e})")

        # Debug: find which layer diverges
        print("\n--- Layer-by-layer debug ---")
        embedding = base.model.embed_tokens(input_ids)
        position_ids = torch.arange(T, device=device).unsqueeze(0)

        # Run HF layer by layer
        hf_hidden = embedding
        position_embeddings = base.model.rotary_emb(hf_hidden, position_ids)

        for l in range(num_layers):
            hf_hidden_next = base.model.layers[l](
                hf_hidden,
                position_embeddings=position_embeddings,
                attention_mask=None,
            )
            # FourWay stores layer outputs — get the corresponding one
            # Re-run FourWay forward and capture intermediates
            print(f"  Layer {l}: HF hidden norm = {hf_hidden_next.float().norm():.4f}")
            hf_hidden = hf_hidden_next

    return max_diff


def test_gradient_equivalence():
    """Test that gradients through FourWay == gradients through HF forward."""
    device = "cuda"
    dtype = torch.float32  # Use f32 for precise gradient comparison
    B, T = 1, 64
    num_layers = 4  # Small model for speed
    num_heads = 4
    head_dim = 32
    hidden = num_heads * head_dim

    cfg = Olmo2Config(
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        intermediate_size=hidden * 4,
        vocab_size=1000,
        tie_word_embeddings=True,
        max_position_embeddings=256,
    )

    base = Olmo2ForCausalLM(cfg).to(device, dtype=dtype)
    replace_rmsnorm(base)

    fourway = FourWayDAGFormer(
        model=base,
        num_layers=num_layers,
        num_heads=num_heads,
        use_local_correction=False,
    )
    fourway.train()

    input_ids = torch.randint(0, 100, (B, T), device=device)
    labels = torch.randint(0, 100, (B, T), device=device)
    rw = make_identity_routing(B, T, num_layers, num_heads, device, dtype)

    # FourWay backward
    logits_fw = fourway(input_ids, rw)
    loss_fw = torch.nn.functional.cross_entropy(
        logits_fw.view(-1, 1000), labels.view(-1)
    )
    loss_fw.backward()

    # Collect FourWay gradients
    fw_grads = {}
    for name, p in base.named_parameters():
        if p.grad is not None:
            fw_grads[name] = p.grad.clone()

    base.zero_grad()

    # HF backward
    out_hf = base(input_ids=input_ids)
    loss_hf = torch.nn.functional.cross_entropy(
        out_hf.logits.view(-1, 1000), labels.view(-1)
    )
    loss_hf.backward()

    # Compare gradients
    print(f"\n=== Gradient Equivalence Test (f32, {num_layers} layers) ===")
    print(f"FourWay loss: {loss_fw.item():.6f}")
    print(f"HF loss:      {loss_hf.item():.6f}")
    print(f"Loss diff:    {abs(loss_fw.item() - loss_hf.item()):.6e}")

    max_grad_diff = 0
    worst_param = ""
    for name, p in base.named_parameters():
        if p.grad is not None and name in fw_grads:
            diff = (fw_grads[name] - p.grad).abs().max().item()
            if diff > max_grad_diff:
                max_grad_diff = diff
                worst_param = name
            if diff > 1e-5:
                rel_diff = diff / max(fw_grads[name].abs().max().item(), 1e-10)
                print(f"  GRAD DIFF {name}: abs={diff:.4e} rel={rel_diff:.4e}")

    print(f"\nMax gradient diff: {max_grad_diff:.6e} ({worst_param})")
    if max_grad_diff < 1e-5:
        print("✓ PASS: Gradients match")
    else:
        print("✗ FAIL: Gradients DIVERGE")

    return max_grad_diff


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: Forward equivalence (bf16)")
    print("=" * 60)
    test_forward_equivalence()

    print("\n" + "=" * 60)
    print("TEST 2: Gradient equivalence (f32)")
    print("=" * 60)
    test_gradient_equivalence()
