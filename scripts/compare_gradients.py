"""Compare ∂Loss/∂W for base model params between dense and FourWay forward.

At identity init, both should give IDENTICAL gradients.
At non-identity α, FourWay gradients go through einsum — do they diverge?

Tests:
1. Identity α: dense grad vs FourWay grad for W_q, W_k, W_v, W_o, MLP
2. Non-identity α: are FourWay gradients sane? (direction, magnitude)
3. bf16 vs fp32: does bf16 introduce gradient corruption specific to FourWay?
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Olmo2Config, Olmo2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model.olmo_graph import FourWayDAGFormer

device = torch.device("cuda")

def run_test(dtype_name, dtype):
    print(f"\n{'='*60}")
    print(f"DTYPE: {dtype_name}")
    print(f"{'='*60}")

    torch.manual_seed(42)
    L, H, D, V = 4, 4, 128, 1000
    B, T = 2, 64

    mc = Olmo2Config(hidden_size=D, num_hidden_layers=L, num_attention_heads=H,
                     num_key_value_heads=H, intermediate_size=D*4, vocab_size=V,
                     tie_word_embeddings=True, max_position_embeddings=256)

    # Build model
    base = Olmo2ForCausalLM(mc).to(device, dtype=dtype)
    fw = FourWayDAGFormer(model=base, num_layers=L, num_heads=H).to(device, dtype=dtype)

    ids = torch.randint(0, V, (B, T), device=device)
    labels = torch.randint(0, V, (B, T), device=device)

    # === TEST 1: Identity α — dense vs FourWay grad comparison ===
    print(f"\nTEST 1: Identity α — dense grad vs FourWay grad")

    # Dense forward + backward
    base.zero_grad()
    dense_logits = base(input_ids=ids).logits
    dense_loss = F.cross_entropy(dense_logits.float().view(-1, V), labels.view(-1))
    dense_loss.backward()
    dense_grads = {}
    for name, p in base.named_parameters():
        if p.grad is not None:
            dense_grads[name] = p.grad.clone()

    # FourWay forward + backward with identity α
    base.zero_grad()
    rw_id = {"q": [], "k": [], "v": [], "r": []}
    for l in range(1, L):
        n_src = l + 1
        for s in ("q", "k", "v"):
            a = torch.zeros(B, T, H, n_src, device=device, dtype=dtype)
            a[..., -1] = 1.0
            rw_id[s].append(a)
        ar = torch.zeros(B, T, n_src, device=device, dtype=dtype)
        ar[..., -1] = 1.0
        rw_id["r"].append(ar)

    fw_logits = fw(ids, rw_id)
    fw_loss = F.cross_entropy(fw_logits.float().view(-1, V), labels.view(-1))
    fw_loss.backward()
    fw_grads = {}
    for name, p in base.named_parameters():
        if p.grad is not None:
            fw_grads[name] = p.grad.clone()

    print(f"  Dense loss: {dense_loss.item():.6f}")
    print(f"  FW loss:    {fw_loss.item():.6f}")
    print(f"  Loss diff:  {abs(dense_loss.item() - fw_loss.item()):.2e}")

    # Compare gradients for key parameters
    print(f"\n  {'Parameter':<45} {'max_diff':>10} {'rel_diff':>10} {'cos_sim':>10}")
    print(f"  {'-'*80}")
    max_rel_overall = 0
    min_cos_overall = 1
    for name in sorted(dense_grads.keys()):
        if name not in fw_grads:
            continue
        dg = dense_grads[name].float()
        fg = fw_grads[name].float()
        max_diff = (dg - fg).abs().max().item()
        rel_diff = max_diff / (dg.abs().max().item() + 1e-10)
        cos = F.cosine_similarity(dg.flatten().unsqueeze(0), fg.flatten().unsqueeze(0)).item()
        max_rel_overall = max(max_rel_overall, rel_diff)
        min_cos_overall = min(min_cos_overall, cos)
        # Only print layers 0 and 1 to keep output manageable
        if "layers.0." in name or "layers.1." in name or "embed" in name or "norm" in name or "lm_head" in name:
            print(f"  {name:<45} {max_diff:>10.2e} {rel_diff:>10.4f} {cos:>10.6f}")

    print(f"\n  Overall: max_rel_diff={max_rel_overall:.4f}, min_cosine_sim={min_cos_overall:.6f}")
    if max_rel_overall < 0.01:
        print(f"  PASS ✓ — dense and FourWay gradients match at identity init")
    else:
        print(f"  FAIL ✗ — gradients DIFFER at identity init! rel_diff={max_rel_overall:.4f}")

    # === TEST 2: Non-identity α — gradient sanity check ===
    print(f"\nTEST 2: Non-identity α — gradient properties")

    base.zero_grad()
    rw_nonid = {"q": [], "k": [], "v": [], "r": []}
    for l in range(1, L):
        n_src = l + 1
        for s in ("q", "k", "v"):
            a = torch.zeros(B, T, H, n_src, device=device, dtype=dtype)
            a[..., -1] = 0.7  # non-trivial mixing
            a[..., 0] = 0.3
            rw_nonid[s].append(a)
        ar = torch.zeros(B, T, n_src, device=device, dtype=dtype)
        ar[..., -1] = 0.8
        ar[..., 0] = 0.2
        rw_nonid["r"].append(ar)

    fw_logits2 = fw(ids, rw_nonid)
    fw_loss2 = F.cross_entropy(fw_logits2.float().view(-1, V), labels.view(-1))
    fw_loss2.backward()

    # Compare non-identity grads to identity grads
    print(f"  Non-identity loss: {fw_loss2.item():.6f}")
    print(f"  Identity loss:     {fw_loss.item():.6f}")

    nonid_grads = {}
    for name, p in base.named_parameters():
        if p.grad is not None:
            nonid_grads[name] = p.grad.clone()

    # Check gradient magnitude ratio and cosine similarity
    print(f"\n  {'Parameter':<45} {'id_norm':>10} {'nonid_norm':>10} {'ratio':>8} {'cos_sim':>10}")
    print(f"  {'-'*90}")
    for name in sorted(fw_grads.keys()):
        if name not in nonid_grads:
            continue
        ig = fw_grads[name].float()
        ng = nonid_grads[name].float()
        in_norm = ig.norm().item()
        nn_norm = ng.norm().item()
        ratio = nn_norm / (in_norm + 1e-10)
        cos = F.cosine_similarity(ig.flatten().unsqueeze(0), ng.flatten().unsqueeze(0)).item()
        if "layers.0." in name or "layers.1." in name:
            print(f"  {name:<45} {in_norm:>10.4f} {nn_norm:>10.4f} {ratio:>8.2f} {cos:>10.4f}")

    # === TEST 3: Check if FourWay adds extra gradient to layer 0 ===
    print(f"\nTEST 3: Layer 0 gradient comparison")
    print(f"  (Layer 0 has no routing — should be identical between dense and FourWay)")

    l0_params = [(n, p) for n, p in base.named_parameters() if "layers.0." in n and p.grad is not None]
    l0_max_diff = 0
    for name, p in l0_params:
        dg = dense_grads[name].float()
        fg = fw_grads[name].float()
        diff = (dg - fg).abs().max().item()
        l0_max_diff = max(l0_max_diff, diff)

    print(f"  Layer 0 max grad diff (dense vs FW identity): {l0_max_diff:.2e}")
    if l0_max_diff < 1e-5:
        print(f"  PASS ✓ — Layer 0 grads identical")
    else:
        print(f"  INTERESTING — Layer 0 grads DIFFER by {l0_max_diff:.2e}")
        print(f"  This means FourWay routing affects even non-routed layer 0 gradients!")


# Run in both fp32 and bf16
run_test("float32", torch.float32)
run_test("bfloat16", torch.bfloat16)
