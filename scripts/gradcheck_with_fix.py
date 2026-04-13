"""Verify: does replacing Olmo2RMSNorm with nn.RMSNorm fix the gradient bug?

Runs the same torch.autograd.gradcheck that FAILED before, but on a model
where all Olmo2RMSNorm have been replaced with nn.RMSNorm.

If PASS → the fix is confirmed, training should work.
If FAIL → nn.RMSNorm didn't fix it, need to look elsewhere.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from transformers import Olmo2Config, Olmo2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model.olmo_graph import FourWayDAGFormer


def replace_olmo_rmsnorm(model):
    count = 0
    for name, module in list(model.named_modules()):
        if type(module).__name__ == "Olmo2RMSNorm":
            hidden_size = module.weight.shape[0]
            eps = module.variance_epsilon
            new_norm = nn.RMSNorm(hidden_size, eps=eps).to(
                device=module.weight.device, dtype=module.weight.dtype)
            new_norm.weight = module.weight
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_norm)
            count += 1
    return count


device = torch.device("cuda")
torch.manual_seed(42)

L, H, D, V = 3, 2, 64, 100
B, T = 1, 8

print("Building model...")
mc = Olmo2Config(hidden_size=D, num_hidden_layers=L, num_attention_heads=H,
                 num_key_value_heads=H, intermediate_size=D*4, vocab_size=V,
                 tie_word_embeddings=True, max_position_embeddings=64)

# === WITHOUT fix (original Olmo2RMSNorm) ===
print("\n" + "=" * 60)
print("CONTROL: Original Olmo2RMSNorm (should FAIL)")
print("=" * 60)

base1 = Olmo2ForCausalLM(mc).to(device, dtype=torch.float64)
fw1 = FourWayDAGFormer(model=base1, num_layers=L, num_heads=H).to(device, dtype=torch.float64)
fw1.eval()

ids = torch.randint(0, V, (B, T), device=device)
labs = torch.randint(0, V, (B, T), device=device)

alphas1 = []
rw1_template = {"q": [], "k": [], "v": [], "r": []}
for l in range(1, L):
    n_src = l + 1
    for s in ("q", "k", "v"):
        a = torch.randn(B, T, H, n_src, device=device, dtype=torch.float64, requires_grad=True) * 0.1
        with torch.no_grad(): a[..., -1] += 1.0
        rw1_template[s].append(a)
        alphas1.append(a)
    ar = torch.randn(B, T, n_src, device=device, dtype=torch.float64, requires_grad=True) * 0.1
    with torch.no_grad(): ar[..., -1] += 1.0
    rw1_template["r"].append(ar)
    alphas1.append(ar)

def loss1(*alpha_tensors):
    idx = 0
    rw = {"q": [], "k": [], "v": [], "r": []}
    for l in range(1, L):
        rw["q"].append(alpha_tensors[idx]); idx += 1
        rw["k"].append(alpha_tensors[idx]); idx += 1
        rw["v"].append(alpha_tensors[idx]); idx += 1
        rw["r"].append(alpha_tensors[idx]); idx += 1
    return F.cross_entropy(fw1(ids, rw).view(-1, V), labs.view(-1))

try:
    ok1 = gradcheck(loss1, tuple(alphas1), eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=False)
    print(f"  Result: {'PASS ✓' if ok1 else 'FAIL ✗'}")
except Exception as e:
    print(f"  Result: ERROR ({e})")

# === WITH fix (nn.RMSNorm) ===
print("\n" + "=" * 60)
print("FIX: nn.RMSNorm replacing Olmo2RMSNorm (should PASS)")
print("=" * 60)

torch.manual_seed(42)  # same init
base2 = Olmo2ForCausalLM(mc).to(device, dtype=torch.float64)
n = replace_olmo_rmsnorm(base2)
print(f"  Replaced {n} modules")
base2 = base2.to(device, dtype=torch.float64)
fw2 = FourWayDAGFormer(model=base2, num_layers=L, num_heads=H).to(device, dtype=torch.float64)
fw2.eval()

alphas2 = []
for l in range(1, L):
    n_src = l + 1
    for s in ("q", "k", "v"):
        a = torch.randn(B, T, H, n_src, device=device, dtype=torch.float64, requires_grad=True) * 0.1
        with torch.no_grad(): a[..., -1] += 1.0
        alphas2.append(a)
    ar = torch.randn(B, T, n_src, device=device, dtype=torch.float64, requires_grad=True) * 0.1
    with torch.no_grad(): ar[..., -1] += 1.0
    alphas2.append(ar)

def loss2(*alpha_tensors):
    idx = 0
    rw = {"q": [], "k": [], "v": [], "r": []}
    for l in range(1, L):
        rw["q"].append(alpha_tensors[idx]); idx += 1
        rw["k"].append(alpha_tensors[idx]); idx += 1
        rw["v"].append(alpha_tensors[idx]); idx += 1
        rw["r"].append(alpha_tensors[idx]); idx += 1
    return F.cross_entropy(fw2(ids, rw).view(-1, V), labs.view(-1))

try:
    ok2 = gradcheck(loss2, tuple(alphas2), eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=False)
    print(f"  Result: {'PASS ✓' if ok2 else 'FAIL ✗'}")
except Exception as e:
    print(f"  Result: ERROR ({e})")

# === Summary ===
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  Olmo2RMSNorm (control): {'PASS ✓' if ok1 else 'FAIL ✗'}")
print(f"  nn.RMSNorm (fix):       {'PASS ✓' if ok2 else 'FAIL ✗'}")
if not ok1 and ok2:
    print("  → FIX CONFIRMED: nn.RMSNorm resolves the gradient bug!")
elif ok1 and ok2:
    print("  → Both pass — Olmo2RMSNorm was not the issue after all")
elif not ok1 and not ok2:
    print("  → Both fail — nn.RMSNorm does NOT fix the gradient bug!")
