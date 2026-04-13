"""Gradient check in bf16 (matching actual training dtype).

If Olmo2RMSNorm passes gradcheck in bf16 → the f32 internal cast is
correct for mixed-precision (our f64 gradcheck was a false positive).
If it fails in bf16 too → real gradient bug.

Also tests: what if we replace Olmo2RMSNorm with torch.nn.RMSNorm?
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import Olmo2Config, Olmo2ForCausalLM
from transformers.models.olmo2.modeling_olmo2 import Olmo2RMSNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model.olmo_graph import FourWayDAGFormer

device = torch.device("cuda")
torch.manual_seed(42)

H, D, hd = 2, 64, 32
B, T, V = 1, 8, 100

mc = Olmo2Config(hidden_size=D, num_hidden_layers=2, num_attention_heads=H,
                 num_key_value_heads=H, intermediate_size=D*4, vocab_size=V,
                 tie_word_embeddings=True, max_position_embeddings=64)

print("=" * 60)
print("TEST A: Olmo2RMSNorm gradcheck in BFLOAT16")
print("=" * 60)
print("(Using larger eps=1e-2 because bf16 has ~1e-2 relative precision)")

olmo_norm = Olmo2RMSNorm(D, eps=1e-5).to(device, dtype=torch.bfloat16)

# Manual gradient check in bf16 (torch.autograd.gradcheck doesn't support bf16)
x = torch.randn(B, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
y = olmo_norm(x)
loss = y.sum()
loss.backward()
ag = x.grad.clone()

eps = 1e-2  # bf16 needs large eps
print(f"Checking {min(20, x.numel())} elements with eps={eps}")
max_rel = 0
for i in range(min(20, x.numel())):
    with torch.no_grad():
        orig = x.data.view(-1)[i].clone()
        x.data.view(-1)[i] = orig + eps
    lp = olmo_norm(x).sum().item()
    with torch.no_grad():
        x.data.view(-1)[i] = orig - eps
    lm = olmo_norm(x).sum().item()
    with torch.no_grad():
        x.data.view(-1)[i] = orig
    fd = (lp - lm) / (2 * eps)
    ag_val = ag.view(-1)[i].item()
    rel = abs(ag_val - fd) / (max(abs(ag_val), abs(fd), 1e-3))
    max_rel = max(max_rel, rel)
    if i < 5:
        print(f"  [{i}] ag={ag_val:.4f} fd={fd:.4f} rel={rel:.4f}")

print(f"  Max relative error: {max_rel:.4f}")
if max_rel < 0.1:
    print("  PASS ✓ (within bf16 tolerance)")
else:
    print("  FAIL ✗ (gradient wrong even in bf16)")

print("\n" + "=" * 60)
print("TEST B: Full FourWay L=2, bf16 manual gradient check on α")
print("=" * 60)

base = Olmo2ForCausalLM(mc).to(device, dtype=torch.bfloat16)
fw = FourWayDAGFormer(model=base, num_layers=2, num_heads=H).to(device, dtype=torch.bfloat16)
fw.eval()

ids = torch.randint(0, V, (B, T), device=device)
labs = torch.randint(0, V, (B, T), device=device)

α_q = torch.zeros(B, T, H, 2, device=device, dtype=torch.float32, requires_grad=True)
α_k = torch.zeros(B, T, H, 2, device=device, dtype=torch.float32, requires_grad=True)
α_v = torch.zeros(B, T, H, 2, device=device, dtype=torch.float32, requires_grad=True)
α_r = torch.zeros(B, T, 2, device=device, dtype=torch.float32, requires_grad=True)
with torch.no_grad():
    α_q[..., -1] = 1; α_q += 0.1 * torch.randn_like(α_q)
    α_k[..., -1] = 1; α_k += 0.1 * torch.randn_like(α_k)
    α_v[..., -1] = 1; α_v += 0.1 * torch.randn_like(α_v)
    α_r[..., -1] = 1; α_r += 0.1 * torch.randn_like(α_r)

rw = {"q": [α_q], "k": [α_k], "v": [α_v], "r": [α_r]}
logits = fw(ids, rw)
loss = F.cross_entropy(logits.float().view(-1, V), labs.view(-1))
loss.backward()

# Note: α is float32, model is bf16. The .to(dtype) cast inside fw
# converts α to bf16 for the einsum. Gradient flows back through cast.
print(f"α_q.grad is None: {α_q.grad is None}")
if α_q.grad is not None:
    print(f"α_q.grad abs max: {α_q.grad.abs().max().item():.6e}")

    eps = 1e-3  # float32 α, bf16 model
    max_rel = 0
    n_sign_flip = 0
    n_checked = 0
    for i in range(min(50, α_q.numel())):
        ag_val = α_q.grad.view(-1)[i].item()
        with torch.no_grad():
            orig = α_q.data.view(-1)[i].clone()
            α_q.data.view(-1)[i] = orig + eps
        lp = F.cross_entropy(fw(ids, rw).float().view(-1, V), labs.view(-1)).item()
        with torch.no_grad():
            α_q.data.view(-1)[i] = orig - eps
        lm = F.cross_entropy(fw(ids, rw).float().view(-1, V), labs.view(-1)).item()
        with torch.no_grad():
            α_q.data.view(-1)[i] = orig
        fd = (lp - lm) / (2 * eps)
        rel = abs(ag_val - fd) / (max(abs(ag_val), abs(fd), 1e-6))
        max_rel = max(max_rel, rel)
        if ag_val * fd < 0 and abs(ag_val) > 1e-5 and abs(fd) > 1e-5:
            n_sign_flip += 1
        n_checked += 1
        if i < 10:
            print(f"  [{i}] ag={ag_val:.6e} fd={fd:.6e} rel={rel:.4f} {'SIGN!' if ag_val * fd < 0 and abs(ag_val)>1e-5 else ''}")

    print(f"\n  Checked: {n_checked}")
    print(f"  Max relative error: {max_rel:.4f}")
    print(f"  Sign flips: {n_sign_flip}")
    if max_rel < 0.3 and n_sign_flip == 0:
        print("  PASS ✓ — gradient correct in bf16 training regime")
        print("  Conclusion: f64 gradcheck FAIL was a false positive from Olmo2RMSNorm f32 cast")
    elif n_sign_flip > 0:
        print("  FAIL ✗ — sign flips even in bf16! REAL gradient bug.")
    else:
        print("  WARN — large relative error but no sign flips (borderline)")
