"""Gradcheck on CPU only — no CUDA, no non-determinism excuses."""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from transformers import Olmo2Config, Olmo2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model.olmo_graph import FourWayDAGFormer

torch.manual_seed(42)
device = torch.device("cpu")

L, H, D, V = 3, 2, 64, 100
B, T = 1, 8

mc = Olmo2Config(hidden_size=D, num_hidden_layers=L, num_attention_heads=H,
                 num_key_value_heads=H, intermediate_size=D*4, vocab_size=V,
                 tie_word_embeddings=True, max_position_embeddings=64)

base = Olmo2ForCausalLM(mc).to(device, dtype=torch.float64)
fw = FourWayDAGFormer(model=base, num_layers=L, num_heads=H).to(device, dtype=torch.float64)
fw.eval()

ids = torch.randint(0, V, (B, T), device=device)
labs = torch.randint(0, V, (B, T), device=device)

alphas = []
for l in range(1, L):
    n_src = l + 1
    for s in ("q", "k", "v"):
        a = torch.randn(B, T, H, n_src, device=device, dtype=torch.float64, requires_grad=True) * 0.1
        with torch.no_grad(): a[..., -1] += 1.0
        alphas.append(a)
    ar = torch.randn(B, T, n_src, device=device, dtype=torch.float64, requires_grad=True) * 0.1
    with torch.no_grad(): ar[..., -1] += 1.0
    alphas.append(ar)

def loss_fn(*alpha_tensors):
    idx = 0
    rw = {"q": [], "k": [], "v": [], "r": []}
    for l in range(1, L):
        rw["q"].append(alpha_tensors[idx]); idx += 1
        rw["k"].append(alpha_tensors[idx]); idx += 1
        rw["v"].append(alpha_tensors[idx]); idx += 1
        rw["r"].append(alpha_tensors[idx]); idx += 1
    return F.cross_entropy(fw(ids, rw).view(-1, V), labs.view(-1))

print(f"Full FourWay gradcheck on CPU (L={L}, H={H}, D={D})")
print(f"α tensors: {len(alphas)}, total elements: {sum(a.numel() for a in alphas)}")

ok = gradcheck(loss_fn, tuple(alphas), eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=False)
print(f"\nResult: {'PASS ✓' if ok else 'FAIL ✗'}")

if not ok:
    print("\nCPU also fails → real autograd bug, not CUDA issue")
else:
    print("\nCPU passes → if GPU failed, it was CUDA non-determinism (false positive)")
