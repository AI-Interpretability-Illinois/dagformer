"""Test: is CUDA non-determinism causing gradcheck failure?

1. Check if same forward gives same output twice (determinism test)
2. Run gradcheck with torch.use_deterministic_algorithms(True)
3. Run gradcheck on CPU (no CUDA non-determinism)
"""
import sys, os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from transformers import Olmo2Config, Olmo2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model.olmo_graph import FourWayDAGFormer

torch.manual_seed(42)

L, H, D, V = 3, 2, 64, 100
B, T = 1, 8

mc = Olmo2Config(hidden_size=D, num_hidden_layers=L, num_attention_heads=H,
                 num_key_value_heads=H, intermediate_size=D*4, vocab_size=V,
                 tie_word_embeddings=True, max_position_embeddings=64)

def make_alpha(device, dtype):
    alphas = []
    for l in range(1, L):
        n_src = l + 1
        for s in ("q", "k", "v"):
            a = torch.randn(B, T, H, n_src, device=device, dtype=dtype, requires_grad=True) * 0.1
            with torch.no_grad(): a[..., -1] += 1.0
            alphas.append(a)
        ar = torch.randn(B, T, n_src, device=device, dtype=dtype, requires_grad=True) * 0.1
        with torch.no_grad(): ar[..., -1] += 1.0
        alphas.append(ar)
    return alphas

def make_loss_fn(fw, ids, labs):
    def loss_fn(*alpha_tensors):
        idx = 0
        rw = {"q": [], "k": [], "v": [], "r": []}
        for l in range(1, L):
            rw["q"].append(alpha_tensors[idx]); idx += 1
            rw["k"].append(alpha_tensors[idx]); idx += 1
            rw["v"].append(alpha_tensors[idx]); idx += 1
            rw["r"].append(alpha_tensors[idx]); idx += 1
        return F.cross_entropy(fw(ids, rw).view(-1, V), labs.view(-1))
    return loss_fn

# ============================================================
print("TEST 1: Forward determinism on GPU")
print("=" * 60)
device = torch.device("cuda")
base = Olmo2ForCausalLM(mc).to(device, dtype=torch.float64)
fw = FourWayDAGFormer(model=base, num_layers=L, num_heads=H).to(device, dtype=torch.float64)
fw.eval()

ids = torch.randint(0, V, (B, T), device=device)
labs = torch.randint(0, V, (B, T), device=device)
alphas = make_alpha(device, torch.float64)

with torch.no_grad():
    rw1 = {"q": [alphas[0], alphas[4]], "k": [alphas[1], alphas[5]],
            "v": [alphas[2], alphas[6]], "r": [alphas[3], alphas[7]]}
    out1 = fw(ids, rw1)
    out2 = fw(ids, rw1)
    diff = (out1 - out2).abs().max().item()
    print(f"  Same input, two forwards: max diff = {diff:.2e}")
    if diff == 0:
        print("  Forward IS deterministic ✓")
    else:
        print(f"  Forward is NON-DETERMINISTIC! diff={diff:.2e}")

# ============================================================
print("\nTEST 2: Gradcheck on GPU with deterministic algorithms")
print("=" * 60)
try:
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print("  torch.use_deterministic_algorithms(True) set")
except:
    print("  Could not set deterministic algorithms")

alphas_det = make_alpha(device, torch.float64)
loss_fn = make_loss_fn(fw, ids, labs)
ok_det = gradcheck(loss_fn, tuple(alphas_det), eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=False)
print(f"  Gradcheck (GPU, deterministic): {'PASS ✓' if ok_det else 'FAIL ✗'}")
torch.use_deterministic_algorithms(False)

# ============================================================
print("\nTEST 3: Gradcheck on CPU (no CUDA at all)")
print("=" * 60)
torch.manual_seed(42)
cpu = torch.device("cpu")
base_cpu = Olmo2ForCausalLM(mc).to(cpu, dtype=torch.float64)
fw_cpu = FourWayDAGFormer(model=base_cpu, num_layers=L, num_heads=H).to(cpu, dtype=torch.float64)
fw_cpu.eval()

ids_cpu = ids.to(cpu)
labs_cpu = labs.to(cpu)
alphas_cpu = make_alpha(cpu, torch.float64)
loss_fn_cpu = make_loss_fn(fw_cpu, ids_cpu, labs_cpu)
ok_cpu = gradcheck(loss_fn_cpu, tuple(alphas_cpu), eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=False)
print(f"  Gradcheck (CPU): {'PASS ✓' if ok_cpu else 'FAIL ✗'}")

# ============================================================
print("\nSUMMARY")
print("=" * 60)
print(f"  Forward deterministic: {diff == 0}")
print(f"  GPU gradcheck (deterministic): {'PASS' if ok_det else 'FAIL'}")
print(f"  CPU gradcheck: {'PASS' if ok_cpu else 'FAIL'}")
if ok_cpu and not ok_det:
    print("  → CUDA non-determinism is the cause! Forward is correct.")
elif not ok_cpu:
    print("  → CPU also fails. This is a real autograd/math bug, not CUDA issue.")
