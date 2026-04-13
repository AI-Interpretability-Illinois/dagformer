"""Isolate WHERE the gradient bug is.

Test 1: einsum backward in isolation
Test 2: single-layer FourWay (L=2, one routing layer)
Test 3: layer_outputs accumulation pattern
Test 4: RoPE interaction
Test 5: q_norm interaction
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.autograd import gradcheck
from transformers import Olmo2Config, Olmo2ForCausalLM
from transformers.models.olmo2.modeling_olmo2 import apply_rotary_pos_emb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model.olmo_graph import FourWayDAGFormer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

H, D, V = 2, 64, 100
hd = D // H
B, T = 1, 8

def check(name, fn, inputs, **kwargs):
    try:
        ok = gradcheck(fn, inputs, eps=1e-6, atol=1e-4, rtol=1e-3,
                        raise_exception=False, **kwargs)
        print(f"  {name}: {'PASS ✓' if ok else 'FAIL ✗'}")
        return ok
    except Exception as e:
        print(f"  {name}: ERROR ({e})")
        return False

print("=" * 60)
print("TEST 1: Raw einsum backward")
print("=" * 60)

q_stack = torch.randn(2, B, T, H, hd, device=device, dtype=torch.float64)
α = torch.randn(B, T, H, 2, device=device, dtype=torch.float64, requires_grad=True)

def einsum_fn(a):
    return torch.einsum('lbthd, bthl -> bhtd', q_stack, a).sum()

check("einsum 'lbthd, bthl -> bhtd'", einsum_fn, (α,))

# Also test with gradient through both inputs
q_stack2 = torch.randn(2, B, T, H, hd, device=device, dtype=torch.float64, requires_grad=True)
α2 = torch.randn(B, T, H, 2, device=device, dtype=torch.float64, requires_grad=True)

def einsum_fn2(qs, a):
    return torch.einsum('lbthd, bthl -> bhtd', qs, a).sum()

check("einsum (both inputs grad)", einsum_fn2, (q_stack2, α2))

print("\n" + "=" * 60)
print("TEST 2: einsum + q_norm")
print("=" * 60)

from torch import nn
q_norm = nn.RMSNorm(D, eps=1e-5).to(device, dtype=torch.float64)

α3 = torch.randn(B, T, H, 2, device=device, dtype=torch.float64, requires_grad=True)

def einsum_qnorm(a):
    from einops import rearrange
    q = torch.einsum('lbthd, bthl -> bhtd', q_stack, a)
    q_cat = rearrange(q, 'b h t d -> b t (h d)')
    q_normed = q_norm(q_cat)
    return q_normed.sum()

check("einsum + RMSNorm", einsum_qnorm, (α3,))

print("\n" + "=" * 60)
print("TEST 3: einsum + q_norm + RoPE")
print("=" * 60)

mc = Olmo2Config(hidden_size=D, num_hidden_layers=2, num_attention_heads=H,
                 num_key_value_heads=H, intermediate_size=D*4, vocab_size=V,
                 tie_word_embeddings=True, max_position_embeddings=64)
base = Olmo2ForCausalLM(mc).to(device, dtype=torch.float64)
base.eval()

emb = torch.randn(B, T, D, device=device, dtype=torch.float64)
pos_ids = torch.arange(T, device=device).unsqueeze(0)
cos, sin = base.model.rotary_emb(emb, pos_ids)
attn0 = base.model.layers[0].self_attn

α4 = torch.randn(B, T, H, 2, device=device, dtype=torch.float64, requires_grad=True)

def einsum_qnorm_rope(a):
    from einops import rearrange
    q = torch.einsum('lbthd, bthl -> bhtd', q_stack, a)
    q_cat = rearrange(q, 'b h t d -> b t (h d)')
    q_normed = attn0.q_norm(q_cat)
    q_heads = rearrange(q_normed, 'b t (h d) -> b h t d', h=H)
    # Use a dummy K for RoPE (same shape, not part of gradient)
    k_dummy = torch.randn_like(q_heads)
    q_rot, _ = apply_rotary_pos_emb(q_heads, k_dummy, cos, sin)
    return q_rot.sum()

check("einsum + q_norm + RoPE", einsum_qnorm_rope, (α4,))

print("\n" + "=" * 60)
print("TEST 4: Full single-layer FourWay (L=2)")
print("=" * 60)

mc2 = Olmo2Config(hidden_size=D, num_hidden_layers=2, num_attention_heads=H,
                  num_key_value_heads=H, intermediate_size=D*4, vocab_size=V,
                  tie_word_embeddings=True, max_position_embeddings=64)
base2 = Olmo2ForCausalLM(mc2).to(device, dtype=torch.float64)
fw2 = FourWayDAGFormer(model=base2, num_layers=2, num_heads=H).to(device, dtype=torch.float64)
fw2.eval()

ids = torch.randint(0, V, (B, T), device=device)
labs = torch.randint(0, V, (B, T), device=device)

α_q2 = torch.randn(B, T, H, 2, device=device, dtype=torch.float64, requires_grad=True) * 0.1
α_k2 = torch.randn(B, T, H, 2, device=device, dtype=torch.float64, requires_grad=True) * 0.1
α_v2 = torch.randn(B, T, H, 2, device=device, dtype=torch.float64, requires_grad=True) * 0.1
α_r2 = torch.randn(B, T, 2, device=device, dtype=torch.float64, requires_grad=True) * 0.1
with torch.no_grad():
    α_q2[..., -1] += 1; α_k2[..., -1] += 1; α_v2[..., -1] += 1; α_r2[..., -1] += 1

def fw2_q(a):
    rw = {"q": [a], "k": [α_k2.detach()], "v": [α_v2.detach()], "r": [α_r2.detach()]}
    return F.cross_entropy(fw2(ids, rw).view(-1, V), labs.view(-1))

def fw2_k(a):
    rw = {"q": [α_q2.detach()], "k": [a], "v": [α_v2.detach()], "r": [α_r2.detach()]}
    return F.cross_entropy(fw2(ids, rw).view(-1, V), labs.view(-1))

def fw2_v(a):
    rw = {"q": [α_q2.detach()], "k": [α_k2.detach()], "v": [a], "r": [α_r2.detach()]}
    return F.cross_entropy(fw2(ids, rw).view(-1, V), labs.view(-1))

def fw2_r(a):
    rw = {"q": [α_q2.detach()], "k": [α_k2.detach()], "v": [α_v2.detach()], "r": [a]}
    return F.cross_entropy(fw2(ids, rw).view(-1, V), labs.view(-1))

def fw2_all(aq, ak, av, ar):
    rw = {"q": [aq], "k": [ak], "v": [av], "r": [ar]}
    return F.cross_entropy(fw2(ids, rw).view(-1, V), labs.view(-1))

check("FW L=2, α_q only", fw2_q, (α_q2,))
check("FW L=2, α_k only", fw2_k, (α_k2,))
check("FW L=2, α_v only", fw2_v, (α_v2,))
check("FW L=2, α_r only", fw2_r, (α_r2,))
check("FW L=2, all α", fw2_all, (α_q2, α_k2, α_v2, α_r2))

print("\n" + "=" * 60)
print("TEST 5: Full L=3 FourWay (original failing case)")
print("=" * 60)

mc3 = Olmo2Config(hidden_size=D, num_hidden_layers=3, num_attention_heads=H,
                  num_key_value_heads=H, intermediate_size=D*4, vocab_size=V,
                  tie_word_embeddings=True, max_position_embeddings=64)
base3 = Olmo2ForCausalLM(mc3).to(device, dtype=torch.float64)
fw3 = FourWayDAGFormer(model=base3, num_layers=3, num_heads=H).to(device, dtype=torch.float64)
fw3.eval()

# Only check layer 1 α, hold layer 2 fixed (no grad)
α_q3_l1 = torch.randn(B, T, H, 2, device=device, dtype=torch.float64, requires_grad=True) * 0.1
α_k3_l1 = torch.randn(B, T, H, 2, device=device, dtype=torch.float64, requires_grad=True) * 0.1
α_v3_l1 = torch.randn(B, T, H, 2, device=device, dtype=torch.float64, requires_grad=True) * 0.1
α_r3_l1 = torch.randn(B, T, 2, device=device, dtype=torch.float64, requires_grad=True) * 0.1
with torch.no_grad():
    α_q3_l1[..., -1] += 1; α_k3_l1[..., -1] += 1; α_v3_l1[..., -1] += 1; α_r3_l1[..., -1] += 1

# Identity for layer 2
α_q3_l2 = torch.zeros(B, T, H, 3, device=device, dtype=torch.float64); α_q3_l2[..., -1] = 1
α_k3_l2 = torch.zeros(B, T, H, 3, device=device, dtype=torch.float64); α_k3_l2[..., -1] = 1
α_v3_l2 = torch.zeros(B, T, H, 3, device=device, dtype=torch.float64); α_v3_l2[..., -1] = 1
α_r3_l2 = torch.zeros(B, T, 3, device=device, dtype=torch.float64); α_r3_l2[..., -1] = 1

def fw3_l1_all(aq, ak, av, ar):
    rw = {"q": [aq, α_q3_l2], "k": [ak, α_k3_l2], "v": [av, α_v3_l2], "r": [ar, α_r3_l2]}
    return F.cross_entropy(fw3(ids, rw).view(-1, V), labs.view(-1))

check("FW L=3, layer 1 only (layer 2 identity)", fw3_l1_all, (α_q3_l1, α_k3_l1, α_v3_l1, α_r3_l1))

# Also check layer 2 α with layer 1 at identity
α_q3_l2g = torch.randn(B, T, H, 3, device=device, dtype=torch.float64, requires_grad=True) * 0.1
α_k3_l2g = torch.randn(B, T, H, 3, device=device, dtype=torch.float64, requires_grad=True) * 0.1
α_v3_l2g = torch.randn(B, T, H, 3, device=device, dtype=torch.float64, requires_grad=True) * 0.1
α_r3_l2g = torch.randn(B, T, 3, device=device, dtype=torch.float64, requires_grad=True) * 0.1
with torch.no_grad():
    α_q3_l2g[..., -1] += 1; α_k3_l2g[..., -1] += 1; α_v3_l2g[..., -1] += 1; α_r3_l2g[..., -1] += 1

α_id_l1_q = torch.zeros(B, T, H, 2, device=device, dtype=torch.float64); α_id_l1_q[..., -1] = 1
α_id_l1_k = torch.zeros(B, T, H, 2, device=device, dtype=torch.float64); α_id_l1_k[..., -1] = 1
α_id_l1_v = torch.zeros(B, T, H, 2, device=device, dtype=torch.float64); α_id_l1_v[..., -1] = 1
α_id_l1_r = torch.zeros(B, T, 2, device=device, dtype=torch.float64); α_id_l1_r[..., -1] = 1

def fw3_l2_all(aq, ak, av, ar):
    rw = {"q": [α_id_l1_q, aq], "k": [α_id_l1_k, ak], "v": [α_id_l1_v, av], "r": [α_id_l1_r, ar]}
    return F.cross_entropy(fw3(ids, rw).view(-1, V), labs.view(-1))

check("FW L=3, layer 2 only (layer 1 identity)", fw3_l2_all, (α_q3_l2g, α_k3_l2g, α_v3_l2g, α_r3_l2g))

# Check both layers simultaneously
def fw3_both(aq1, ak1, av1, ar1, aq2, ak2, av2, ar2):
    rw = {"q": [aq1, aq2], "k": [ak1, ak2], "v": [av1, av2], "r": [ar1, ar2]}
    return F.cross_entropy(fw3(ids, rw).view(-1, V), labs.view(-1))

aq1 = torch.randn(B, T, H, 2, device=device, dtype=torch.float64, requires_grad=True) * 0.1
ak1 = torch.randn(B, T, H, 2, device=device, dtype=torch.float64, requires_grad=True) * 0.1
av1 = torch.randn(B, T, H, 2, device=device, dtype=torch.float64, requires_grad=True) * 0.1
ar1 = torch.randn(B, T, 2, device=device, dtype=torch.float64, requires_grad=True) * 0.1
aq2 = torch.randn(B, T, H, 3, device=device, dtype=torch.float64, requires_grad=True) * 0.1
ak2 = torch.randn(B, T, H, 3, device=device, dtype=torch.float64, requires_grad=True) * 0.1
av2 = torch.randn(B, T, H, 3, device=device, dtype=torch.float64, requires_grad=True) * 0.1
ar2 = torch.randn(B, T, 3, device=device, dtype=torch.float64, requires_grad=True) * 0.1
with torch.no_grad():
    aq1[..., -1] += 1; ak1[..., -1] += 1; av1[..., -1] += 1; ar1[..., -1] += 1
    aq2[..., -1] += 1; ak2[..., -1] += 1; av2[..., -1] += 1; ar2[..., -1] += 1

check("FW L=3, both layers", fw3_both, (aq1, ak1, av1, ar1, aq2, ak2, av2, ar2))
