"""Test: does switching to SDPA fix the bf16 gradient deviation?
Compares dense vs FourWay gradients at identity init in bf16.
Previous result: 7% max_rel_diff. Expected after SDPA fix: ~0%.
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
torch.manual_seed(42)

L, H, D, V = 4, 4, 128, 1000
B, T = 2, 64

mc = Olmo2Config(hidden_size=D, num_hidden_layers=L, num_attention_heads=H,
                 num_key_value_heads=H, intermediate_size=D*4, vocab_size=V,
                 tie_word_embeddings=True, max_position_embeddings=256)

base = Olmo2ForCausalLM(mc).to(device, dtype=torch.bfloat16)
fw = FourWayDAGFormer(model=base, num_layers=L, num_heads=H).to(device, dtype=torch.bfloat16)

ids = torch.randint(0, V, (B, T), device=device)
labels = torch.randint(0, V, (B, T), device=device)

# Dense forward + backward
base.zero_grad()
dense_logits = base(input_ids=ids).logits
dense_loss = F.cross_entropy(dense_logits.float().view(-1, V), labels.view(-1))
dense_loss.backward()
dense_grads = {n: p.grad.clone() for n, p in base.named_parameters() if p.grad is not None}

# FourWay forward + backward with identity α
base.zero_grad()
rw_id = {"q": [], "k": [], "v": [], "r": []}
for l in range(1, L):
    n_src = l + 1
    for s in ("q", "k", "v"):
        a = torch.zeros(B, T, H, n_src, device=device, dtype=torch.bfloat16)
        a[..., -1] = 1.0
        rw_id[s].append(a)
    ar = torch.zeros(B, T, n_src, device=device, dtype=torch.bfloat16)
    ar[..., -1] = 1.0
    rw_id["r"].append(ar)

fw_logits = fw(ids, rw_id)
fw_loss = F.cross_entropy(fw_logits.float().view(-1, V), labels.view(-1))
fw_loss.backward()

print(f"Dense loss:  {dense_loss.item():.6f}")
print(f"FW loss:     {fw_loss.item():.6f}")
print(f"Loss diff:   {abs(dense_loss.item() - fw_loss.item()):.2e}")

max_rel = 0
min_cos = 1
print(f"\n{'Parameter':<45} {'max_diff':>10} {'rel_diff':>10} {'cos_sim':>10}")
print("-" * 80)

for name in sorted(dense_grads.keys()):
    p = dict(base.named_parameters())[name]
    if p.grad is None:
        continue
    dg = dense_grads[name].float()
    fg = p.grad.float()
    max_diff = (dg - fg).abs().max().item()
    rel_diff = max_diff / (dg.abs().max().item() + 1e-10)
    cos = F.cosine_similarity(dg.flatten().unsqueeze(0), fg.flatten().unsqueeze(0)).item()
    max_rel = max(max_rel, rel_diff)
    min_cos = min(min_cos, cos)
    if "layers.0." in name or "layers.1." in name or "embed" in name:
        print(f"{name:<45} {max_diff:>10.2e} {rel_diff:>10.4f} {cos:>10.6f}")

print(f"\nOverall: max_rel_diff={max_rel:.4f}, min_cosine_sim={min_cos:.6f}")
print(f"\nPrevious (manual attn): max_rel_diff=0.0710, min_cos=0.999262")
if max_rel < 0.01:
    print(f"PASS ✓ — SDPA fixed the gradient deviation! ({max_rel:.4f} vs 0.0710)")
elif max_rel < 0.03:
    print(f"IMPROVED — deviation reduced ({max_rel:.4f} vs 0.0710)")
else:
    print(f"NO CHANGE — still {max_rel:.4f} (SDPA is not the cause)")
