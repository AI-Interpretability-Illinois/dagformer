"""Test: SDPA + nn.RMSNorm together — does this eliminate ALL gradient deviation?"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Olmo2Config, Olmo2ForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model.olmo_graph import FourWayDAGFormer

def replace_olmo_rmsnorm(model):
    count = 0
    for name, module in list(model.named_modules()):
        if type(module).__name__ == "Olmo2RMSNorm":
            new_norm = nn.RMSNorm(module.weight.shape[0], eps=module.variance_epsilon).to(
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

L, H, D, V = 4, 4, 128, 1000
B, T = 2, 64

mc = Olmo2Config(hidden_size=D, num_hidden_layers=L, num_attention_heads=H,
                 num_key_value_heads=H, intermediate_size=D*4, vocab_size=V,
                 tie_word_embeddings=True, max_position_embeddings=256)

base = Olmo2ForCausalLM(mc)
n = replace_olmo_rmsnorm(base)
print(f"Replaced {n} Olmo2RMSNorm → nn.RMSNorm")
base = base.to(device, dtype=torch.bfloat16)
fw = FourWayDAGFormer(model=base, num_layers=L, num_heads=H).to(device, dtype=torch.bfloat16)

ids = torch.randint(0, V, (B, T), device=device)
labels = torch.randint(0, V, (B, T), device=device)

# Dense
base.zero_grad()
dense_loss = F.cross_entropy(base(input_ids=ids).logits.float().view(-1, V), labels.view(-1))
dense_loss.backward()
dense_grads = {n: p.grad.clone() for n, p in base.named_parameters() if p.grad is not None}

# FourWay identity
base.zero_grad()
rw_id = {"q": [], "k": [], "v": [], "r": []}
for l in range(1, L):
    ns = l + 1
    for s in ("q", "k", "v"):
        a = torch.zeros(B, T, H, ns, device=device, dtype=torch.bfloat16); a[..., -1] = 1
        rw_id[s].append(a)
    ar = torch.zeros(B, T, ns, device=device, dtype=torch.bfloat16); ar[..., -1] = 1
    rw_id["r"].append(ar)

fw_loss = F.cross_entropy(fw(ids, rw_id).float().view(-1, V), labels.view(-1))
fw_loss.backward()

print(f"Dense loss: {dense_loss.item():.6f}, FW loss: {fw_loss.item():.6f}")

max_rel = 0
min_cos = 1
for name in sorted(dense_grads.keys()):
    p = dict(base.named_parameters())[name]
    if p.grad is None: continue
    dg = dense_grads[name].float()
    fg = p.grad.float()
    rel = (dg - fg).abs().max().item() / (dg.abs().max().item() + 1e-10)
    cos = F.cosine_similarity(dg.flatten().unsqueeze(0), fg.flatten().unsqueeze(0)).item()
    max_rel = max(max_rel, rel)
    min_cos = min(min_cos, cos)

print(f"\nmax_rel_diff={max_rel:.6f}, min_cosine_sim={min_cos:.6f}")
print(f"\nComparison:")
print(f"  Manual attn + Olmo2RMSNorm:  max_rel=0.0710, min_cos=0.999262")
print(f"  SDPA + Olmo2RMSNorm:         max_rel=0.0234, min_cos=0.999896")
print(f"  SDPA + nn.RMSNorm:           max_rel={max_rel:.4f}, min_cos={min_cos:.6f}")
if max_rel < 0.005:
    print(f"\n✓ BOTH FIXES TOGETHER ELIMINATE GRADIENT DEVIATION!")
elif max_rel < 0.01:
    print(f"\n~ Nearly eliminated — very small residual deviation")
else:
    print(f"\n✗ Still significant deviation — there's yet another source")
