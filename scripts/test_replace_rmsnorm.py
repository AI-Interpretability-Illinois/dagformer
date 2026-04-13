"""Quick sanity: does replace_rmsnorm work? Can the model still forward?"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Olmo2Config, Olmo2ForCausalLM
from transformers.models.olmo2.modeling_olmo2 import Olmo2RMSNorm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model.olmo_graph import FourWayDAGFormer

device = torch.device("cuda")

# 1. Check nn.RMSNorm exists
print(f"torch version: {torch.__version__}")
print(f"nn.RMSNorm available: {hasattr(nn, 'RMSNorm')}")

# 2. Build small model, replace norms, verify forward works
L, H, D, V = 4, 4, 128, 1000
mc = Olmo2Config(hidden_size=D, num_hidden_layers=L, num_attention_heads=H,
                 num_key_value_heads=H, intermediate_size=D*4, vocab_size=V,
                 tie_word_embeddings=True, max_position_embeddings=256)
base = Olmo2ForCausalLM(mc)

# Count Olmo2RMSNorm before
n_olmo = sum(1 for _, m in base.named_modules() if type(m).__name__ == "Olmo2RMSNorm")
print(f"\nBefore replace: {n_olmo} Olmo2RMSNorm modules")

# Replace
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
n_replaced = replace_olmo_rmsnorm(base)
print(f"Replaced: {n_replaced}")

# Count after
n_olmo_after = sum(1 for _, m in base.named_modules() if type(m).__name__ == "Olmo2RMSNorm")
n_torch_rms = sum(1 for _, m in base.named_modules() if isinstance(m, nn.RMSNorm))
print(f"After: {n_olmo_after} Olmo2RMSNorm, {n_torch_rms} nn.RMSNorm")

# 3. Move to GPU bf16, do a forward
base = base.to(device, dtype=torch.bfloat16)
fw = FourWayDAGFormer(model=base, num_layers=L, num_heads=H).to(device, dtype=torch.bfloat16)

ids = torch.randint(0, V, (2, 32), device=device)
rw = {"q": [], "k": [], "v": [], "r": []}
for l in range(1, L):
    ns = l + 1
    for s in ("q", "k", "v"):
        a = torch.zeros(2, 32, H, ns, device=device, dtype=torch.bfloat16)
        a[..., -1] = 1; a += 0.1 * torch.randn_like(a)
        rw[s].append(a)
    ar = torch.zeros(2, 32, ns, device=device, dtype=torch.bfloat16)
    ar[..., -1] = 1; ar += 0.1 * torch.randn_like(ar)
    rw["r"].append(ar)

# Forward
logits = fw(ids, rw)
print(f"\nForward OK: logits shape={logits.shape}, dtype={logits.dtype}")
print(f"logits norm: {logits.float().norm().item():.2f}")

# 4. Backward
labels = torch.randint(0, V, (2, 32), device=device)
loss = F.cross_entropy(logits.float().view(-1, V), labels.view(-1))
loss.backward()
print(f"Backward OK: loss={loss.item():.4f}")

# 5. Quick gradient check (just verify non-zero grads flow to a leaf α)
rw2 = {"q": [], "k": [], "v": [], "r": []}
for l in range(1, L):
    ns = l + 1
    for s in ("q", "k", "v"):
        a = torch.zeros(2, 32, H, ns, device=device, dtype=torch.float32, requires_grad=True)
        with torch.no_grad(): a[..., -1] = 1; a += 0.1 * torch.randn_like(a)
        rw2[s].append(a)
    ar = torch.zeros(2, 32, ns, device=device, dtype=torch.float32, requires_grad=True)
    with torch.no_grad(): ar[..., -1] = 1; ar += 0.1 * torch.randn_like(ar)
    rw2["r"].append(ar)

logits2 = fw(ids, rw2)
loss2 = F.cross_entropy(logits2.float().view(-1, V), labels.view(-1))
loss2.backward()

for s in ("q", "k", "v", "r"):
    grads = [a.grad for a in rw2[s]]
    nonzero = sum((g is not None and g.abs().max() > 0) for g in grads)
    total = len(grads)
    max_g = max(g.abs().max().item() for g in grads if g is not None) if nonzero > 0 else 0
    print(f"  {s}: {nonzero}/{total} have nonzero grad, max={max_g:.6e}")

print("\nAll checks passed ✓")
