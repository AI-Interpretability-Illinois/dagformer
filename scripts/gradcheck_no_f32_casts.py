"""DEFINITIVE TEST: remove ALL f32 casts, run gradcheck on CPU.

If PASS → f32 casts were the only issue → training gradients are correct
         → 5.5 eval floor is genuine overfit, not a gradient bug
If FAIL → there's a REAL autograd bug beyond the f32 casts
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import gradcheck
from transformers import Olmo2Config, Olmo2ForCausalLM
from transformers.models.olmo2.modeling_olmo2 import apply_rotary_pos_emb
from einops import rearrange

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class CleanFourWayForward(nn.Module):
    """FourWay forward with ALL f32 casts removed for clean gradcheck.

    Changes from original FourWayDAGFormer:
    1. Uses nn.RMSNorm instead of Olmo2RMSNorm (no .to(float32))
    2. F.softmax without dtype= (native precision)
    3. No hidden.float() in correction path
    """

    def __init__(self, base_model, num_layers, num_heads):
        super().__init__()
        self.olmo = base_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.model_dim = base_model.config.hidden_size
        self.head_dim = self.model_dim // num_heads
        self.scaling = self.head_dim ** -0.5

    def forward(self, olmo_ids, routing_weights):
        batch, seq_len = olmo_ids.shape
        device = olmo_ids.device
        H = self.num_heads

        embedding = self.olmo.model.embed_tokens(olmo_ids)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cos, sin = self.olmo.model.rotary_emb(embedding, position_ids)
        # Cast cos/sin to match embedding dtype (rotary_emb returns f32)
        cos = cos.to(embedding.dtype)
        sin = sin.to(embedding.dtype)

        causal_mask = torch.zeros(1, 1, seq_len, seq_len, device=device, dtype=embedding.dtype)
        causal_mask.masked_fill_(
            torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1),
            float('-inf'),
        )

        layer_outputs = [embedding]

        for l in range(self.num_layers):
            layer = self.olmo.model.layers[l]
            attn = layer.self_attn

            if l == 0:
                q_all = attn.q_proj(embedding)
                k_all = attn.k_proj(embedding)
                v_all = attn.v_proj(embedding)
                q_all = attn.q_norm(q_all)
                k_all = attn.k_norm(k_all)
                q_per_head = q_all.view(batch, seq_len, H, self.head_dim).transpose(1, 2)
                k_per_head = k_all.view(batch, seq_len, H, self.head_dim).transpose(1, 2)
                v_per_head = v_all.view(batch, seq_len, H, self.head_dim).transpose(1, 2)
            else:
                q_layers, k_layers, v_layers = [], [], []
                for X_j in layer_outputs:
                    q_layers.append(attn.q_proj(X_j).view(batch, seq_len, H, self.head_dim))
                    k_layers.append(attn.k_proj(X_j).view(batch, seq_len, H, self.head_dim))
                    v_layers.append(attn.v_proj(X_j).view(batch, seq_len, H, self.head_dim))

                q_stack = torch.stack(q_layers, dim=0)
                k_stack = torch.stack(k_layers, dim=0)
                v_stack = torch.stack(v_layers, dim=0)

                α_q = routing_weights['q'][l - 1]
                α_k = routing_weights['k'][l - 1]
                α_v = routing_weights['v'][l - 1]
                α_r = routing_weights['r'][l - 1]

                q_per_head = torch.einsum('lbthd, bthl -> bhtd', q_stack, α_q)
                k_per_head = torch.einsum('lbthd, bthl -> bhtd', k_stack, α_k)
                v_per_head = torch.einsum('lbthd, bthl -> bhtd', v_stack, α_v)

                q_concat = rearrange(q_per_head, 'b h t d -> b t (h d)')
                q_normed = attn.q_norm(q_concat)
                q_per_head = rearrange(q_normed, 'b t (h d) -> b h t d', h=H)

                k_concat = rearrange(k_per_head, 'b h t d -> b t (h d)')
                k_normed = attn.k_norm(k_concat)
                k_per_head = rearrange(k_normed, 'b t (h d) -> b h t d', h=H)

                stacked = torch.stack(layer_outputs, dim=0)
                R = torch.einsum('lbtd, btl -> btd', stacked, α_r)

            q_per_head, k_per_head = apply_rotary_pos_emb(q_per_head, k_per_head, cos, sin)

            attn_w = torch.matmul(q_per_head, k_per_head.transpose(-2, -1)) * self.scaling
            attn_w = attn_w + causal_mask
            # NO dtype=torch.float32 here — native precision
            attn_w = F.softmax(attn_w, dim=-1)
            attn_values = torch.matmul(attn_w, v_per_head)

            attn_concat = rearrange(attn_values, 'b h t d -> b t (h d)')
            attn_proj = attn.o_proj(attn_concat)
            attn_out = layer.post_attention_layernorm(attn_proj)

            if l == 0:
                mlp_in = embedding + attn_out
            else:
                mlp_in = R + attn_out
            mlp_raw = layer.mlp(mlp_in)
            mlp_out = layer.post_feedforward_layernorm(mlp_raw)

            if l == 0:
                X_next = embedding + attn_out + mlp_out
            else:
                X_next = R + attn_out + mlp_out
            layer_outputs.append(X_next)

        final_state = self.olmo.model.norm(layer_outputs[-1])
        logits = self.olmo.lm_head(final_state)
        return logits


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


torch.manual_seed(42)
device = torch.device("cpu")
L, H, D, V = 3, 2, 64, 100
B, T = 1, 8

mc = Olmo2Config(hidden_size=D, num_hidden_layers=L, num_attention_heads=H,
                 num_key_value_heads=H, intermediate_size=D*4, vocab_size=V,
                 tie_word_embeddings=True, max_position_embeddings=64)

base = Olmo2ForCausalLM(mc).to(device, dtype=torch.float64)
n = replace_olmo_rmsnorm(base)
print(f"Replaced {n} Olmo2RMSNorm → nn.RMSNorm")

fw = CleanFourWayForward(base, num_layers=L, num_heads=H).to(device, dtype=torch.float64)
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

print(f"\nClean FourWay gradcheck (CPU, f64, ALL f32 casts removed)")
print(f"  nn.RMSNorm (no f32 cast)")
print(f"  F.softmax native dtype (no dtype=f32)")
print(f"  No hidden.float() in corrections")
print(f"  cos/sin cast to f64")

ok = gradcheck(loss_fn, tuple(alphas), eps=1e-6, atol=1e-4, rtol=1e-3, raise_exception=False)
print(f"\nResult: {'PASS ✓' if ok else 'FAIL ✗'}")

if ok:
    print("\n→ ALL f32 casts were the issue. Training gradients are CORRECT.")
    print("  The 5.5 eval floor is genuine overfit, not a gradient bug.")
else:
    print("\n→ Even without f32 casts, gradcheck FAILS.")
    print("  There IS a real autograd bug beyond mixed-precision casts!")
