"""Quick latency benchmark: original vs batched QKV projection.

Compares forward+backward time of FourWayDAGFormer with:
  - Original: per-source loop (L separate matmuls per stream)
  - Batched: stack + single matmul per stream

Usage:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python scripts/bench_forward_backward.py
"""
import time
import torch
import torch.nn as nn
from transformers import Olmo2Config, Olmo2ForCausalLM
from einops import rearrange
import torch.nn.functional as F


def make_model_and_inputs(device="cuda"):
    cfg = Olmo2Config(
        hidden_size=1024,
        num_hidden_layers=12,
        num_attention_heads=16,
        intermediate_size=4096,
        vocab_size=100352,
        tie_word_embeddings=True,
        max_position_embeddings=4096,
    )
    base = Olmo2ForCausalLM(cfg).to(device, dtype=torch.bfloat16)

    # Replace Olmo2RMSNorm with nn.RMSNorm
    for name, module in list(base.named_modules()):
        if type(module).__name__ == "Olmo2RMSNorm":
            new_norm = nn.RMSNorm(module.weight.shape[0], eps=module.variance_epsilon).to(
                device=module.weight.device, dtype=module.weight.dtype
            )
            new_norm.weight = module.weight
            parts = name.split(".")
            parent = base
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_norm)

    B, T = 4, 1024
    input_ids = torch.randint(0, 100352, (B, T), device=device)

    # Build fake routing weights (identity init)
    num_layers = 12
    H = 16
    rw = {'q': [], 'k': [], 'v': [], 'r': []}
    for l in range(1, num_layers):
        n_src = l + 1
        for stream in ('q', 'k', 'v'):
            α = torch.zeros(B, T, H, n_src, device=device, dtype=torch.bfloat16)
            α[..., -1] = 1.0
            rw[stream].append(α)
        α_r = torch.zeros(B, T, n_src, device=device, dtype=torch.bfloat16)
        α_r[..., -1] = 1.0
        rw['r'].append(α_r)

    return base, input_ids, rw


def forward_original(base, input_ids, routing_weights):
    """Original: per-source loop."""
    batch, seq_len = input_ids.shape
    H = 16
    head_dim = 64
    model_dim = 1024
    scaling = head_dim ** -0.5

    embedding = base.model.embed_tokens(input_ids)
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    cos, sin = base.model.rotary_emb(embedding, position_ids)

    from transformers.models.olmo2.modeling_olmo2 import apply_rotary_pos_emb

    layer_outputs = [embedding]

    for l in range(12):
        attn = base.model.layers[l].self_attn
        layer = base.model.layers[l]

        if l == 0:
            q_all = attn.q_norm(attn.q_proj(embedding))
            k_all = attn.k_norm(attn.k_proj(embedding))
            v_all = attn.v_proj(embedding)
            q_ph = q_all.view(batch, seq_len, H, head_dim).transpose(1, 2)
            k_ph = k_all.view(batch, seq_len, H, head_dim).transpose(1, 2)
            v_ph = v_all.view(batch, seq_len, H, head_dim).transpose(1, 2)
        else:
            # ORIGINAL: per-source loop
            q_layers, k_layers, v_layers = [], [], []
            for X_j in layer_outputs:
                q_layers.append(attn.q_proj(X_j).view(batch, seq_len, H, head_dim))
                k_layers.append(attn.k_proj(X_j).view(batch, seq_len, H, head_dim))
                v_layers.append(attn.v_proj(X_j).view(batch, seq_len, H, head_dim))
            q_stack = torch.stack(q_layers, dim=0)
            k_stack = torch.stack(k_layers, dim=0)
            v_stack = torch.stack(v_layers, dim=0)

            α_q = routing_weights['q'][l - 1]
            α_k = routing_weights['k'][l - 1]
            α_v = routing_weights['v'][l - 1]
            α_r = routing_weights['r'][l - 1]

            q_ph = torch.einsum('lbthd, bthl -> bhtd', q_stack, α_q)
            k_ph = torch.einsum('lbthd, bthl -> bhtd', k_stack, α_k)
            v_ph = torch.einsum('lbthd, bthl -> bhtd', v_stack, α_v)

            q_ph = rearrange(attn.q_norm(rearrange(q_ph, 'b h t d -> b t (h d)')), 'b t (h d) -> b h t d', h=H)
            k_ph = rearrange(attn.k_norm(rearrange(k_ph, 'b h t d -> b t (h d)')), 'b t (h d) -> b h t d', h=H)

            stacked = torch.stack(layer_outputs, dim=0)
            R = torch.einsum('lbtd, btl -> btd', stacked, α_r)

        q_ph, k_ph = apply_rotary_pos_emb(q_ph, k_ph, cos, sin)
        attn_out_raw = F.scaled_dot_product_attention(q_ph, k_ph, v_ph, is_causal=True, scale=scaling)
        attn_proj = attn.o_proj(rearrange(attn_out_raw, 'b h t d -> b t (h d)'))
        attn_out = layer.post_attention_layernorm(attn_proj)

        res = embedding if l == 0 else R
        mlp_out = layer.post_feedforward_layernorm(layer.mlp(res + attn_out))
        X_next = res + attn_out + mlp_out
        layer_outputs.append(X_next)

    final = base.model.norm(layer_outputs[-1])
    logits = base.lm_head(final)
    return logits


def forward_batched(base, input_ids, routing_weights):
    """Batched: stack + single matmul per stream."""
    batch, seq_len = input_ids.shape
    H = 16
    head_dim = 64
    model_dim = 1024
    scaling = head_dim ** -0.5

    embedding = base.model.embed_tokens(input_ids)
    position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
    cos, sin = base.model.rotary_emb(embedding, position_ids)

    from transformers.models.olmo2.modeling_olmo2 import apply_rotary_pos_emb

    layer_outputs = [embedding]

    for l in range(12):
        attn = base.model.layers[l].self_attn
        layer = base.model.layers[l]

        if l == 0:
            q_all = attn.q_norm(attn.q_proj(embedding))
            k_all = attn.k_norm(attn.k_proj(embedding))
            v_all = attn.v_proj(embedding)
            q_ph = q_all.view(batch, seq_len, H, head_dim).transpose(1, 2)
            k_ph = k_all.view(batch, seq_len, H, head_dim).transpose(1, 2)
            v_ph = v_all.view(batch, seq_len, H, head_dim).transpose(1, 2)
        else:
            # BATCHED: single matmul per stream
            stacked_sources = torch.stack(layer_outputs, dim=0)
            L_cur = stacked_sources.shape[0]
            flat = stacked_sources.reshape(L_cur * batch * seq_len, model_dim)
            q_stack = attn.q_proj(flat).view(L_cur, batch, seq_len, H, head_dim)
            k_stack = attn.k_proj(flat).view(L_cur, batch, seq_len, H, head_dim)
            v_stack = attn.v_proj(flat).view(L_cur, batch, seq_len, H, head_dim)

            α_q = routing_weights['q'][l - 1]
            α_k = routing_weights['k'][l - 1]
            α_v = routing_weights['v'][l - 1]
            α_r = routing_weights['r'][l - 1]

            q_ph = torch.einsum('lbthd, bthl -> bhtd', q_stack, α_q)
            k_ph = torch.einsum('lbthd, bthl -> bhtd', k_stack, α_k)
            v_ph = torch.einsum('lbthd, bthl -> bhtd', v_stack, α_v)

            q_ph = rearrange(attn.q_norm(rearrange(q_ph, 'b h t d -> b t (h d)')), 'b t (h d) -> b h t d', h=H)
            k_ph = rearrange(attn.k_norm(rearrange(k_ph, 'b h t d -> b t (h d)')), 'b t (h d) -> b h t d', h=H)

            R = torch.einsum('lbtd, btl -> btd', stacked_sources, α_r)

        q_ph, k_ph = apply_rotary_pos_emb(q_ph, k_ph, cos, sin)
        attn_out_raw = F.scaled_dot_product_attention(q_ph, k_ph, v_ph, is_causal=True, scale=scaling)
        attn_proj = attn.o_proj(rearrange(attn_out_raw, 'b h t d -> b t (h d)'))
        attn_out = layer.post_attention_layernorm(attn_proj)

        res = embedding if l == 0 else R
        mlp_out = layer.post_feedforward_layernorm(layer.mlp(res + attn_out))
        X_next = res + attn_out + mlp_out
        layer_outputs.append(X_next)

    final = base.model.norm(layer_outputs[-1])
    logits = base.lm_head(final)
    return logits


def bench(fn, base, input_ids, rw, label, warmup=3, iters=10):
    # Warmup
    for _ in range(warmup):
        logits = fn(base, input_ids, rw)
        loss = logits.sum()
        loss.backward()
        base.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = fn(base, input_ids, rw)
        loss = logits.sum()
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        base.zero_grad(set_to_none=True)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    print(f"{label:20s}: {avg*1000:.1f} ± {std*1000:.1f} ms  (min={min(times)*1000:.1f}, max={max(times)*1000:.1f})")
    return avg


if __name__ == "__main__":
    print("Setting up model (B=4, T=1024, 12 layers, 16 heads)...")
    base, input_ids, rw = make_model_and_inputs("cuda")

    # Make routing weights require grad so backward traces through them
    for stream in ('q', 'k', 'v', 'r'):
        for i in range(len(rw[stream])):
            rw[stream][i] = rw[stream][i].requires_grad_(True)

    print()
    t_orig = bench(forward_original, base, input_ids, rw, "Original (loop)")
    t_batch = bench(forward_batched, base, input_ids, rw, "Batched (stack)")
    print()
    print(f"Speedup: {t_orig / t_batch:.2f}x")
