"""Pinpoint EXACTLY where FourWayDAGFormer diverges from HF.

Strategy: instrument the actual FourWay forward with hooks to capture
intermediate values and compare against HF's forward.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Olmo2Config, Olmo2ForCausalLM
from transformers.models.olmo2.modeling_olmo2 import apply_rotary_pos_emb
from einops import rearrange
import sys
sys.path.insert(0, '.')


def replace_rmsnorm(model):
    for name, module in list(model.named_modules()):
        if type(module).__name__ == "Olmo2RMSNorm":
            new_norm = nn.RMSNorm(module.weight.shape[0], eps=module.variance_epsilon).to(
                device=module.weight.device, dtype=module.weight.dtype
            )
            new_norm.weight = module.weight
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_norm)


def compare(name, a, b):
    diff = (a.float() - b.float()).abs()
    max_diff = diff.max().item()
    status = "✓" if max_diff < 1e-6 else "✗"
    if max_diff > 0:
        print(f"  {status} {name:45s}  max_diff={max_diff:.6e}  shape={list(a.shape)}")
    return max_diff


def main():
    device = "cuda"
    dtype = torch.bfloat16
    B, T = 2, 128
    num_layers = 12
    H = 16
    head_dim = 64
    D = 1024
    scaling = head_dim ** -0.5

    cfg = Olmo2Config(
        hidden_size=D, num_hidden_layers=num_layers, num_attention_heads=H,
        intermediate_size=4096, vocab_size=100352, tie_word_embeddings=True,
        max_position_embeddings=4096,
    )
    base = Olmo2ForCausalLM(cfg).to(device, dtype=dtype)
    replace_rmsnorm(base)
    base.eval()

    input_ids = torch.randint(0, 1000, (B, T), device=device)

    # Build identity routing weights
    rw = {'q': [], 'k': [], 'v': [], 'r': []}
    for l in range(1, num_layers):
        n_src = l + 1
        for stream in ('q', 'k', 'v'):
            a = torch.zeros(B, T, H, n_src, device=device, dtype=dtype)
            a[..., -1] = 1.0
            rw[stream].append(a)
        a_r = torch.zeros(B, T, n_src, device=device, dtype=dtype)
        a_r[..., -1] = 1.0
        rw['r'].append(a_r)

    with torch.no_grad():
        # ====== HF forward (layer by layer) ======
        embedding = base.model.embed_tokens(input_ids)
        position_ids = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = base.model.rotary_emb(embedding, position_ids)
        cos_hf, sin_hf = pos_emb

        hf_hidden = embedding.clone()
        hf_layers = [embedding.clone()]
        for l in range(num_layers):
            hf_hidden = base.model.layers[l](hf_hidden, position_embeddings=pos_emb, attention_mask=None)
            hf_layers.append(hf_hidden.clone())

        # ====== Replicate ACTUAL FourWay code path (from olmo_graph.py) ======
        fw_embedding = base.model.embed_tokens(input_ids)
        fw_pos_ids = torch.arange(T, device=device).unsqueeze(0)
        fw_cos, fw_sin = base.model.rotary_emb(fw_embedding, fw_pos_ids)

        layer_outputs = [fw_embedding]

        for l in range(num_layers):
            attn = base.model.layers[l].self_attn
            layer_mod = base.model.layers[l]

            if l == 0:
                # Layer 0: standard (same in both paths)
                q_all = attn.q_proj(fw_embedding)
                k_all = attn.k_proj(fw_embedding)
                v_all = attn.v_proj(fw_embedding)
                q_all = attn.q_norm(q_all)
                k_all = attn.k_norm(k_all)
                q_ph = q_all.view(B, T, H, head_dim).transpose(1, 2)
                k_ph = k_all.view(B, T, H, head_dim).transpose(1, 2)
                v_ph = v_all.view(B, T, H, head_dim).transpose(1, 2)
            else:
                # Layer l: replicate ACTUAL FourWay code with einsum
                # Stack all layer outputs
                stacked_sources = torch.stack(layer_outputs, dim=0)  # [L, B, T, D]
                L_cur = stacked_sources.shape[0]
                flat = stacked_sources.reshape(L_cur * B * T, D)

                q_stack = attn.q_proj(flat).view(L_cur, B, T, H, head_dim)
                k_stack = attn.k_proj(flat).view(L_cur, B, T, H, head_dim)
                v_stack = attn.v_proj(flat).view(L_cur, B, T, H, head_dim)

                alpha_q = rw['q'][l-1]
                alpha_k = rw['k'][l-1]
                alpha_v = rw['v'][l-1]
                alpha_r = rw['r'][l-1]

                # === TEST: Is einsum exact with identity? ===
                q_via_einsum = torch.einsum('lbthd, bthl -> bhtd', q_stack, alpha_q)
                q_direct = q_stack[-1].permute(0, 2, 1, 3)  # [B,T,H,hd] -> [B,H,T,hd]
                compare(f"L{l} einsum vs direct select (Q)", q_via_einsum, q_direct)

                q_ph = q_via_einsum
                k_ph = torch.einsum('lbthd, bthl -> bhtd', k_stack, alpha_k)
                v_ph = torch.einsum('lbthd, bthl -> bhtd', v_stack, alpha_v)

                # Q/K norm
                q_concat = rearrange(q_ph, 'b h t d -> b t (h d)')
                q_normed = attn.q_norm(q_concat)
                q_ph = rearrange(q_normed, 'b t (h d) -> b h t d', h=H)

                k_concat = rearrange(k_ph, 'b h t d -> b t (h d)')
                k_normed = attn.k_norm(k_concat)
                k_ph = rearrange(k_normed, 'b t (h d) -> b h t d', h=H)

                # === TEST: Compare Q after norm vs direct ===
                # Direct path: q_proj on last layer output, then norm
                q_direct_all = attn.q_proj(layer_outputs[-1])  # [B,T,H*hd]
                q_direct_normed = attn.q_norm(q_direct_all)    # [B,T,H*hd]
                q_direct_ph = q_direct_normed.view(B, T, H, head_dim).transpose(1, 2)
                compare(f"L{l} Q (einsum+norm) vs (direct+norm)", q_ph, q_direct_ph)

                # R stream
                R = torch.einsum('lbtd, btl -> btd', stacked_sources, alpha_r)
                R_direct = layer_outputs[-1]
                compare(f"L{l} R einsum vs direct", R, R_direct)

            # RoPE
            q_ph, k_ph = apply_rotary_pos_emb(q_ph, k_ph, fw_cos, fw_sin)

            # SDPA
            attn_values = F.scaled_dot_product_attention(
                q_ph, k_ph, v_ph,
                attn_mask=None, dropout_p=0.0, is_causal=True, scale=scaling,
            )

            # O proj + post-attn norm
            attn_concat = rearrange(attn_values, 'b h t d -> b t (h d)')
            attn_proj = attn.o_proj(attn_concat)
            attn_out = layer_mod.post_attention_layernorm(attn_proj)

            # Residual + MLP
            if l == 0:
                mlp_in = fw_embedding + attn_out
            else:
                mlp_in = R + attn_out

            mlp_raw = layer_mod.mlp(mlp_in)
            mlp_out = layer_mod.post_feedforward_layernorm(mlp_raw)

            if l == 0:
                X_next = fw_embedding + attn_out + mlp_out
            else:
                X_next = R + attn_out + mlp_out

            layer_outputs.append(X_next)

            print(f"  >>> L{l} output vs HF:  max_diff={compare(f'L{l} LAYER OUTPUT vs HF', X_next, hf_layers[l+1]):.6e}")


if __name__ == "__main__":
    main()
