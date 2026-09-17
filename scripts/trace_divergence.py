"""Trace exactly WHERE and WHY FourWay diverges from HF in bf16.

Instruments both forward paths layer by layer to find the first divergence point.
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


def compare(name, a, b, atol=1e-4):
    diff = (a.float() - b.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    status = "✓" if max_diff < atol else "✗"
    print(f"  {status} {name:40s}  max_diff={max_diff:.6e}  mean={mean_diff:.6e}")
    return max_diff


def main():
    device = "cuda"
    dtype = torch.bfloat16
    B, T = 2, 128
    num_layers = 12
    H = 16
    head_dim = 64

    cfg = Olmo2Config(
        hidden_size=1024, num_hidden_layers=num_layers, num_attention_heads=H,
        intermediate_size=4096, vocab_size=100352, tie_word_embeddings=True,
        max_position_embeddings=4096,
    )
    base = Olmo2ForCausalLM(cfg).to(device, dtype=dtype)
    replace_rmsnorm(base)
    base.eval()

    input_ids = torch.randint(0, 1000, (B, T), device=device)

    with torch.no_grad():
        # ====== HF forward (step by step) ======
        hf_hidden = base.model.embed_tokens(input_ids)
        position_ids = torch.arange(T, device=device).unsqueeze(0)
        position_embeddings = base.model.rotary_emb(hf_hidden, position_ids)

        hf_layers = [hf_hidden.clone()]
        for l in range(num_layers):
            hf_hidden = base.model.layers[l](
                hf_hidden,
                position_embeddings=position_embeddings,
                attention_mask=None,
            )
            hf_layers.append(hf_hidden.clone())

        hf_final = base.model.norm(hf_hidden)
        hf_logits = base.lm_head(hf_final)

        # ====== FourWay forward (step by step, matching olmo_graph.py) ======
        fw_embedding = base.model.embed_tokens(input_ids)
        fw_pos_ids = torch.arange(T, device=device).unsqueeze(0)
        fw_cos, fw_sin = base.model.rotary_emb(fw_embedding, fw_pos_ids)

        print("=== Embedding ===")
        compare("embedding", fw_embedding, hf_layers[0])
        compare("cos", fw_cos, position_embeddings[0])
        compare("sin", fw_sin, position_embeddings[1])

        layer_outputs = [fw_embedding]
        scaling = head_dim ** -0.5

        for l in range(num_layers):
            attn = base.model.layers[l].self_attn
            layer = base.model.layers[l]

            if l == 0:
                # Layer 0: standard projection
                q_all = attn.q_proj(fw_embedding)
                k_all = attn.k_proj(fw_embedding)
                v_all = attn.v_proj(fw_embedding)
                q_all = attn.q_norm(q_all)
                k_all = attn.k_norm(k_all)
                q_ph = q_all.view(B, T, H, head_dim).transpose(1, 2)
                k_ph = k_all.view(B, T, H, head_dim).transpose(1, 2)
                v_ph = v_all.view(B, T, H, head_dim).transpose(1, 2)
            else:
                # Layer l: identity routing (select only most recent output)
                X_prev = layer_outputs[-1]
                q_all = attn.q_proj(X_prev)
                k_all = attn.k_proj(X_prev)
                v_all = attn.v_proj(X_prev)
                q_all = attn.q_norm(q_all)
                k_all = attn.k_norm(k_all)
                q_ph = q_all.view(B, T, H, head_dim).transpose(1, 2)
                k_ph = k_all.view(B, T, H, head_dim).transpose(1, 2)
                v_ph = v_all.view(B, T, H, head_dim).transpose(1, 2)

            # RoPE
            q_ph, k_ph = apply_rotary_pos_emb(q_ph, k_ph, fw_cos, fw_sin)

            # SDPA
            attn_values = F.scaled_dot_product_attention(
                q_ph, k_ph, v_ph,
                attn_mask=None, dropout_p=0.0, is_causal=True, scale=scaling,
            )

            # O projection
            attn_concat = rearrange(attn_values, 'b h t d -> b t (h d)')
            attn_proj = attn.o_proj(attn_concat)

            # Post-attention norm
            attn_out = layer.post_attention_layernorm(attn_proj)

            # Residual + MLP
            if l == 0:
                mlp_in = fw_embedding + attn_out
            else:
                mlp_in = layer_outputs[-1] + attn_out  # R=identity=last layer output

            mlp_raw = layer.mlp(mlp_in)
            mlp_out = layer.post_feedforward_layernorm(mlp_raw)

            if l == 0:
                X_next = fw_embedding + attn_out + mlp_out
            else:
                X_next = layer_outputs[-1] + attn_out + mlp_out

            layer_outputs.append(X_next)

            print(f"\n=== Layer {l} ===")
            compare(f"L{l} output", X_next, hf_layers[l + 1])

        fw_final = base.model.norm(layer_outputs[-1])
        fw_logits = base.lm_head(fw_final)

        print(f"\n=== Final ===")
        compare("final_norm", fw_final, hf_final)
        compare("logits", fw_logits, hf_logits)

        # Now check: what does HF's attention_mask look like?
        print(f"\n=== HF Attention Config ===")
        print(f"  _attn_implementation: {base.config._attn_implementation}")

        # Compare with the ACTUAL FourWay code (using einsum path)
        print(f"\n=== Now testing ACTUAL FourWay einsum path ===")
        from src.model.olmo_graph import FourWayDAGFormer
        fourway = FourWayDAGFormer(
            model=base, num_layers=num_layers, num_heads=H,
            use_local_correction=False,
        )
        fourway.eval()

        rw = {'q': [], 'k': [], 'v': [], 'r': []}
        for ll in range(1, num_layers):
            n_src = ll + 1
            for stream in ('q', 'k', 'v'):
                α = torch.zeros(B, T, H, n_src, device=device, dtype=dtype)
                α[..., -1] = 1.0
                rw[stream].append(α)
            α_r = torch.zeros(B, T, n_src, device=device, dtype=dtype)
            α_r[..., -1] = 1.0
            rw['r'].append(α_r)

        logits_actual = fourway(input_ids, rw)
        compare("FourWay(actual) vs HF", logits_actual, hf_logits)
        compare("FourWay(actual) vs manual", logits_actual, fw_logits)


if __name__ == "__main__":
    main()
