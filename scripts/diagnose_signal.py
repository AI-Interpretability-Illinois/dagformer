"""Diagnose signal amplification: hidden state norms, routing magnitudes, per-layer gain.

Usage:
    python scripts/diagnose_signal.py \
        --checkpoint checkpoints/fourway_reg_l2decay/checkpoint_step4000.pt \
        --config configs/fourway_reg_l2decay.yaml
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.pretrain_dagformer import (
    DAGFormerPretrainConfig, create_model,
)
from src.model.predictor import FourWayPredictor
from src.model.olmo_graph import FourWayDAGFormer
from src.data.dolma import build_eval_dataloader


def load_model(config, checkpoint_path, device):
    base_model = create_model(config).to(device, dtype=torch.bfloat16)
    predictor = FourWayPredictor(
        vocab_size=config.vocab_size,
        encoder_dim=config.predictor_encoder_dim,
        encoder_layers=config.predictor_encoder_layers,
        encoder_heads=config.predictor_encoder_heads,
        max_seq_len=config.predictor_max_seq_len,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        hidden_dim=config.fourway_hidden,
        causal=config.predictor_causal,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'predictor_state_dict' in ckpt:
        predictor.load_state_dict(ckpt['predictor_state_dict'])
    if 'model_state_path' in ckpt:
        model_state = torch.load(ckpt['model_state_path'], map_location=device)
        base_model.load_state_dict(model_state)
        del model_state

    predictor.eval()
    base_model.eval()
    return base_model, predictor


def diagnose(base_model, predictor, eval_batch, config, device):
    input_ids = eval_batch["olmo_ids"].to(device)
    B, T = input_ids.shape
    H = config.num_attention_heads

    # Get routing weights
    with torch.no_grad():
        rw = predictor(input_ids)

    # === 1. Routing weight magnitudes ===
    print("\n=== Routing Weight Magnitudes ===")
    print(f"{'Layer':<8} {'Stream':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'|Max|':>8}")
    for stream in ('q', 'k', 'v', 'r'):
        for l, alpha in enumerate(rw[stream]):
            a = alpha.float()
            print(f"L{l+1:<6} {stream:<8} {a.mean().item():>8.3f} {a.std().item():>8.3f} "
                  f"{a.min().item():>8.3f} {a.max().item():>8.3f} {a.abs().max().item():>8.3f}")

    # === 2. Hidden state norms across layers (with routing) ===
    print("\n=== Hidden State Norms (with routing) ===")
    fourway = FourWayDAGFormer(
        model=base_model,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
    ).to(device)
    fourway.eval()

    # Hook into layer_outputs during forward
    # We'll manually run the forward and collect norms
    embedding = base_model.model.embed_tokens(input_ids)
    layer_outputs = [embedding]

    position_ids = torch.arange(T, device=device).unsqueeze(0)
    cos, sin = base_model.model.rotary_emb(embedding, position_ids)

    from einops import rearrange
    from transformers.models.olmo2.modeling_olmo2 import apply_rotary_pos_emb

    causal_mask = torch.zeros(1, 1, T, T, device=device, dtype=embedding.dtype)
    causal_mask.masked_fill_(
        torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1),
        float('-inf'),
    )

    norms_routing = [embedding.float().norm(dim=-1).mean().item()]

    with torch.no_grad():
        for l in range(config.num_hidden_layers):
            attn = base_model.model.layers[l].self_attn
            weights = {
                'post_attn_norm': base_model.model.layers[l].post_attention_layernorm,
                'post_ff_norm': base_model.model.layers[l].post_feedforward_layernorm,
                'mlp': base_model.model.layers[l].mlp,
            }

            if l == 0:
                q_all = attn.q_proj(embedding)
                k_all = attn.k_proj(embedding)
                v_all = attn.v_proj(embedding)
                q_all = attn.q_norm(q_all)
                k_all = attn.k_norm(k_all)
                q_ph = q_all.view(B, T, H, -1).transpose(1, 2)
                k_ph = k_all.view(B, T, H, -1).transpose(1, 2)
                v_ph = v_all.view(B, T, H, -1).transpose(1, 2)
                R = embedding
            else:
                alpha_q = rw['q'][l-1].to(dtype=embedding.dtype)
                alpha_k = rw['k'][l-1].to(dtype=embedding.dtype)
                alpha_v = rw['v'][l-1].to(dtype=embedding.dtype)
                alpha_r = rw['r'][l-1].to(dtype=embedding.dtype)

                q_layers, k_layers, v_layers = [], [], []
                for X_j in layer_outputs:
                    q_layers.append(attn.q_proj(X_j).view(B, T, H, -1))
                    k_layers.append(attn.k_proj(X_j).view(B, T, H, -1))
                    v_layers.append(attn.v_proj(X_j).view(B, T, H, -1))

                q_stack = torch.stack(q_layers, dim=0)
                k_stack = torch.stack(k_layers, dim=0)
                v_stack = torch.stack(v_layers, dim=0)

                q_ph = torch.einsum('lbthd, bthl -> bhtd', q_stack, alpha_q)
                k_ph = torch.einsum('lbthd, bthl -> bhtd', k_stack, alpha_k)
                v_ph = torch.einsum('lbthd, bthl -> bhtd', v_stack, alpha_v)

                q_concat = rearrange(q_ph, 'b h t d -> b t (h d)')
                q_ph = rearrange(attn.q_norm(q_concat), 'b t (h d) -> b h t d', h=H)
                k_concat = rearrange(k_ph, 'b h t d -> b t (h d)')
                k_ph = rearrange(attn.k_norm(k_concat), 'b t (h d) -> b h t d', h=H)

                stacked = torch.stack(layer_outputs, dim=0)
                R = torch.einsum('lbtd, btl -> btd', stacked, alpha_r)

            q_ph, k_ph = apply_rotary_pos_emb(q_ph, k_ph, cos, sin)
            hd = q_ph.shape[-1]
            attn_w = torch.matmul(q_ph, k_ph.transpose(-2, -1)) * (hd ** -0.5)
            attn_w = attn_w + causal_mask
            attn_w = F.softmax(attn_w, dim=-1, dtype=torch.float32).to(q_ph.dtype)
            attn_values = torch.matmul(attn_w, v_ph)

            attn_concat = rearrange(attn_values, 'b h t d -> b t (h d)')
            attn_proj = attn.o_proj(attn_concat)
            attn_out = weights['post_attn_norm'](attn_proj)

            mlp_in = R + attn_out
            mlp_raw = weights['mlp'](mlp_in)
            mlp_out = weights['post_ff_norm'](mlp_raw)
            X_next = R + attn_out + mlp_out
            layer_outputs.append(X_next)

            norms_routing.append(X_next.float().norm(dim=-1).mean().item())

    # === 3. Hidden state norms (dense baseline, same weights) ===
    print("\n=== Hidden State Norms (dense forward, same weights) ===")
    norms_dense = [embedding.float().norm(dim=-1).mean().item()]
    with torch.no_grad():
        x = embedding
        for l in range(config.num_hidden_layers):
            layer_mod = base_model.model.layers[l]
            residual = x
            attn_out_d = layer_mod.self_attn(x)[0] if hasattr(layer_mod.self_attn, '__call__') else x
            # Use the layer's own forward
            x_out = layer_mod(x, position_embeddings=(cos, sin))
            x = x_out
            norms_dense.append(x.float().norm(dim=-1).mean().item())

    print(f"{'Layer':<8} {'Routing ||X||':>14} {'Dense ||X||':>14} {'Gain (R)':>10} {'Gain (D)':>10}")
    for l in range(config.num_hidden_layers + 1):
        gain_r = norms_routing[l] / norms_routing[max(0, l-1)] if l > 0 else 1.0
        gain_d = norms_dense[l] / norms_dense[max(0, l-1)] if l > 0 else 1.0
        label = f"emb" if l == 0 else f"L{l-1}"
        print(f"{label:<8} {norms_routing[l]:>14.2f} {norms_dense[l]:>14.2f} {gain_r:>10.3f} {gain_d:>10.3f}")

    # Summary
    total_gain_r = norms_routing[-1] / norms_routing[0]
    total_gain_d = norms_dense[-1] / norms_dense[0]
    print(f"\nTotal gain (emb → final): routing={total_gain_r:.2f}x, dense={total_gain_d:.2f}x")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = DAGFormerPretrainConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading from {args.checkpoint}...")
    base_model, predictor = load_model(config, args.checkpoint, device)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)
    eval_batches = build_eval_dataloader(
        olmo_tokenizer=tokenizer, seq_len=config.seq_len,
        batch_size=config.micro_batch_size,
        dataset_name=config.dataset, dataset_version=config.dataset_name,
        eval_skip=config.eval_skip, eval_size=5,
        cache_path=os.path.join(config.save_dir, "eval_cache.pt"),
    )

    print("Running diagnosis on first eval batch...")
    diagnose(base_model, predictor, eval_batches[0], config, device)
    print("\nDone.")


if __name__ == "__main__":
    main()
