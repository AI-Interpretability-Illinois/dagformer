"""Routing analysis + identity baseline comparison.

Usage:
    python scripts/diagnose_routing_v2.py \
        --checkpoint checkpoints/fourway_corrected_joint_causal/checkpoint_step5000.pt \
        --config configs/fourway_corrected_joint_causal.yaml
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.pretrain_dagformer import DAGFormerPretrainConfig, create_model
from src.model.predictor import FourWayPredictor
from src.model.olmo_graph import FourWayDAGFormer
from src.data.dolma import build_eval_dataloader


def load_model(config, checkpoint_path, device):
    base_model = create_model(config).to(device, dtype=torch.bfloat16)
    use_correction = (config.routing_mode == "fourway_corrected")
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
    fourway = FourWayDAGFormer(
        model=base_model,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
        use_local_correction=use_correction,
        correction_hidden=config.correction_hidden if use_correction else 128,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'predictor_state_dict' in ckpt:
        predictor.load_state_dict(ckpt['predictor_state_dict'])
    if 'model_state_path' in ckpt:
        model_state = torch.load(ckpt['model_state_path'], map_location=device)
        base_model.load_state_dict(model_state)
        del model_state
    if 'routing_state_dict' in ckpt and use_correction:
        fourway.load_state_dict(ckpt['routing_state_dict'], strict=False)

    predictor.eval()
    fourway.eval()
    base_model.eval()
    return base_model, predictor, fourway


def analyze_routing(predictor, eval_batches, device, num_layers, num_heads):
    """Analyze routing weight distribution on eval data."""
    print("\n" + "=" * 60)
    print("1. ROUTING WEIGHT ANALYSIS (on eval data)")
    print("=" * 60)

    all_rw = {'q': [], 'k': [], 'v': [], 'r': []}

    with torch.no_grad():
        for eb in eval_batches[:10]:
            eids = eb["olmo_ids"].to(device)
            rw = predictor(eids)
            for stream in ('q', 'k', 'v', 'r'):
                for l, alpha in enumerate(rw[stream]):
                    while len(all_rw[stream]) <= l:
                        all_rw[stream].append([])
                    all_rw[stream][l].append(alpha.float().cpu())

    # Aggregate
    for stream in ('q', 'k', 'v', 'r'):
        print(f"\n--- Stream: {stream.upper()} ---")
        for l in range(len(all_rw[stream])):
            cat = torch.cat(all_rw[stream][l], dim=0)  # [total_B, T, ...]
            n_src = cat.shape[-1]

            # Mean weight per source layer
            if stream in ('q', 'k', 'v'):
                # [B, T, H, n_src] → mean over B, T, H
                mean_weights = cat.mean(dim=(0, 1, 2))
            else:
                # [B, T, n_src] → mean over B, T
                mean_weights = cat.mean(dim=(0, 1))

            # Softmax version for entropy
            p = F.softmax(cat, dim=-1)
            if stream in ('q', 'k', 'v'):
                p_mean = p.mean(dim=(0, 1, 2))
                ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean().item()
            else:
                p_mean = p.mean(dim=(0, 1))
                ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean().item()

            # Identity fraction: weight on last source (identity = self-connection)
            identity_weight = mean_weights[-1].item()
            total_weight = mean_weights.sum().item()

            src_labels = [f"L{j}" if j > 0 else "emb" for j in range(n_src)]
            weights_str = ", ".join(f"{src_labels[j]}={mean_weights[j].item():.3f}" for j in range(n_src))

            print(f"  Layer {l+1} ({n_src} sources): entropy={ent:.3f}, "
                  f"identity_frac={identity_weight/total_weight:.2%}")
            print(f"    weights: {weights_str}")


def identity_baseline(fourway, predictor, eval_batches, device, vocab_size, num_layers):
    """Compare learned routing vs identity routing on eval data."""
    print("\n" + "=" * 60)
    print("2. IDENTITY ROUTING BASELINE")
    print("=" * 60)

    # Eval with learned routing
    learned_nll = 0.0
    identity_nll = 0.0
    n = 0

    with torch.no_grad():
        for eb in eval_batches[:10]:
            eids = eb["olmo_ids"].to(device)
            elabels = eb["olmo_labels"].to(device)

            # Learned routing
            rw = predictor(eids)
            logits_learned = fourway(eids, rw)
            nll_l = F.cross_entropy(
                logits_learned.contiguous().view(-1, vocab_size),
                elabels.contiguous().view(-1),
            )
            learned_nll += nll_l.item()

            # Identity routing: replace all weights with [0,...,0,1]
            rw_identity = {'q': [], 'k': [], 'v': [], 'r': []}
            for stream in ('q', 'k', 'v', 'r'):
                for alpha in rw[stream]:
                    identity = torch.zeros_like(alpha)
                    identity[..., -1] = 1.0
                    rw_identity[stream].append(identity)

            logits_identity = fourway(eids, rw_identity)
            nll_i = F.cross_entropy(
                logits_identity.contiguous().view(-1, vocab_size),
                elabels.contiguous().view(-1),
            )
            identity_nll += nll_i.item()

            n += 1

    learned_nll /= max(n, 1)
    identity_nll /= max(n, 1)

    print(f"  Learned routing eval NLL:  {learned_nll:.4f}")
    print(f"  Identity routing eval NLL: {identity_nll:.4f}")
    print(f"  Difference:                {learned_nll - identity_nll:+.4f}")
    if learned_nll < identity_nll:
        print(f"  → Learned routing is BETTER by {identity_nll - learned_nll:.4f}")
    else:
        print(f"  → Learned routing is WORSE by {learned_nll - identity_nll:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = DAGFormerPretrainConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading from {args.checkpoint}...")
    base_model, predictor, fourway = load_model(config, args.checkpoint, device)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)
    eval_batches = build_eval_dataloader(
        olmo_tokenizer=tokenizer, seq_len=config.seq_len,
        batch_size=config.micro_batch_size,
        dataset_name=config.dataset, dataset_version=config.dataset_name,
        eval_skip=config.eval_skip, eval_size=config.eval_size,
        cache_path=os.path.join(config.save_dir, "eval_cache.pt"),
    )

    analyze_routing(predictor, eval_batches, device, config.num_hidden_layers, config.num_attention_heads)
    identity_baseline(fourway, predictor, eval_batches, device, config.vocab_size, config.num_hidden_layers)

    print("\nDone.")


if __name__ == "__main__":
    main()
