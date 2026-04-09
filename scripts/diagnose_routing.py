"""Diagnose routing weights: stats, diversity, deviation, train vs eval KL, temperature sweep.

Usage:
    python scripts/diagnose_routing.py \
        --checkpoint checkpoints/fourway_joint_causal/checkpoint_step5000.pt \
        --config configs/fourway_joint_causal.yaml
"""

import argparse
import csv
import os
import sys
from dataclasses import fields

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.pretrain_dagformer import (
    DAGFormerPretrainConfig, create_model, FourWayDAGFormer, FourWayPredictor,
)
from src.data.dolma import build_eval_dataloader, build_train_dataloader


def load_model_and_predictor(config, checkpoint_path, device):
    """Load base model + predictor from checkpoint."""
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

    fourway_model = FourWayDAGFormer(
        model=base_model,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
    ).to(device)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'predictor_state_dict' in ckpt:
        predictor.load_state_dict(ckpt['predictor_state_dict'])
    if 'model_state_path' in ckpt:
        model_state = torch.load(ckpt['model_state_path'], map_location=device)
        base_model.load_state_dict(model_state)
        del model_state

    predictor.eval()
    fourway_model.eval()
    return base_model, predictor, fourway_model


def get_routing_weights(predictor, batch, device):
    """Run predictor on a batch, return routing weights dict."""
    input_ids = batch["olmo_ids"].to(device)
    with torch.no_grad():
        rw = predictor(input_ids)
    return rw


def compute_stats(rw, prefix=""):
    """Compute per-layer per-stream stats."""
    results = []
    for stream in ('q', 'k', 'v', 'r'):
        for l, alpha in enumerate(rw[stream]):
            a = alpha.float()
            # Entropy of softmax'd weights
            p = F.softmax(a, dim=-1)
            ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean().item()

            results.append({
                'prefix': prefix,
                'stream': stream,
                'layer': l + 1,
                'mean': a.mean().item(),
                'std': a.std().item(),
                'min': a.min().item(),
                'max': a.max().item(),
                'entropy': ent,
                'n_sources': a.shape[-1],
            })
    return results


def compute_diversity(rw, prefix=""):
    """Per-layer pairwise cosine similarity across tokens."""
    results = []
    for stream in ('q', 'k', 'v', 'r'):
        for l, alpha in enumerate(rw[stream]):
            a = alpha.float()
            # Flatten per-token: [B, T, ...] → [B*T, dim]
            if a.dim() == 4:  # [B, T, H, L] for q/k/v
                bt = a.shape[0] * a.shape[1]
                flat = a.view(bt, -1)
            else:  # [B, T, L] for r
                bt = a.shape[0] * a.shape[1]
                flat = a.view(bt, -1)

            # Sample 256 tokens to avoid O(n^2) cost
            n = min(256, bt)
            idx = torch.randperm(bt)[:n]
            sampled = flat[idx]
            sampled_norm = F.normalize(sampled, dim=-1)
            cos_sim = (sampled_norm @ sampled_norm.T).triu(diagonal=1)
            mask = torch.ones_like(cos_sim).triu(diagonal=1).bool()
            mean_cos = cos_sim[mask].mean().item()

            results.append({
                'prefix': prefix,
                'stream': stream,
                'layer': l + 1,
                'mean_cosine_sim': mean_cos,
            })
    return results


def compute_deviation(rw, prefix=""):
    """L2 distance from identity [0,...,0,1] per layer."""
    results = []
    for stream in ('q', 'k', 'v', 'r'):
        for l, alpha in enumerate(rw[stream]):
            a = alpha.float()
            identity = torch.zeros_like(a)
            identity[..., -1] = 1.0
            l2 = (a - identity).pow(2).sum(dim=-1).sqrt().mean().item()
            results.append({
                'prefix': prefix,
                'stream': stream,
                'layer': l + 1,
                'l2_from_identity': l2,
            })
    return results


def compute_kl(rw_train, rw_eval):
    """KL divergence between train and eval routing (after softmax)."""
    results = []
    for stream in ('q', 'k', 'v', 'r'):
        for l in range(len(rw_train[stream])):
            p = F.softmax(rw_train[stream][l].float(), dim=-1).mean(dim=(0, 1))
            q = F.softmax(rw_eval[stream][l].float(), dim=-1).mean(dim=(0, 1))
            # Average KL across heads (if q/k/v) or just compute
            if p.dim() == 2:  # [H, L] for q/k/v
                kl = (p * (p / (q + 1e-8) + 1e-8).log()).sum(dim=-1).mean().item()
            else:  # [L] for r
                kl = (p * (p / (q + 1e-8) + 1e-8).log()).sum().item()
            results.append({
                'stream': stream,
                'layer': l + 1,
                'kl_train_eval': kl,
            })
    return results


def temperature_sweep(fourway_model, predictor, eval_batches, device, vocab_size, temps=[0.5, 1.0, 2.0]):
    """Sweep temperature on routing weights, measure eval NLL."""
    results = []
    for tau in temps:
        total_nll = 0.0
        n = 0
        with torch.no_grad():
            for eb in eval_batches:
                eids = eb["olmo_ids"].to(device)
                elabels = eb["olmo_labels"].to(device)
                rw = predictor(eids)
                # Scale routing weights
                for stream in ('q', 'k', 'v', 'r'):
                    for i, alpha in enumerate(rw[stream]):
                        rw[stream][i] = alpha / tau
                logits = fourway_model(eids, rw)
                nll = F.cross_entropy(
                    logits.contiguous().view(-1, vocab_size),
                    elabels.contiguous().view(-1),
                )
                total_nll += nll.item()
                n += 1
        avg_nll = total_nll / max(n, 1)
        results.append({'temperature': tau, 'eval_nll': avg_nll})
        print(f"  τ={tau:.1f}: eval_nll={avg_nll:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="experiments/routing_diagnosis")
    args = parser.parse_args()

    config = DAGFormerPretrainConfig.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading from {args.checkpoint}...")
    base_model, predictor, fourway_model = load_model_and_predictor(config, args.checkpoint, device)

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)

    # Get one train batch and one eval batch
    print("Building data...")
    train_loader = build_train_dataloader(
        olmo_tokenizer=tokenizer, seq_len=config.seq_len,
        batch_size=config.micro_batch_size,
        dataset_name=config.dataset, dataset_version=config.dataset_name,
        rank=0, world_size=1,
    )
    train_batch = next(iter(train_loader))

    eval_batches = build_eval_dataloader(
        olmo_tokenizer=tokenizer, seq_len=config.seq_len,
        batch_size=config.micro_batch_size,
        dataset_name=config.dataset, dataset_version=config.dataset_name,
        eval_skip=config.eval_skip, eval_size=config.eval_size,
        cache_path=os.path.join(config.save_dir, "eval_cache.pt"),
    )

    eval_batch = eval_batches[0] if eval_batches else train_batch

    print("\n=== A. Routing Weight Stats ===")
    rw_train = get_routing_weights(predictor, train_batch, device)
    rw_eval = get_routing_weights(predictor, eval_batch, device)

    train_stats = compute_stats(rw_train, "train")
    eval_stats = compute_stats(rw_eval, "eval")
    all_stats = train_stats + eval_stats

    with open(os.path.join(args.output_dir, "routing_stats.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
        writer.writeheader()
        writer.writerows(all_stats)

    print("Train routing stats (last 3 layers):")
    for s in train_stats[-12:]:
        print(f"  {s['stream']} L{s['layer']}: mean={s['mean']:.3f} std={s['std']:.3f} ent={s['entropy']:.3f}")

    print("\n=== B. Per-token Diversity ===")
    train_div = compute_diversity(rw_train, "train")
    eval_div = compute_diversity(rw_eval, "eval")

    print("Train diversity (mean cosine sim, last 3 layers):")
    for d in train_div[-12:]:
        print(f"  {d['stream']} L{d['layer']}: cos_sim={d['mean_cosine_sim']:.4f}")

    print("\n=== C. Deviation from Identity ===")
    train_dev = compute_deviation(rw_train, "train")
    eval_dev = compute_deviation(rw_eval, "eval")

    print("Train deviation (last 3 layers):")
    for d in train_dev[-12:]:
        print(f"  {d['stream']} L{d['layer']}: L2={d['l2_from_identity']:.4f}")

    print("\n=== D. Train vs Eval KL Divergence ===")
    kl_results = compute_kl(rw_train, rw_eval)

    with open(os.path.join(args.output_dir, "routing_kl.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=kl_results[0].keys())
        writer.writeheader()
        writer.writerows(kl_results)

    print("KL(train || eval):")
    for k in kl_results:
        print(f"  {k['stream']} L{k['layer']}: KL={k['kl_train_eval']:.4f}")

    print("\n=== H. Temperature Sweep ===")
    temp_results = temperature_sweep(
        fourway_model, predictor, eval_batches[:10], device,
        config.vocab_size, temps=[0.25, 0.5, 1.0, 2.0, 4.0],
    )

    with open(os.path.join(args.output_dir, "temperature_sweep.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=temp_results[0].keys())
        writer.writeheader()
        writer.writerows(temp_results)

    print("\nDiagnosis complete. Results saved to", args.output_dir)


if __name__ == "__main__":
    main()
