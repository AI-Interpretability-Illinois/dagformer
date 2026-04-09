"""Benchmark inference latency: prefill + generation for dense vs routing model.

Usage:
    python scripts/benchmark_inference.py \
        --checkpoint checkpoints/fourway_reg_l2decay/checkpoint_step4000.pt \
        --config configs/fourway_reg_l2decay.yaml
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.pretrain_dagformer import DAGFormerPretrainConfig, create_model
from src.model.predictor import FourWayPredictor
from src.model.olmo_graph import FourWayDAGFormer


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
    fourway = FourWayDAGFormer(
        model=base_model,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    if 'predictor_state_dict' in ckpt:
        predictor.load_state_dict(ckpt['predictor_state_dict'])
    if 'model_state_path' in ckpt:
        model_state = torch.load(ckpt['model_state_path'], map_location=device)
        base_model.load_state_dict(model_state)
        del model_state

    predictor.eval()
    fourway.eval()
    base_model.eval()
    return base_model, predictor, fourway


def benchmark_prefill(model_fn, input_ids, warmup=3, repeats=10):
    """Measure prefill latency (processing full prompt)."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model_fn(input_ids)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model_fn(input_ids)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times


def benchmark_generation(model_fn, prompt_ids, gen_tokens=128, warmup=2, repeats=5):
    """Measure autoregressive generation latency (token by token).

    Note: This is a naive implementation without KV cache.
    Each new token requires reprocessing the entire sequence.
    """
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model_fn(prompt_ids)
    torch.cuda.synchronize()

    times = []
    per_token_times = []
    for _ in range(repeats):
        ids = prompt_ids.clone()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        token_times = []
        for step in range(gen_tokens):
            tt0 = time.perf_counter()
            with torch.no_grad():
                logits = model_fn(ids)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_token], dim=1)
            torch.cuda.synchronize()
            tt1 = time.perf_counter()
            token_times.append(tt1 - tt0)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        per_token_times.append(token_times)

    return times, per_token_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt_len", type=int, default=512)
    parser.add_argument("--gen_tokens", type=int, default=64)
    args = parser.parse_args()

    config = DAGFormerPretrainConfig.from_yaml(args.config)
    device = torch.device("cuda")

    print(f"Loading from {args.checkpoint}...")
    base_model, predictor, fourway = load_model(config, args.checkpoint, device)

    # Create random prompt
    prompt = torch.randint(0, config.vocab_size, (1, args.prompt_len), device=device)

    # Dense model function
    def dense_fn(ids):
        return base_model(ids).logits

    # Routing model function
    def routing_fn(ids):
        rw = predictor(ids)
        return fourway(ids, rw)

    print(f"\n=== Prefill Benchmark (prompt_len={args.prompt_len}) ===")

    dense_times = benchmark_prefill(dense_fn, prompt)
    print(f"Dense:   mean={sum(dense_times)/len(dense_times)*1000:.1f}ms, "
          f"min={min(dense_times)*1000:.1f}ms, max={max(dense_times)*1000:.1f}ms")

    routing_times = benchmark_prefill(routing_fn, prompt)
    print(f"Routing: mean={sum(routing_times)/len(routing_times)*1000:.1f}ms, "
          f"min={min(routing_times)*1000:.1f}ms, max={max(routing_times)*1000:.1f}ms")

    overhead = (sum(routing_times)/len(routing_times)) / (sum(dense_times)/len(dense_times))
    print(f"Overhead: {overhead:.2f}x")

    print(f"\n=== Generation Benchmark (prompt={args.prompt_len}, gen={args.gen_tokens}) ===")
    print("Note: naive implementation, no KV cache (each token reprocesses full sequence)")

    dense_gen_times, dense_per_token = benchmark_generation(
        dense_fn, prompt, gen_tokens=args.gen_tokens, warmup=1, repeats=3)
    mean_dense = sum(dense_gen_times)/len(dense_gen_times)
    mean_dense_per_tok = mean_dense / args.gen_tokens
    print(f"Dense:   total={mean_dense*1000:.0f}ms, per_token={mean_dense_per_tok*1000:.1f}ms, "
          f"tok/s={args.gen_tokens/mean_dense:.1f}")

    routing_gen_times, routing_per_token = benchmark_generation(
        routing_fn, prompt, gen_tokens=args.gen_tokens, warmup=1, repeats=3)
    mean_routing = sum(routing_gen_times)/len(routing_gen_times)
    mean_routing_per_tok = mean_routing / args.gen_tokens
    print(f"Routing: total={mean_routing*1000:.0f}ms, per_token={mean_routing_per_tok*1000:.1f}ms, "
          f"tok/s={args.gen_tokens/mean_routing:.1f}")

    gen_overhead = mean_routing / mean_dense
    print(f"Overhead: {gen_overhead:.2f}x")

    print("\nDone.")


if __name__ == "__main__":
    main()
