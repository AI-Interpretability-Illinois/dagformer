"""Common per-window FP32 NLL and complete parameter/compute accounting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_lm_harness import load_dense, load_fourway
from scripts.scaling_collect import train_flops_per_token


def progress(path):
    try:
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except RuntimeError:
        state = torch.load(path, map_location="cpu", weights_only=False)
    updates = state.get("completed_updates")
    steps = [int(v["step"]) for v in state.get("optimizer_state_dict", {}).get("state", {}).values()
             if "step" in v]
    source = "completed_updates"
    if updates is None:
        updates = max(steps) if steps else None
        source = "optimizer_state" if steps else "unavailable"
    return {"checkpoint_step": state.get("step"), "completed_updates": updates,
            "optimizer_updates": max(steps) if steps else None,
            "progress_source": source}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval", action="append", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--expected-updates", type=int)
    p.add_argument("--tokens-per-update", type=int)
    p.add_argument("--staging-metadata", help="Export provenance containing the original optimizer update counts")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    info = progress(args.checkpoint)
    staging = None
    if args.staging_metadata:
        staging = json.loads(Path(args.staging_metadata).read_text())
        if staging["checkpoint_step"] != info["checkpoint_step"]:
            raise ValueError("Export metadata and checkpoint steps differ")
        counts = staging.get("optimizer_update_counts", [])
        if len(counts) == 1 and info["completed_updates"] is None:
            info.update(completed_updates=counts[0], optimizer_updates=counts[0],
                        progress_source="exported_optimizer_count")
        if args.tokens_per_update is None:
            args.tokens_per_update = staging.get("tokens_per_update")
    if args.expected_updates is not None and info["completed_updates"] != args.expected_updates:
        raise ValueError(f"Checkpoint training budget mismatch: {info}, expected {args.expected_updates}")
    if args.expected_updates is not None and info["optimizer_updates"] != args.expected_updates:
        raise ValueError(f"Optimizer update count differs from the planned budget: {info}")
    device = torch.device(args.device)
    routed = str(cfg.get("routing_mode", "")).startswith("fourway")
    if routed:
        model, predictor = load_fourway(args.checkpoint, cfg, device)
        modules = [model, predictor]
        forward = lambda x: model(x, predictor(x))
    else:
        model = load_dense(args.checkpoint, cfg, device)
        modules = [model]
        forward = lambda x: model(input_ids=x).logits
    total_params = sum(p.numel() for m in modules for p in m.parameters())
    embeddings = {id(m.weight): m.weight for root in modules for m in root.modules()
                  if isinstance(m, torch.nn.Embedding)}
    embedding_params = sum(p.numel() for p in embeddings.values())
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays, evaluations = {}, {}
    start = time.time()
    with torch.inference_mode():
        for item in args.eval:
            name, cache = item.split("=", 1)
            batches = torch.load(cache, map_location="cpu", weights_only=False)
            ids = torch.cat([b["olmo_ids"] for b in batches])
            labels = torch.cat([b["olmo_labels"] for b in batches])
            losses, counts = [], []
            for lo in range(0, len(ids), args.batch_size):
                x = ids[lo:lo + args.batch_size].to(device)
                y = labels[lo:lo + args.batch_size].to(device)
                logits = forward(x)
                per_token = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                            y.reshape(-1), reduction="none").reshape_as(y)
                losses.extend(per_token.sum(1).cpu().tolist())
                counts.extend((y != -100).sum(1).cpu().tolist())
                del logits, per_token, x, y
            arrays[f"{name}_nll_sum"] = np.asarray(losses)
            arrays[f"{name}_tokens"] = np.asarray(counts)
            evaluations[name] = {"nll": sum(losses) / sum(counts), "tokens": sum(counts),
                                 "windows": len(losses), "cache": cache}
            print(name, evaluations[name], flush=True)
    flops = train_flops_per_token(cfg, cfg.get("seq_len", 1024))
    tokens = (info["completed_updates"] * args.tokens_per_update
              if info["completed_updates"] is not None and args.tokens_per_update is not None else None)
    result = {"checkpoint": args.checkpoint, "config": args.config, **info,
              "staging_metadata": staging,
              "total_params": total_params, "embedding_params": embedding_params,
              "non_embedding_params": total_params - embedding_params,
              "training_tokens": tokens, "analytic_flops": flops,
              "analytic_training_flops": tokens * flops["train_flops_per_token"] if tokens else None,
              "evals": evaluations, "eval_seconds": time.time() - start,
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
              "torch": torch.__version__}
    np.savez_compressed(output.with_suffix(".npz"), **arrays)
    tmp = output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(output)


if __name__ == "__main__":
    main()
