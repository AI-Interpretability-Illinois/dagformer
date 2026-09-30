"""Evaluate finished prune-finetune checkpoints on extra eval caches (e.g. GSM8K).

The final checkpoint of scripts/prune_finetune.py has the pruned weights baked
in, so it loads through the project's verified loaders with no masker. This
script walks a checkpoint root, loads every ``<run>/final`` and reports
per-token NLL on each eval cache given, appending to ``<run>/extra_eval.json``.

Usage (GPU job):
    python scripts/eval_pruned.py \
        --ckpt-root /work/hdd/bfqt/xiaocong/dagformer_pruning/checkpoints \
        --eval gsm8k=/work/hdd/bfqt/xiaocong/dagformer_pruning/data/gsm8k/eval_cache.pt
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval_lm_harness import load_dense, load_fourway  # noqa: E402


@torch.no_grad()
def nll_on_cache(forward, cache_path: str, device) -> float:
    batches = torch.load(cache_path, map_location="cpu", weights_only=False)
    total, n = 0.0, 0
    for b in batches:
        ids = b["olmo_ids"].to(device)
        labels = b["olmo_labels"].to(device)
        logits = forward(ids)
        total += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1)).item()
        n += 1
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", required=True)
    p.add_argument("--eval", action="append", required=True, help="name=path/to/eval_cache.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    device = torch.device(args.device)
    evals = dict(e.split("=", 1) for e in args.eval)

    for final in sorted(glob.glob(os.path.join(args.ckpt_root, "*", "final", "checkpoint.pt"))):
        run_dir = os.path.dirname(os.path.dirname(final))
        out_path = os.path.join(run_dir, "extra_eval.json")
        results = json.load(open(out_path)) if os.path.exists(out_path) else {}
        todo = {k: v for k, v in evals.items() if args.overwrite or k not in results}
        if not todo:
            continue
        cfg = yaml.safe_load(open(os.path.join(os.path.dirname(final), "config.yaml")))
        if str(cfg.get("routing_mode", "")).startswith("fourway"):
            fw, pred = load_fourway(final, cfg, device)
            forward = lambda ids: fw(ids, pred(ids))  # noqa: E731
        else:
            model = load_dense(final, cfg, device)
            forward = lambda ids: model(input_ids=ids).logits  # noqa: E731
        for name, path in todo.items():
            results[name] = nll_on_cache(forward, path, device)
            print(f"{os.path.basename(run_dir)}: {name} nll={results[name]:.4f}", flush=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        del forward
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
