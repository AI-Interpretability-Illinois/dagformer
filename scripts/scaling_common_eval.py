"""Re-evaluate every finished pretraining run of one machine on the SAME held-out caches.

The training-time eval curves in metrics.csv use a per-corpus eval cache, so runs
trained on different corpora (Delta 21B, shared 12B slice, timan 12B rebuild) are
not directly comparable. This evaluates the final checkpoint of each run listed in
experiments/scaling/runs.yaml (for --machine) on a fixed set of caches and writes
experiments/scaling/common_eval_<machine>.json (resumable: runs already in the
file are skipped unless --force).

    python scripts/scaling_common_eval.py --machine delta \
        --eval dolma21b=/work/.../dolma_v1_7_21b/eval_cache.pt --eval wikitext2=... \
        --eval mathinstruct=... --eval gsm8k=...
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval_lm_harness import (load_dense, load_denseformer, load_fourway,  # noqa: E402
                                     load_hyperconnection, load_muddformer, load_hc_paper, load_mhc)

# Baseline families with their own architecture: loading them through load_dense would silently drop their
# routing parameters (the 2026-10-05 DenseFormer re-eval was 0.45 nats off for that reason).
FAMILY_LOADERS = {"denseformer": load_denseformer, "muddformer": load_muddformer,
                  "hyperconnection": load_hyperconnection, "hc_paper": load_hc_paper, "mhc": load_mhc}
from scripts.eval_pretrained import latest_checkpoint, nll_on_cache  # noqa: E402
from scripts.scaling_collect import is_done  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="experiments/scaling/runs.yaml")
    p.add_argument("--machine", required=True)
    p.add_argument("--eval", action="append", required=True, help="name=eval_cache.pt")
    p.add_argument("--out", default=None)
    p.add_argument("--only", default="", help="comma-separated run names (default: all finished)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()
    device = torch.device(args.device)
    evals = [e.split("=", 1) for e in args.eval]
    out = args.out or f"experiments/scaling/common_eval_{args.machine}.json"
    results = json.load(open(out)) if os.path.exists(out) else {}
    only = set(filter(None, args.only.split(",")))
    root = os.path.join(os.path.dirname(__file__), "..")

    for r in yaml.safe_load(open(args.manifest))["runs"]:
        if r["machine"] != args.machine or (only and r["name"] not in only):
            continue
        if not os.path.isdir(r["save_dir"]):
            print(f"skip (missing dir): {r['name']}"); continue
        cfg_path = r["config"] if os.path.isabs(r["config"]) else os.path.join(root, r["config"])
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(r["save_dir"], "config.yaml")
        cfg = yaml.safe_load(open(cfg_path))
        total_steps = int(r.get("steps_override") or cfg["total_steps"])
        if not is_done(r["save_dir"], total_steps):
            print(f"skip (not finished): {r['name']}"); continue
        if r["name"] in results and not args.force and all(e[0] in results[r["name"]] for e in evals):
            print(f"skip (already evaluated): {r['name']}"); continue
        ckpt, step = latest_checkpoint(r["save_dir"])
        if str(cfg.get("routing_mode", "")).startswith("fourway"):
            fw, pred = load_fourway(ckpt, cfg, device)
            fwd = lambda x: fw(x, pred(x))  # noqa: E731
        elif r.get("family") in FAMILY_LOADERS:
            model = FAMILY_LOADERS[r["family"]](ckpt, cfg, device)
            fwd = lambda x: model(input_ids=x).logits  # noqa: E731
        else:
            model = load_dense(ckpt, cfg, device)
            fwd = lambda x: model(input_ids=x).logits  # noqa: E731
        res = {"checkpoint": ckpt, "step": step, "family": r["family"], "size": r["size"], "corpus": r["corpus"]}
        for ename, epath in evals:
            res[ename] = nll_on_cache(fwd, epath, device, batch_size=args.batch_size)
            print(f"{r['name']} @ {step}: {ename} nll={res[ename]:.4f}", flush=True)
        results[r["name"]] = res
        del fwd
        torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(results, open(out, "w"), indent=1)
    print(f"wrote {out} ({len(results)} runs)")


if __name__ == "__main__":
    main()
