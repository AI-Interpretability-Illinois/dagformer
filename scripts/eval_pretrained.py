"""Held-out NLL evaluation of pretrained checkpoints (dense / fourway / modular).

Loads each model directory's config.yaml + latest ``checkpoint_step*.pt`` (the
trainer's layout) through the verified loaders and reports mean per-token NLL
on every eval cache given. Used by the reserved-node queue to compare the 300M
fourway_corrected and fourway_modular runs and pick the 1B architecture.

Usage (GPU):
    python scripts/eval_pretrained.py \
        --model /work/.../300m_fourway_corrected --model /work/.../300m_fourway_modular \
        --eval dolma=/work/.../dolma_v1_7_21b/eval_cache.pt --eval wikitext2=... \
        --out experiments/pretrain_300m/eval_300m.json --decide experiments/pretrain_1b
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval_lm_harness import load_dense, load_fourway  # noqa: E402


def latest_checkpoint(model_dir: str) -> tuple[str, int]:
    best, best_step = None, -1
    for f in glob.glob(os.path.join(model_dir, "checkpoint_step*.pt")):
        m = re.search(r"checkpoint_step(\d+)\.pt$", f)
        if m and int(m.group(1)) > best_step:
            best, best_step = f, int(m.group(1))
    if best is None and os.path.exists(os.path.join(model_dir, "checkpoint.pt")):
        # shared-model layout (checkpoint.pt only): read the step from the file if it has one
        best = os.path.join(model_dir, "checkpoint.pt")
        try:
            best_step = int(torch.load(best, map_location="cpu", weights_only=False).get("step", -1))
        except Exception:
            best_step = -1
    assert best, f"no checkpoint in {model_dir}"
    return best, best_step


@torch.no_grad()
def nll_on_cache(forward, cache_path: str, device, batch_size: int = 8) -> float:
    batches = torch.load(cache_path, map_location="cpu", weights_only=False)
    ids = torch.cat([b["olmo_ids"] for b in batches])
    labels = torch.cat([b["olmo_labels"] for b in batches])
    total, n = 0.0, 0
    for i in range(0, ids.shape[0], batch_size):
        x = ids[i:i + batch_size].to(device)
        y = labels[i:i + batch_size].to(device)
        logits = forward(x)
        total += F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                                 reduction="sum").item()
        n += y.numel()
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, help="model dir (config.yaml + checkpoints)")
    p.add_argument("--config", action="append", default=[],
                   help="config.yaml per --model if not inside the dir (same order)")
    p.add_argument("--name", action="append", default=[], help="result key per --model (default: dir basename)")
    p.add_argument("--eval", action="append", required=True, help="name=eval_cache.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--decide", default="", help="dir to write CHOSEN files into (lowest NLL on the first eval wins)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = torch.device(args.device)
    evals = [e.split("=", 1) for e in args.eval]

    results: dict = {}
    for i, mdir in enumerate(args.model):
        cfg_path = args.config[i] if i < len(args.config) else os.path.join(mdir, "config.yaml")
        cfg = yaml.safe_load(open(cfg_path))
        ckpt, step = latest_checkpoint(mdir)
        name = args.name[i] if i < len(args.name) else os.path.basename(mdir.rstrip("/"))
        if str(cfg.get("routing_mode", "")).startswith("fourway"):
            fw, pred = load_fourway(ckpt, cfg, device)
            fwd = lambda x: fw(x, pred(x))  # noqa: E731
        else:
            model = load_dense(ckpt, cfg, device)
            fwd = lambda x: model(input_ids=x).logits  # noqa: E731
        r = {"checkpoint": ckpt, "step": step, "routing_mode": cfg.get("routing_mode", "dense")}
        for ename, epath in evals:
            r[ename] = nll_on_cache(fwd, epath, device)
            print(f"{name} @ {step}: {ename} nll={r[ename]:.4f}", flush=True)
        results[name] = r
        del fwd
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    md = [f"| model | step | " + " | ".join(e[0] for e in evals) + " |", "|---|---|" + "---|" * len(evals)]
    for name, r in results.items():
        md.append(f"| {name} | {r['step']} | " + " | ".join(f"{r[e[0]]:.4f}" for e in evals) + " |")
    open(args.out.rsplit(".", 1)[0] + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))

    if args.decide:
        primary = evals[0][0]
        best = min(results, key=lambda n: results[n][primary])
        os.makedirs(args.decide, exist_ok=True)
        arch = results[best]["routing_mode"]
        with open(os.path.join(args.decide, "CHOSEN"), "w") as f:
            f.write(f"{arch}\n{best}\n{json.dumps({n: results[n][primary] for n in results})}\n")
        for n in results:
            if n != best:
                open(os.path.join(args.decide, f"CHOSEN_NOT_{results[n]['routing_mode']}"), "w").close()
        print(f"decision: {arch} ({best}) has the lowest {primary} NLL -> {args.decide}/CHOSEN")


if __name__ == "__main__":
    main()
