"""Sensitivity of routing-loss intervals to dependence between nearby windows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocks", nargs="+", type=int, default=[1, 4, 8, 16])
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()
    result = {"protocol": "circular moving-block bootstrap of paired NLL changes; consecutive packed windows are kept together within each sampled block; block 1 is the IID-window bootstrap; sensitivity analysis, not training-seed uncertainty",
              "args": {k: [str(x) for x in v] if k == "inputs" else str(v) if isinstance(v, Path) else v
                       for k, v in vars(args).items()}, "models": {}}
    lines = ["# Dependence between natural-text windows", "",
             "Adjacent nonoverlapping windows can share the same source article.",
             "This sensitivity analysis resamples circular blocks of 1, 4, 8 or 16 consecutive",
             "windows. Each window contains 1,024 evaluated tokens. The point estimate stays",
             "fixed; the intervals reflect different assumptions about dependence length.",
             "These are unadjusted paired intervals for fixed checkpoints and this packed corpus.", "",
             "| Model | Intervention | ΔNLL | Block 1: 95% CI | Block 4 | Block 8 | Block 16 |",
             "|---|---|---:|---|---|---|---|"]
    selected = {"pred_position/corr_dynamic": "predictor → position table",
                "pred_dynamic/corr_position": "correction → position table",
                "pred_dynamic/corr_zero": "disable correction"}
    indexes = {}
    for path in args.inputs:
        payload = json.loads(path.read_text())
        reference = np.array(payload["arms"]["pred_dynamic/corr_dynamic"]["nll"])
        n = len(reference)
        arms = {}
        for name, arm in payload["arms"].items():
            delta = np.array(arm["nll"]) - reference
            record = {"delta": float(delta.mean()), "n_windows": n, "intervals": {}}
            for block in args.blocks:
                if block < 1 or block > n:
                    raise ValueError("Block length must be between 1 and the number of windows")
                key = (n, block)
                if key not in indexes:
                    rng = np.random.default_rng(args.seed + block)
                    starts = rng.integers(n, size=(args.draws, (n + block - 1) // block))
                    indexes[key] = ((starts[..., None] + np.arange(block)) % n).reshape(args.draws, -1)[:, :n]
                means = delta[indexes[key]].mean(1)
                record["intervals"][str(block)] = np.quantile(means, [.025, .975]).tolist()
            arms[name] = record
            if name in selected:
                intervals = [f"[{record['intervals'][str(b)][0]:+.6f}, {record['intervals'][str(b)][1]:+.6f}]"
                             for b in args.blocks]
                lines.append(f"| {path.stem} | {selected[name]} | {record['delta']:+.6f} | " + " | ".join(intervals) + " |")
        result["models"][path.stem] = {"source": str(path), "arms": arms}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
