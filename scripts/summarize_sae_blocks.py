"""Check SAE target-loss transfer while keeping adjacent text windows together."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--blocks", nargs="+", type=int, default=[1, 4, 8, 16])
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()
    source = json.loads(args.input.read_text())
    if not source["complete"]:
        raise ValueError("The SAE evaluation is not complete")
    result = {"source": str(args.input), "draws": args.draws, "seed": args.seed,
              "protocol": "circular moving-block bootstrap of token-weighted paired target NLL changes; zero-target draws omitted; block-length sensitivity for a fixed corpus and checkpoint; unadjusted intervals",
              "features": {}}
    lines = ["# SAE transfer with adjacent-window blocks", "",
             "WikiText articles can span several packed windows. This repeats the target-token",
             "loss uncertainty calculation while keeping blocks of adjacent windows together.",
             "Resampled counts weight the tokens, and draws with zero target tokens are omitted.",
             "Intervals are paired and unadjusted. The dose span is ΔNLL(+4) − ΔNLL(−4).", "",
             "| Feature | Contrast | ΔNLL | " + " | ".join(f"Block {b}: 95% interval" for b in args.blocks) + " |",
             "|---|---|---:|" + "---|" * len(args.blocks)]
    indexes = {}
    for feature_id, feature in source["features"].items():
        minus = feature["arms"]["feature/alpha-4"]["on_rule"]
        plus = feature["arms"]["feature/alpha4"]["on_rule"]
        counts = np.asarray(plus["window_token_count"])
        if not np.array_equal(counts, minus["window_token_count"]):
            raise ValueError("SAE dose arms have different target positions")
        n = len(counts)
        m, p = np.asarray(minus["window_delta_sum"]), np.asarray(plus["window_delta_sum"])
        output = {"target_tokens": int(counts.sum()), "target_windows": int((counts > 0).sum()), "contrasts": {}}
        for contrast, sums in (("alpha-4", m), ("alpha4", p), ("dose_span", p - m)):
            record = {"delta": float(sums.sum() / counts.sum()) if counts.sum() else None, "blocks": {}}
            cells = []
            for block in args.blocks:
                if not 1 <= block <= n:
                    raise ValueError("Block length is outside the window count")
                if (n, block) not in indexes:
                    starts = np.random.default_rng(args.seed + block).integers(n, size=(args.draws, (n + block - 1) // block))
                    indexes[n, block] = ((starts[..., None] + np.arange(block)) % n).reshape(args.draws, -1)[:, :n]
                idx = indexes[n, block]
                den = counts[idx].sum(1)
                keep = den > 0
                interval = np.quantile(sums[idx[keep]].sum(1) / den[keep], [.025, .975]).tolist() if keep.any() else None
                record["blocks"][str(block)] = {"paired_bootstrap_95ci": interval, "valid_draws": int(keep.sum())}
                cells.append(f"[{interval[0]:+.4f}, {interval[1]:+.4f}]" if interval else "no target tokens")
            output["contrasts"][contrast] = record
            point = f"{record['delta']:+.4f}" if record["delta"] is not None else "—"
            lines.append(f"| {feature_id} | {contrast} | {point} | " + " | ".join(cells) + " |")
        result["features"][feature_id] = output
    folder = args.input.parent
    (folder / "block_bootstrap.json").write_text(json.dumps(result, indent=2) + "\n")
    (folder / "block_bootstrap.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
