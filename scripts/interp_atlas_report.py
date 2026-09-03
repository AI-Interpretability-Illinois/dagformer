"""Turn the atlas JSON into named heads: effect matrix, specialization
labels, heatmaps, and a markdown table.

Effect definitions (positive = ablation hurts):
  NLL-type metric:  d = metric_ablated - metric_baseline
  copy_acc metric:  d = 100 * (baseline - ablated)   [points]
Specialization (NLL-type): excess = d_metric - d_overall (nats beyond the
head's general effect). A head is labelled "specialized:<metric>" when its
largest excess >= --excess-thr (default 0.05 nats) or a copy metric drops
>= --copy-thr points (default 10); else "general" when d_overall >=
--general-thr (0.03); else "dispensable".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

NLL_KEYS_SPEC = None  # filled at runtime from baseline keys


def effect_matrix(block: dict, base: dict):
    heads = list(block.keys())
    keys = [k for k in base if k != "nll_overall"]
    M = np.zeros((len(heads), len(keys)))
    overall = np.zeros(len(heads))
    for i, h in enumerate(heads):
        m = block[h]
        overall[i] = m["nll_overall"] - base["nll_overall"]
        for j, k in enumerate(keys):
            if k.startswith("copy_acc"):
                M[i, j] = 100 * (base[k] - m[k])
            else:
                M[i, j] = (m[k] - base[k]) - overall[i]   # excess over general effect
    return heads, keys, M, overall


def label_head(keys, row, overall, excess_thr, copy_thr, general_thr):
    best, best_val = None, -1e9
    for k, v in zip(keys, row):
        thr = copy_thr if k.startswith("copy_acc") else excess_thr
        score = v / thr
        if v >= thr and score > best_val:
            best, best_val = k, score
    if best is not None:
        return f"specialized:{best}"
    if overall >= general_thr:
        return "general"
    return "dispensable"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="experiments/results/interp/atlas_step9000.json")
    ap.add_argument("--out", default="experiments/results/interp")
    ap.add_argument("--fig", default="experiments/figures")
    ap.add_argument("--excess-thr", type=float, default=0.05)
    ap.add_argument("--copy-thr", type=float, default=10.0)
    ap.add_argument("--general-thr", type=float, default=0.03)
    args = ap.parse_args()

    atlas = json.load(open(args.atlas))
    base = atlas["baseline"]
    report = {}
    md = []
    for abl in ("wire", "out"):
        if not atlas.get(abl):
            continue
        heads, keys, M, overall = effect_matrix(atlas[abl], base)
        labels = [label_head(keys, M[i], overall[i], args.excess_thr,
                             args.copy_thr, args.general_thr) for i in range(len(heads))]
        counts = {}
        for lb in labels:
            cat = lb.split(":")[0] if not lb.startswith("specialized") else lb
            counts[cat] = counts.get(cat, 0) + 1
        order = np.argsort(-overall)
        report[abl] = {
            "n_heads": len(heads), "label_counts": counts,
            "heads": {heads[i]: {"d_overall": float(overall[i]), "label": labels[i],
                                 "top_excess": {keys[j]: float(M[i, j]) for j in np.argsort(-M[i])[:3]}}
                      for i in range(len(heads))},
        }
        md.append(f"\n#### ablation = {abl}  ({len(heads)} heads)\n")
        md.append("label counts: " + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])))
        md.append("\n| head | Δ overall NLL | label | top excess (metric: value) |\n|---|---|---|---|")
        for i in order[:40]:
            top = ", ".join(f"{keys[j]}: {M[i, j]:+.3f}" for j in np.argsort(-M[i])[:2])
            md.append(f"| {heads[i]} | {overall[i]:+.4f} | {labels[i]} | {top} |")

        # heatmap of specialization matrix (z-scored per metric)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Z = (M - M.mean(0)) / (M.std(0) + 1e-9)
        fig, ax = plt.subplots(figsize=(0.45 * len(keys) + 3, 0.11 * len(heads) + 2))
        im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4)
        ax.set_yticks(range(len(heads))); ax.set_yticklabels(heads, fontsize=4)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys, rotation=90, fontsize=7)
        ax.set_title(f"Head x metric specialization (z-scored excess), ablation={abl}")
        fig.colorbar(im, ax=ax, shrink=0.5)
        fig.tight_layout()
        fig.savefig(Path(args.fig) / f"interp_atlas_{abl}.png", dpi=170)
        print(f"saved interp_atlas_{abl}.png")

    Path(args.out, "atlas_report.md").write_text("\n".join(md))
    json.dump(report, open(Path(args.out) / "atlas_report.json", "w"), indent=1)
    print("\n".join(md[:60]))


if __name__ == "__main__":
    main()
