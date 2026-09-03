"""Render figures from interp pilot outputs (swap_eval + probe JSONs).

Usage:
  python3 scripts/interp_plot.py --results experiments/results/interp \
      --step 9000 --out experiments/figures
Produces:
  interp_swap_ladder.png      NLL under routing interventions
  interp_probe_acc.png        probe accuracy vs baselines per target
  interp_emergence.png        probe accuracy across training steps
  interp_induction_heat.png   |delta alpha| by stream x layer (repeat-vs-random)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LADDER_ORDER = [
    "identity_no_corr", "identity", "static_mean_no_corr", "static_mean",
    "shuffle_time", "swap_context", "static_q", "static_k", "static_v",
    "static_r", "dynamic_no_corr", "dynamic",
]


def plot_swap(swap: dict, out: Path, step: int):
    nll = swap["modes_nll"]
    names = [m for m in LADDER_ORDER if m in nll]
    vals = [nll[m] for m in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#888888" if "identity" in m else
              "#c78f2f" if "static" in m else
              "#7a5195" if m in ("shuffle_time", "swap_context") else
              "#2f6fc7" for m in names]
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    if "dense_baseline" in nll:
        ax.axvline(nll["dense_baseline"], ls="--", c="k", lw=1,
                   label=f"dense baseline ({nll['dense_baseline']:.3f})")
        ax.legend(fontsize=8)
    ax.axvline(nll.get("dynamic", np.nan), ls=":", c="#2f6fc7", lw=1)
    ax.set_xlabel("held-out NLL (nats/token)")
    ax.set_title(f"Routing intervention ladder — 300M FourWay step {step}")
    lo = min(vals + [nll.get("dense_baseline", min(vals))])
    hi = max(vals)
    ax.set_xlim(lo - 0.02, hi + 0.02)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "interp_swap_ladder.png", dpi=150)
    print("saved interp_swap_ladder.png")


def plot_probes(pr: dict, out: Path, step: int):
    targets = [t for t in pr if isinstance(pr[t], dict) and "alpha" in pr[t]]
    keys = ["alpha", "alpha_resid", "token_id", "majority"]
    x = np.arange(len(targets))
    w = 0.2
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for k, off, c in zip(keys, (-1.5, -0.5, 0.5, 1.5),
                         ("#2f6fc7", "#6fa8dc", "#c78f2f", "#bbbbbb")):
        ax.bar(x + off * w, [pr[t].get(k, np.nan) for t in targets], w, label=k, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=9)
    ax.set_ylabel("probe accuracy")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8)
    ax.set_title(f"Linear probes on per-token routing (step {step}) vs baselines")
    fig.tight_layout()
    fig.savefig(out / "interp_probe_acc.png", dpi=150)
    print("saved interp_probe_acc.png")


def plot_emergence(em: dict, out: Path):
    if not em:
        return
    steps = sorted(int(s) for s in em)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for t in sorted({k for s in em.values() for k in s}):
        ax.plot(steps, [em[str(s)].get(t) if str(s) in em else em[s].get(t)
                        for s in steps], marker="o", label=t)
    ax.set_xlabel("training step")
    ax.set_ylabel("probe accuracy")
    ax.set_title("Emergence of decodable structure in routing")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "interp_emergence.png", dpi=150)
    print("saved interp_emergence.png")


def plot_induction(ind: dict, out: Path, step: int):
    by = ind["by_stream_layer"]  # keys like "q/L3"
    streams = ["q", "k", "v", "r"]
    layers = sorted({int(k.split("/L")[1]) for k in by})
    mat = np.full((len(streams), len(layers)), np.nan)
    for k, v in by.items():
        s, l = k.split("/L")
        mat[streams.index(s), layers.index(int(l))] = v
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(streams)))
    ax.set_yticklabels([s.upper() for s in streams])
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{l}" for l in layers], fontsize=8)
    ax.set_title(f"|mean Δα| repeat vs random (positions ≥128), step {step}")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    fig.savefig(out / "interp_induction_heat.png", dpi=150)
    print("saved interp_induction_heat.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results/interp")
    ap.add_argument("--step", type=int, default=9000)
    ap.add_argument("--out", default="experiments/figures")
    args = ap.parse_args()
    res = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    swap_p = res / f"swap_eval_step{args.step}.json"
    if swap_p.exists():
        plot_swap(json.load(open(swap_p)), out, args.step)
    probe_p = res / f"probe_results_step{args.step}.json"
    if probe_p.exists():
        pj = json.load(open(probe_p))
        plot_probes(pj["probes"], out, args.step)
        plot_emergence(pj.get("emergence", {}), out)
        plot_induction(pj["induction"], out, args.step)
    print("DONE")


if __name__ == "__main__":
    main()
