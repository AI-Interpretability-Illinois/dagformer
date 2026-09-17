"""Optional figures for a discovered circuit.

matplotlib is not installed in every environment this runs in, so it is
imported lazily and this script is never a dependency of the pipeline — the
markdown and .dot files from steps 2 and 3 are the primary artifacts.

    python experiments/interp/plot_circuit.py --behavior honesty --channel pred

Produces, next to the circuit files:

    circuit_<b>_<c>_heatmap.png    |Delta| per (target layer, source), per stream
    circuit_<b>_<c>_sparsity.png   cumulative |Delta| mass vs a shuffled control
    circuit_<b>_<c>_volcano.png    effect size vs significance, circuit marked
    verify_<b>_<c>_dose.png        behaviour shift vs lambda, circuit vs controls
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from circuit_common import RoutingLayout

STREAMS = ("q", "k", "v", "r")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--behavior", default="honesty")
    ap.add_argument("--channel", default="pred", choices=("pred", "corr", "eff"))
    ap.add_argument("--dir", default="experiments/results/interp/circuits")
    ap.add_argument("--dpi", type=int, default=150)
    return ap.parse_args()


def heatmap(plt, layout: RoutingLayout, delta: np.ndarray, chosen: np.ndarray,
            out: Path, dpi: int) -> None:
    """|Delta| summed over heads, as (target layer) x (source) per stream."""
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2), constrained_layout=True)
    for ax, s in zip(axes, STREAMS):
        grid = np.full((layout.L, layout.L + 1), np.nan)
        sm = layout.stream_arr == s
        for l in range(1, layout.L):
            for src in range(l + 1):
                m = sm & (layout.layer_arr == l) & (layout.src_arr == src)
                if m.any():
                    grid[l, src] = np.abs(delta[m]).sum()
        im = ax.imshow(grid, aspect="auto", cmap="magma", origin="lower")
        # Mark where the selection rule actually landed.
        for l, src in {(int(layout.layer_arr[i]), int(layout.src_arr[i]))
                       for i in np.nonzero(chosen & sm)[0]}:
            ax.add_patch(plt.Rectangle((src - 0.5, l - 0.5), 1, 1,
                                       fill=False, ec="#00e5ff", lw=1.6))
        ax.set_title(f"stream {s}")
        ax.set_xlabel("source (0 = embedding, s = out of layer s-1)")
        ax.set_ylabel("target layer" if s == "q" else "")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle("|Delta| summed over heads; cyan = selected edges")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def sparsity(plt, delta: np.ndarray, out: Path, dpi: int) -> None:
    """Cumulative mass curve against the same values shuffled — the sparsity claim."""
    a = np.sort(np.abs(delta))[::-1]
    cum = a.cumsum() / max(a.sum(), 1e-12)
    unif = np.arange(1, a.size + 1) / a.size          # perfectly flat Delta
    fig, ax = plt.subplots(figsize=(5.4, 4.2), constrained_layout=True)
    ax.plot(np.arange(1, a.size + 1), cum, lw=2, label="|Delta|")
    ax.plot(np.arange(1, a.size + 1), unif, ls="--", c="gray", label="uniform")
    for frac, c in ((0.5, "#d62728"), (0.9, "#2ca02c")):
        n = int(np.searchsorted(cum, frac) + 1)
        ax.axvline(n, c=c, ls=":", lw=1.2)
        ax.text(n, frac, f" {n} edges = {frac:.0%}", c=c, va="bottom", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("edges, strongest first")
    ax.set_ylabel("cumulative share of |Delta| mass")
    ax.set_title("Is the difference matrix sparse?")
    ax.legend()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def volcano(plt, delta: np.ndarray, p: np.ndarray, chosen: np.ndarray,
            eligible: np.ndarray, out: Path, dpi: int) -> None:
    y = -np.log10(np.clip(p, 1e-12, 1.0))
    fig, ax = plt.subplots(figsize=(5.8, 4.4), constrained_layout=True)
    rest = eligible & ~chosen
    ax.scatter(delta[rest], y[rest], s=4, c="#bbbbbb", lw=0, label="not selected")
    ax.scatter(delta[chosen], y[chosen], s=14, c="#d62728", lw=0, label="circuit")
    ax.axvline(0, c="k", lw=0.6)
    ax.set_xlabel("Delta = alpha_pos - alpha_neg")
    ax.set_ylabel("-log10 p (sign-flip permutation)")
    ax.set_title("Effect size vs significance")
    ax.legend()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def dose(plt, arms: list[dict], headroom: float, out: Path, dpi: int) -> None:
    """Behaviour shift vs lambda for every arm family that has a grid."""
    fam: dict[str, dict[float, float]] = defaultdict(dict)
    for a in arms:
        m = re.match(r"^(.*)@([-\d.eE+]+)$", a["name"])
        if m and a.get("shift") is not None:
            fam[m.group(1)][float(m.group(2))] = a["shift"]
    fam = {k: v for k, v in fam.items() if len(v) > 1}
    if not fam:
        return
    add = {k: v for k, v in fam.items() if k.endswith("_add") or "_add_" in k}
    mul = {k: v for k, v in fam.items() if k not in add}
    panels = [(add, "additive:  alpha[M] += lam * Delta[M]"),
              (mul, "multiplicative:  alpha[M] *= (1 + lam)")]
    panels = [p for p in panels if p[0]]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.4),
                             constrained_layout=True, squeeze=False)
    for ax, (group, title) in zip(axes[0], panels):
        for name, pts in sorted(group.items()):
            lams = sorted(pts)
            style = dict(marker="o", lw=2.2) if name.startswith("circuit") else \
                dict(marker="x", lw=1.2, ls="--", alpha=0.8)
            ax.plot(lams, [pts[l] for l in lams], label=name, **style)
        if headroom:
            ax.axhline(headroom, c="k", ls=":", lw=1,
                       label="full instruction (headroom)")
        ax.axhline(0, c="k", lw=0.6)
        ax.set_xlabel("lambda")
        ax.set_ylabel("behaviour shift vs baseline arm")
        ax.set_title(title)
        ax.legend(fontsize=7)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "matplotlib is not installed in this environment. The pipeline does "
            "not need it — read the .md reports and render the .dot with "
            "`dot -Tpdf circuit_*.dot -o circuit.pdf` instead.")

    d = Path(args.dir)
    stem = f"circuit_{args.behavior}_{args.channel}"
    circuit = json.loads((d / f"{stem}.json").read_text())
    stats = np.load(d / f"{stem}_stats.npz")
    layout = RoutingLayout(circuit["layout"]["L"], circuit["layout"]["H"])

    delta = stats["delta_train"]
    chosen = stats["chosen"].astype(bool)
    eligible = stats["eligible"].astype(bool)

    heatmap(plt, layout, delta, chosen, d / f"{stem}_heatmap.png", args.dpi)
    sparsity(plt, delta[eligible], d / f"{stem}_sparsity.png", args.dpi)
    volcano(plt, delta, stats["p_train"], chosen, eligible,
            d / f"{stem}_volcano.png", args.dpi)
    print(f"[saved] {stem}_heatmap.png, _sparsity.png, _volcano.png in {d}")

    vpath = d / f"verify_{args.behavior}_{args.channel}.json"
    if vpath.is_file():
        v = json.loads(vpath.read_text())
        dose(plt, v["arms"], v["references"]["headroom"],
             d / f"verify_{args.behavior}_{args.channel}_dose.png", args.dpi)
        print(f"[saved] verify_{args.behavior}_{args.channel}_dose.png in {d}")
    else:
        print(f"[skip] no {vpath.name}; run verify_circuit.py for the dose plot")


if __name__ == "__main__":
    main()
