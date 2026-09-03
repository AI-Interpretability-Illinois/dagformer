"""FLOPs x loss Pareto frontier: DAGFormer vs dense baselines vs MUDDFormer.

X = training FLOPs (6 * N_nonembed * D_tokens). Routing adds negligible FLOPs
(routing-mix ~0.08%, predictor ~1.5% at 1B — see METHODOLOGY_flops_loss_pareto.md),
so DAGFormer / dense / MUDD of equal (N, D) share the same 6ND estimate, exactly
as requested.

Y = wikitext bits-per-byte (tokenizer-independent, already in our lm-eval JSONs).
Y axis is INVERTED so "up = better"; the Pareto frontier (upper envelope, dashed)
is mostly DAGFormer; the dominated region BELOW it ("worse than frontier") is
shaded light khaki to separate the two regions.

Pure CPU / matplotlib — safe to run anywhere. Writes a PNG.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Non-embedding params N = 12 * n_layers * d_model^2  (d_ff = 4*d for all scales).
N = {"75m": 12 * 6 * 512**2, "150m": 12 * 8 * 768**2,
     "300m": 12 * 12 * 1024**2, "1b": 12 * 16 * 2048**2}

# (label, kind, N_nonemb, D_tokens, wikitext_BPB). D = steps * 524288 tok/step.
POINTS = [
    # our matched sweep — same (N, D) per scale for dense / DAG / MUDD
    ("75M dense",   "dense",  N["75m"], 1.57e9, 1.6633),
    ("75M DAG",     "dag",    N["75m"], 1.57e9, 1.5828),
    ("75M MuDD",    "mudd",   N["75m"], 1.57e9, 1.6935),
    ("150M dense",  "dense",  N["150m"], 3.15e9, 1.3242),
    ("150M DAG",    "dag",    N["150m"], 3.15e9, 1.2838),
    ("150M MuDD",   "mudd",   N["150m"], 3.15e9, 1.3120),
    ("300M dense",  "dense",  N["300m"], 6.29e9, 1.0968),
    ("300M DAG",    "dag",    N["300m"], 6.29e9, 1.0514),
    ("1B dense",    "dense",  N["1b"],  4.98e9, 1.1554),
    ("1B DAG",      "dag",    N["1b"],  5.24e9, 0.9832),
    # external OLMo-2-1B anchors (same base we build on; different token budgets)
    ("OLMo2 1B\n(~1B tok)",   "olmo2", N["1b"], 1.0e9,  2.0557),
    ("OLMo2 1B\n(21B tok)",   "olmo2", N["1b"], 21.0e9, 0.8843),
    ("OLMo2 1B\n(~4T tok)",   "olmo2", N["1b"], 4.0e12, 0.7177),
    # external open-weight DENSE anchors, evaluated on the same wikitext BPB.
    # N = non-embedding (12*L*d^2); D = training tokens (Pythia 300B; Cerebras 20*N).
    ("Pythia-70m",   "pythia", 12*6*512**2,    300e9, 1.6101),
    ("Pythia-160m",  "pythia", 12*12*768**2,   300e9, 1.1035),
    ("Pythia-410m",  "pythia", 12*24*1024**2,  300e9, 0.8186),
    ("Pythia-1b",    "pythia", 12*16*2048**2,  300e9, 0.7555),
    ("Pythia-1.4b",  "pythia", 12*24*2048**2,  300e9, 0.7274),
    ("Cerebras-111M", "cerebras", 12*10*768**2,   2.22e9, 1.0841),
    ("Cerebras-256M", "cerebras", 12*14*1088**2,  5.12e9, 0.9732),
    ("Cerebras-590M", "cerebras", 12*18*1536**2, 11.80e9, 0.9006),
    ("Cerebras-1.3B", "cerebras", 12*24*2048**2, 26.00e9, 0.8182),
]

STYLE = {
    "dense": dict(color="#7f7f7f", marker="o", label="dense baseline (ours)"),
    "dag":   dict(color="#c81e3a", marker="*", label="DAGFormer (ours)"),
    "mudd":  dict(color="#e07b00", marker="s", label="MuDDFormer (ours, trained)"),
    "olmo2": dict(color="#1f4e79", marker="D", label="OLMo-2-1B (official)"),
    "pythia":   dict(color="#2ca02c", marker="^", label="Pythia (dense, Pile 300B)"),
    "cerebras": dict(color="#9467bd", marker="v", label="Cerebras-GPT (compute-optimal)"),
}


def flops(N_nonemb: float, D: float) -> float:
    return 6.0 * N_nonemb * D


def pareto_frontier(pts):
    """Min-min skyline: keep points with no other point at <= FLOPs and <= BPB."""
    ordered = sorted(pts, key=lambda p: (flops(p[2], p[3]), p[4]))
    front, best = [], float("inf")
    for p in ordered:
        if p[4] < best - 1e-12:
            front.append(p)
            best = p[4]
    return front


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    xs = [flops(p[2], p[3]) for p in POINTS]
    ys = [p[4] for p in POINTS]
    xmin, xmax = min(xs) / 2, max(xs) * 3
    ymin, ymax = min(ys) - 0.08, max(ys) + 0.08  # BPB data range

    front = pareto_frontier(POINTS)
    fx = [flops(p[2], p[3]) for p in front]
    fy = [p[4] for p in front]

    # --- shade the DOMINATED ("worse than frontier") region light khaki ---
    # Piecewise-linear frontier; everything with higher BPB (below, after y-invert)
    # is worse. Fill between the frontier line and the worst-BPB edge.
    import numpy as np
    gx = np.logspace(np.log10(xmin), np.log10(xmax), 400)
    gxl = np.log10(gx)
    fxl = np.log10(fx)
    gy = np.interp(gxl, fxl, fy, left=fy[0], right=fy[-1])  # frontier BPB at each x
    ax.fill_between(gx, gy, ymax, color="#E8DCA0", alpha=0.55, zorder=0,
                    label="_nolegend_")
    ax.text(xmin * 1.5, ymax - 0.05, "worse than frontier\n(dominated region)",
            fontsize=10, color="#8a7a2a", va="top", ha="left", style="italic")

    # --- frontier dashed line ---
    ax.plot(fx, fy, "--", color="#c81e3a", lw=2.0, zorder=3,
            label="Pareto frontier")

    # --- scatter by kind ---
    seen = set()
    for label, kind, n, d, bpb in POINTS:
        s = STYLE[kind]
        lbl = s["label"] if kind not in seen else "_nolegend_"
        seen.add(kind)
        ax.scatter(flops(n, d), bpb, c=s["color"], marker=s["marker"],
                   s=190 if kind == "dag" else 90, zorder=5,
                   edgecolors="black", linewidths=0.5, label=lbl)
        ax.annotate(label, (flops(n, d), bpb), fontsize=7.5,
                    textcoords="offset points", xytext=(6, 5), zorder=6)

    ax.set_xscale("log")
    ax.set_xlabel("Training FLOPs  (6 · N · D,  routing ≈ negligible extra)", fontsize=12)
    ax.set_ylabel("WikiText  bits-per-byte   (lower = better ↑)", fontsize=12)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.invert_yaxis()  # up = better, so the frontier is the TOP envelope
    ax.grid(True, which="both", ls=":", alpha=0.4, zorder=1)
    ax.set_title("DAGFormer sits on the FLOPs–loss Pareto frontier\n"
                 "(dense baselines & MuDDFormer are dominated)", fontsize=13)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    out = "experiments/figures/pareto_frontier_bpb_flops.png"
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    print("\nFrontier points (up the compute axis):")
    for p in front:
        print(f"  {p[0]:<22} FLOPs={flops(p[2],p[3]):.2e}  BPB={p[4]:.4f}  [{p[1]}]")


if __name__ == "__main__":
    main()
