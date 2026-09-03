"""Confounder-free method comparison: ALL methods on identical data.

Every point here is trained on the SAME dolma mmap shards (same data, tokenizer,
token count, and training setup) and evaluated identically on wikitext BPB. This
removes the data confound that made the reproductions (also mmap) look like they
beat DAGFormer when compared against the original stream-trained DAGFormer.

Result: on matched data, DAGFormer has the lowest BPB at every scale; DenseFormer
and Hyper-Connections do NOT beat it (they are ~= or worse than the dense baseline
at these small scales). Y = wikitext BPB (lower = better).
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N = {"75M": 12*6*512**2, "150M": 12*8*768**2, "300M": 12*12*1024**2}
# tokens actually consumed: 75M/150M ran to completion (3000/6000 steps);
# the 300M points are all read at step 9000 (HC at 8000 — see note below).
D = {"75M": 1.57e9, "150M": 3.15e9, "300M": 9000 * 524288}
def flops(sc): return 6.0 * N[sc] * D[sc]

# all trained on the SAME dolma mmap; None = not yet evaluated
BPB = {
    "dense baseline":       {"75M": 1.3931, "150M": 1.1960, "300M": 1.0715},
    "DAGFormer (ours)":     {"75M": 1.3470, "150M": 1.1523, "300M": 1.0224},
    "DenseFormer":          {"75M": 1.4070, "150M": 1.1900, "300M": 1.0664},
    # HC 300M only reached step 8000 (fewer tokens) — plotted at its own FLOPs.
    "Hyper-Connections":    {"75M": 1.4042, "300M": 1.0772},
}
# per-method FLOPs overrides where token count differs from the column default
FLOPS_OVERRIDE = {("Hyper-Connections", "300M"): 6.0 * N["300M"] * (8000 * 524288)}
STYLE = {
    "dense baseline":    dict(color="#7f7f7f", marker="o", ls="-",  ms=8,  lw=2),
    "DAGFormer (ours)":  dict(color="#c81e3a", marker="*", ls="-",  ms=16, lw=2.5),
    "DenseFormer":       dict(color="#e07b00", marker="s", ls="--", ms=8,  lw=1.8),
    "Hyper-Connections": dict(color="#1f77b4", marker="^", ls="--", ms=8,  lw=1.8),
}


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 5.6))
    for method, pts in BPB.items():
        order = [sc for sc in ["75M", "150M", "300M"] if sc in pts]
        xs = [FLOPS_OVERRIDE.get((method, sc), flops(sc)) for sc in order]
        ys = [pts[sc] for sc in order]
        ax.plot(xs, ys, label=method, **STYLE[method])
    for sc in ["75M", "150M", "300M"]:
        ax.annotate(sc, (flops(sc), BPB["DAGFormer (ours)"][sc]),
                    fontsize=9, xytext=(0, -16), textcoords="offset points", ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("Training FLOPs  (6·N·D,  identical across methods)", fontsize=11)
    ax.set_ylabel("WikiText  bits-per-byte   (lower = better)", fontsize=11)
    ax.set_title("Confounder-free method comparison (ALL on identical dolma mmap)\n"
                 "DAGFormer has the lowest BPB at every scale (75M / 150M / 300M)",
                 fontsize=11.5)
    ax.text(0.02, 0.03, "300M read at step 9000 (Hyper-Connections at 8000 — fewer tokens)",
            transform=ax.transAxes, fontsize=7.5, color="#555", style="italic")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=10, loc="upper right")
    fig.tight_layout()
    out = "experiments/figures/matched_methods_bpb.png"
    import os
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for sc in ["75M", "150M", "300M"]:
        row = {m: p.get(sc) for m, p in BPB.items()}
        best = min((v for v in row.values() if v is not None))
        winner = [m for m, v in row.items() if v == best][0]
        print(f"  {sc}: " + " | ".join(f"{m.split()[0]} {v:.4f}" for m, v in row.items() if v is not None)
              + f"   → best: {winner.split()[0]}")


if __name__ == "__main__":
    main()
