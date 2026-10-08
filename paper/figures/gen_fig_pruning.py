#!/usr/bin/env python3
"""Figure 3: structured pruning during MathInstruct finetuning.
(a) degradation relative to each model's unpruned finetune at 30/50/70% of heads + MLP channels removed, three sizes;
(b) 150M on a common corpus: dense vs MUDDFormer (per-layer routers) vs DAGFormer, with frozen-router points.
Data: experiments/pruning/results/prune_summary.md, experiments/coherence/results/exp3/prune_summary.md.
    python paper/figures/gen_fig_pruning.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pubstyle import C_DAG, C_DENSE, C_MUDD, FIG_FULL, MUTED, REPO, md_table, panel_label, save, use_pub_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

use_pub_style()
pairs = {}
for r in md_table(os.path.join(REPO, "experiments/pruning/results/prune_summary.md"), "size"):
    if len(r) == 5 and r[0] in ("75m", "150m", "300m"):
        pairs[(r[0], r[1])] = (float(r[2]), float(r[3]))   # (dense NLL, DAGFormer NLL)
exp3 = {}
for r in md_table(os.path.join(REPO, "experiments/coherence/results/exp3/prune_summary.md"), "run"):
    if len(r) == 10:
        exp3[r[0]] = float(r[8])

fig, (ax, bx) = plt.subplots(1, 2, figsize=FIG_FULL)

# ---- (a) relative degradation, three sizes (line style = size, color = family)
sp = [30, 50, 70]
size_ls = {"75m": (":", "75M"), "150m": ("--", "150M"), "300m": ("-", "300M")}
for size, (ls, lab) in size_ls.items():
    b0, d0 = pairs[(size, "s0")]
    db = [-(pairs[(size, f"s{s}")][0] / b0 - 1) * 100 for s in sp]
    dd = [-(pairs[(size, f"s{s}")][1] / d0 - 1) * 100 for s in sp]
    ax.plot(sp, db, color=C_DENSE, ls=ls, lw=1.3, marker="o", ms=3.8, label=f"dense {lab}")
    ax.plot(sp, dd, color=C_DAG, ls=ls, lw=1.8, marker="s", ms=3.8, label=f"DAGFormer {lab}")
ax.set_xticks(sp)
ax.set_xlabel("heads + MLP channels removed (%)")
ax.set_ylabel("degradation vs. unpruned finetune (%)")
ax.legend(ncol=2, loc="lower left", columnspacing=0.9, handlelength=2.3)
panel_label(ax, "(a)")

# ---- (b) 150M: DAGFormer vs per-layer routers (MUDDFormer) vs dense; frozen routers at 50%
sp0 = [0, 30, 50, 70]
for fam, c, m, lw, lab in (("baselinest", C_DENSE, "o", 1.3, "dense"), ("muddformer", C_MUDD, "D", 1.3, "MUDDFormer"),
                           ("dagformerst", C_DAG, "s", 1.8, "DAGFormer")):
    bx.plot(sp0, [exp3[f"150m_{fam}_math_s{s}"] for s in sp0], color=c, lw=lw, marker=m, ms=3.8, label=lab)
for fam, c in (("muddformer", C_MUDD), ("dagformerst", C_DAG)):
    bx.scatter([50], [exp3[f"150m_{fam}_math_s50_frozenrouter"]], color=c, marker="x", s=34, linewidth=1.3, zorder=4)
h, l = bx.get_legend_handles_labels()
h.append(Line2D([], [], color=MUTED, marker="x", ls="none", ms=5, mew=1.3))
l.append("routers frozen (50%)")
bx.set_xticks(sp0)
bx.set_xlabel("heads + MLP channels removed (%)")
bx.set_ylabel("held-out MathInstruct NLL (nats)")
bx.legend(h, l, loc="upper left")
panel_label(bx, "(b)")

fig.tight_layout(w_pad=1.2)
save(fig, "fig_pruning")
