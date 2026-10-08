#!/usr/bin/env python3
"""Figure 2: scaling of DAGFormer against matched dense OLMo-2.
(a) final WikiText-2 loss vs backbone parameters with per-family power-law fits (16-21 tokens/param runs);
(b) the same checkpoints plus the 1B runs vs training FLOPs with the routing cost charged;
(c) loss along training vs tokens seen for the four matched pairs, each on its own held-out cache.
Appendix figure fig_scaling_modular: (a) and (b) with the modular variant added (Appendix E).
Data: experiments/scaling/{scaling_fits.json, runs_table.md, collected_*.json}.   python paper/figures/gen_fig_scaling.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pubstyle import C_DAG, C_DENSE, C_MOD, FIG_FULL, INK, MUTED, OUT, REPO, md_col, md_table, panel_label, save, use_pub_style  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

use_pub_style()
fits = json.load(open(os.path.join(REPO, "experiments/scaling/scaling_fits.json")))
TABLE = os.path.join(REPO, "experiments/scaling/runs_table.md")
runs = {r[0]: r for r in md_table(TABLE, "run")}
iN, iC, iW = md_col(TABLE, "N backbone (M)"), md_col(TABLE, "train C (PF)"), md_col(TABLE, "common wikitext2")
# entity -> (color, marker, line width, label); ours drawn heaviest
FAM_ALL = {"dense": (C_DENSE, "o", 1.3, "dense OLMo-2"),
           "corrected": (C_DAG, "s", 2.0, "DAGFormer"),
           "modular": (C_MOD, "^", 1.1, "DAGFormer, modular variant")}
FAM = {k: v for k, v in FAM_ALL.items() if k != "modular"}   # the body figure; the modular variant is in the appendix figure
ANN = dict(fontsize=7, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=0, shrinkB=2.5))

# training curves (own-corpus held-out cache) of the four matched pairs: (size label, dense run, DAGFormer run)
PAIRS = [("75M", "timan_75m_dense", "timan_75m_corrected"), ("150M", "timan_150m_dense", "timan_150m_corrected"),
         ("300M", "300m_dense", "300m_fourway_corrected"), ("1B", "1b_dense_10b", "1b_fourway_corrected_10b")]
curves = {}
for fn in ("collected_delta.json", "collected_timan1.json", "collected_timan108.json"):
    path = os.path.join(REPO, "experiments/scaling", fn)
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    rs = d.get("runs", d)
    for r in (rs if isinstance(rs, list) else rs.values()):
        if isinstance(r, dict) and r.get("curve"):
            curves[r["name"]] = [(c["tokens"], c["eval_nll"]) for c in r["curve"] if c.get("eval_nll") is not None]

fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(FIG_FULL[0], 2.45))

# ---- (a) loss vs size, fits over the 16-21 tokens/param runs; 1B runs hollow (not fitted)
Ns = np.logspace(np.log10(6e7), np.log10(1.5e9), 200)
for fam, (c, m, lw, lab) in FAM.items():
    pts, f = fits["L_of_N"][fam]["points"], fits["L_of_N"][fam]["fit"]
    ax.plot(Ns, f["E"] + f["A"] * Ns ** (-f["alpha"]), color=c, lw=lw, zorder=2)
    ax.scatter([p["N"] for p in pts], [p["L"] for p in pts], color=c, marker=m, s=24, zorder=3,
               edgecolor="white", linewidth=0.6, label=lab)
for name, fam in (("1b_dense_5b", "dense"), ("1b_dense_10b", "dense"),
                  ("1b_fourway_corrected_5b", "corrected"), ("1b_fourway_corrected_10b", "corrected")):
    r = runs[name]
    if r[iW] == "-":
        continue
    c, m, _, _ = FAM[fam]
    ax.scatter(float(r[iN]) * 1e6, float(r[iW]), facecolor="white", edgecolor=c, marker=m, s=28, zorder=3, linewidth=1.1)
ax.set_xscale("log")
ax.set_xlabel("backbone parameters $N$")
ax.set_ylabel("held-out NLL on WikiText-2 (nats)")
panel_label(ax, "(a)")

# ---- (b) loss vs training FLOPs, all final checkpoints, routing cost included
Cs = np.logspace(17.7, 20.4, 200)
for fam, (c, m, lw, lab) in FAM.items():
    pts, f = np.array(fits["L_of_C"][fam]["points"]), fits["L_of_C"][fam]["fit"]
    bx.plot(Cs, f["E"] + f["A"] * Cs ** (-f["alpha"]), color=c, lw=lw, zorder=2)
    bx.scatter(pts[:, 0], pts[:, 1], color=c, marker=m, s=24, zorder=3, edgecolor="white", linewidth=0.6, label=lab)
rd, rc, rc10 = runs["1b_dense_10b"], runs["1b_fourway_corrected_5b"], runs["1b_fourway_corrected_10b"]
bx.annotate("1B DAGF.,\n5B tok", (float(rc[iC]) * 1e15, float(rc[iW])), xytext=(-14, -17), textcoords="offset points",
            ha="right", va="center", multialignment="left", **ANN)
bx.annotate("1B dense, 10B tok", (float(rd[iC]) * 1e15, float(rd[iW])), xytext=(6, 11), textcoords="offset points",
            ha="left", va="center", **ANN)
if rc10[iW] != "-":
    bx.annotate("1B DAGF., 10B tok", (float(rc10[iC]) * 1e15, float(rc10[iW])), xytext=(4, -19), textcoords="offset points",
                ha="left", va="center", **ANN)
bx.set_xscale("log")
bx.set_xlim(4e17, 4e21)
bx.set_ylim(2.55, 5.45)
bx.set_xlabel("training FLOPs (routing cost included)")
panel_label(bx, "(b)")

# ---- (c) loss along training vs tokens seen, matched pairs on their own held-out caches
for lab, dn, rn in PAIRS:
    if dn not in curves or rn not in curves:
        continue
    xd, yd = zip(*curves[dn]); xr, yr = zip(*curves[rn])
    cx.plot(xd, yd, color=C_DENSE, lw=1.2, ls="--", zorder=2)
    cx.plot(xr, yr, color=C_DAG, lw=1.6, zorder=3)
    dy = {"75M": 2, "150M": 0, "300M": 5, "1B": -7}[lab]
    cx.annotate(lab, (xr[-1], yr[-1]), xytext=(3, dy), textcoords="offset points", fontsize=7, color=INK, va="center")
cx.set_xscale("log")
cx.set_xlim(2.2e8, 2.2e10)
cx.set_ylim(1.65, 5.7)
cx.set_xlabel("training tokens $D$")
cx.set_ylabel("held-out NLL, own corpus (nats)")
panel_label(cx, "(c)")

# shared legend above the panels (no in-panel legends; the panels are narrow)
H = [Line2D([], [], color=C_DENSE, marker="o", ls="--", lw=1.2, ms=4.5, markeredgecolor="white", mew=0.6),
     Line2D([], [], color=C_DAG, marker="s", ls="-", lw=1.8, ms=4.5, markeredgecolor="white", mew=0.6),
     Line2D([], [], marker="s", ls="none", markerfacecolor="white", markeredgecolor=MUTED, ms=5, mew=1.0)]
L = ["dense OLMo-2", "DAGFormer", "1B runs, 4–8 tok/param (not fitted)"]
fig.legend(H, L, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.6, handlelength=2.2)

fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.92))
save(fig, "fig_scaling")


# ---- appendix figure: (a) and (b) with the modular variant (Appendix E)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(FIG_FULL[0] * 0.72, 2.35))
for fam, (c, m, lw, lab) in FAM_ALL.items():
    pts, f = fits["L_of_N"][fam]["points"], fits["L_of_N"][fam]["fit"]
    ax.plot(Ns, f["E"] + f["A"] * Ns ** (-f["alpha"]), color=c, lw=lw, zorder=2)
    ax.scatter([p["N"] for p in pts], [p["L"] for p in pts], color=c, marker=m, s=24, zorder=3, edgecolor="white", linewidth=0.6)
    pts, f = np.array(fits["L_of_C"][fam]["points"]), fits["L_of_C"][fam]["fit"]
    bx.plot(Cs, f["E"] + f["A"] * Cs ** (-f["alpha"]), color=c, lw=lw, zorder=2)
    bx.scatter(pts[:, 0], pts[:, 1], color=c, marker=m, s=24, zorder=3, edgecolor="white", linewidth=0.6)
ax.set_xscale("log"); ax.set_xlim(6e7, 1.0e9); ax.set_xlabel("backbone parameters $N$"); ax.set_ylabel("held-out NLL on WikiText-2 (nats)")
bx.set_xscale("log"); bx.set_xlim(4e17, 2e20); bx.set_xlabel("training FLOPs (routing cost included)")
panel_label(ax, "(a)"); panel_label(bx, "(b)")
H = [Line2D([], [], color=c, marker=m, ls="--" if fam == "dense" else "-", lw=lw, ms=4.5, markeredgecolor="white", mew=0.6)
     for fam, (c, m, lw, lab) in FAM_ALL.items()]
fig.legend(H, [v[3] for v in FAM_ALL.values()], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0), columnspacing=1.6, handlelength=2.2)
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.88))
save(fig, "fig_scaling_modular")
