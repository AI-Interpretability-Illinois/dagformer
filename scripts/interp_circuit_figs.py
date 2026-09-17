"""Render one 'circuit card' figure per verified function using the shared
template: left = model grid with the circuit's connections traced; right =
function label, aggregate three-state numbers, and 3 example contexts with
P(target) under suppressed / unchanged / activated.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

TITLES = {
    "target is a open bracket token": ("opening-bracket circuit", "predicting '(' '[' '{'"),
    "target is a close bracket token": ("closing-bracket circuit", "predicting ')' ']' '}'"),
    "target is a quote token": ("quotation-mark circuit", "predicting opening/closing quotes"),
    "target is a pronoun 3rd token": ("third-person-pronoun circuit", "predicting he / she / his / her / him"),
    "target is a sentence end token": ("sentence-end circuit", "predicting . ? ! : at clause ends"),
    "target char-class = whitespace": ("whitespace / newline circuit", "predicting line breaks and spacing"),
}
STREAM_COL = {"q": "#7a5195", "k": "#2f6fc7", "v": "#2a9d8f"}


def draw(rule, rec, out_path, L=12, H=16):
    name, sub = TITLES[rule]
    conns = rec["connections"]; agg = rec["aggregate"]; ex = rec["examples"]
    fig = plt.figure(figsize=(15.5, 7.2))
    ax = fig.add_axes([0.03, 0.06, 0.46, 0.84])
    for l in range(L):
        for h in range(H):
            ax.add_patch(Rectangle((h, l), 0.9, 0.9, facecolor="#eeeeee", edgecolor="white"))
        ax.text(-0.6, l + 0.45, f"L{l}", ha="right", va="center", fontsize=8)
    for h in range(H):
        ax.text(h + 0.45, -0.7, str(h), ha="center", va="center", fontsize=7)
    heads = {}
    for c in conns:
        heads[(c["layer"], c["head"])] = heads.get((c["layer"], c["head"]), 0) + 1
    for (l, h), n in heads.items():
        ax.add_patch(Rectangle((h, l), 0.9, 0.9, facecolor="#c0392b" if n >= 3 else "#e67e22", edgecolor="black", lw=1.2))
    for c in conns:
        l, h, s, src = c["layer"], c["head"], c["stream"], c["src"]
        y_src = -1.6 if src == 0 else (src - 1) + 0.45
        x_src = -1.3 if src == 0 else -0.2
        ax.add_patch(FancyArrowPatch((x_src, y_src), (h + 0.45, l + 0.45), connectionstyle=f"arc3,rad={0.22 if s=='k' else (-0.22 if s=='q' else 0.0)}",
                                     arrowstyle="-|>", mutation_scale=9, lw=0.8 + 2.0 * min(abs(c["z"]) / 8, 1.0), color=STREAM_COL[s], alpha=0.8))
    ax.text(-1.3, -1.6, "embedding", ha="center", va="top", fontsize=7)
    hl = ", ".join(f"L{l}/h{h}" for (l, h) in sorted(heads, key=lambda x: -heads[x]))
    upper_busy = any(l >= 7 and h >= 7 for (l, h) in heads)
    bx, by, bva = (8.2, 11.6, "top") if not upper_busy else (8.2, 0.2, "bottom")
    ax.text(bx, by, f"FUNCTION: {name}\n{sub}\n\nCIRCUIT: top-10 connections\nheads: {hl}\narrows: Q purple / K blue / V green\nfrom source layer to head",
            fontsize=8.2, va=bva, bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999999"))
    ax.set_xlim(-2.2, H); ax.set_ylim(-2.2, L + 0.2); ax.axis("off")
    ax.set_title(f"DAGFormer-300M: {name} traced on the model", fontsize=10.5)

    ax2 = fig.add_axes([0.53, 0.04, 0.46, 0.9]); ax2.axis("off")
    ax2.text(0, 1.0, "Held-out text (200 windows): change in NLL on the function's tokens / on all other tokens", fontsize=9, va="top", fontweight="bold")
    ax2.text(0, 0.95, f"SUPPRESSED (γ=0, connections → layer mean):  {agg['suppress_on']:+.3f} on-function  /  {agg['suppress_off']:+.3f} elsewhere", fontsize=8.8, va="top", color="#c0392b")
    ax2.text(0, 0.91, f"ACTIVATED (γ=2, deviation doubled):            {agg['amplify_on']:+.3f} on-function  /  {agg['amplify_off']:+.3f} elsewhere   (negative = better)", fontsize=8.8, va="top", color="#1f77b4")
    ax2.text(0, 0.84, "Examples: probability the model assigns to the correct next token", fontsize=9, va="top", fontweight="bold")
    y = 0.79
    for e in ex:
        ctx = textwrap.fill(e["context"], 82)
        ax2.text(0, y, ctx + f"  >>{e['target']}<<", fontsize=8, va="top", family="monospace")
        nlines = ctx.count("\n") + 1
        yb = y - 0.028 * nlines - 0.01
        for j, (lab, key, col) in enumerate((("suppressed", "p_suppressed", "#c0392b"), ("unchanged", "p_unchanged", "#555555"), ("activated", "p_activated", "#1f77b4"))):
            p = e[key]
            ax2.add_patch(Rectangle((0.16 + 0.28 * j, yb - 0.022), 0.12 * p, 0.018, color=col))
            ax2.text(0.0 + 0.28 * j, yb - 0.013, f"{lab}: P={p:.2f}", fontsize=7.8, va="center", color=col)
        y = yb - 0.07
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    d = Path("experiments/results/interp/token_effects")
    R = json.load(open(d / "circuit_examples.json"))
    out = Path("experiments/figures/circuit_cards"); out.mkdir(exist_ok=True)
    for rule, rec in R.items():
        fn = out / (TITLES[rule][0].replace(" ", "_").replace("/", "-") + ".png")
        draw(rule, rec, fn)
        print("saved", fn)


if __name__ == "__main__":
    main()
