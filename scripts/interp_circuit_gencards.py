"""Template figures with FREE-GENERATION three-state examples for circuits
whose generation statistics move with gamma. Representative examples are
chosen as the sample closest to the state's median statistic (not extremes).
"""
from __future__ import annotations

import json, re, textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

STREAM_COL = {"q": "#7a5195", "k": "#2f6fc7", "v": "#2a9d8f"}
CARDS = {
    "target char-class = whitespace": ("line-break / hard-wrap circuit", "controls where the model inserts line breaks",
                                       lambda t: len(re.findall(r"[a-z,]\n[a-zA-Z]", t)), "mid-sentence line breaks per sample"),
    "target is a quote token": ("quotation circuit", "controls quoted speech / quotation marks",
                                lambda t: t.count('"') + t.count("“") + t.count("”"), "quotation marks per sample"),
    "target is a pronoun 3rd token": ("third-person-pronoun circuit", "controls he / she / his / her / him",
                                      lambda t: len(re.findall(r"\b(he|she|his|her|him|himself|herself)\b", t, flags=re.I)), "3rd-person pronouns per sample"),
    "target is a preposition token": ("preposition circuit", "controls prepositional phrases (of / in / by / with …)",
                                      lambda t: len(re.findall(r"\b(of|in|on|at|by|for|with|from|to|into|about|over)\b", t, flags=re.I)), "prepositions per sample"),
}


def draw_left(ax, conns, name, sub, L=12, H=16):
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
        y_src = -1.6 if src == 0 else (src - 1) + 0.45; x_src = -1.3 if src == 0 else -0.2
        ax.add_patch(FancyArrowPatch((x_src, y_src), (h + 0.45, l + 0.45), connectionstyle=f"arc3,rad={0.22 if s=='k' else (-0.22 if s=='q' else 0.0)}",
                                     arrowstyle="-|>", mutation_scale=9, lw=0.8 + 2.0 * min(abs(c["z"]) / 8, 1.0), color=STREAM_COL[s], alpha=0.8))
    ax.text(-1.3, -1.6, "embedding", ha="center", va="top", fontsize=7)
    hl = ", ".join(f"L{l}/h{h}" for (l, h) in sorted(heads, key=lambda x: -heads[x]))
    # place the label box in the corner (upper-right or lower-right) free of circuit heads
    upper_busy = any(l >= 7 and h >= 7 for (l, h) in heads)
    bx, by, bva = (8.2, 11.6, "top") if not upper_busy else (8.2, 0.2, "bottom")
    ax.text(bx, by, f"FUNCTION: {name}\n{sub}\n\nCIRCUIT: top-10 connections\nheads: {hl}\narrows: Q purple / K blue / V green",
            fontsize=8.2, va=bva, bbox=dict(boxstyle="round", facecolor="white", edgecolor="#999999"))
    ax.set_xlim(-2.2, H); ax.set_ylim(-2.2, L + 0.2); ax.axis("off")
    ax.set_title(f"DAGFormer-300M: {name} traced on the model", fontsize=10.5)


def main():
    d = Path("experiments/results/interp/token_effects")
    src = d / "circuit_generation_full.json"
    G = json.load(open(src if src.exists() else d / "circuit_generation.json"))
    C = {r["rule"]: r for r in json.load(open(d / "circuits_specific.json"))}
    out = Path("experiments/figures/circuit_cards"); out.mkdir(exist_ok=True)
    for rule, (name, sub, stat, statname) in CARDS.items():
        conns = C[rule]["top_connections"][:10]
        states = [("SUPPRESSED (γ = 0)", "0.0", "#c0392b"), ("UNCHANGED (γ = 1)", "1.0", "#555555"), ("ACTIVATED (γ = 4)", "4.0", "#1f77b4")]
        stats = {g: [stat(t) for t in G[rule][g]["examples"]] for _, g, _ in states}
        fig = plt.figure(figsize=(15.5, 7.2))
        draw_left(fig.add_axes([0.03, 0.06, 0.46, 0.84]), conns, name, sub)
        ax2 = fig.add_axes([0.53, 0.04, 0.46, 0.9]); ax2.axis("off")
        ax2.text(0, 1.0, f"Free generation, {len(G[rule]['1.0']['examples'])} samples per state:  {statname} (mean)  =  " + "  /  ".join(f"{np.mean(stats[g]):.2f}" for _, g, _ in states)
                 + f"\n{rule.replace('target ', '').replace('char-class = ', '')}-class token rate: " + " / ".join(f"{100*G[rule][g]['class_rate']:.2f}%" for _, g, _ in states),
                 fontsize=8.8, va="top", fontweight="bold")
        y = 0.88
        for title, g, col in states:
            ex = G[rule][g]["examples"]; vals = stats[g]
            i = int(np.argsort(vals)[len(vals) // 2])   # median sample = representative
            txt = " ".join(ex[i].replace("\n", " ⏎ ").split())[:420]
            ax2.text(0, y, f"{title}   [{statname}: {vals[i]}]", fontsize=9, color=col, va="top", fontweight="bold")
            ax2.text(0, y - 0.045, textwrap.fill(txt, 82), fontsize=7.8, va="top", family="monospace")
            y -= 0.29
        fn = out / (name.replace(" ", "_").replace("/", "-") + "_generation.png")
        fig.savefig(fn, dpi=160, bbox_inches="tight"); plt.close(fig)
        print(f"{name:34s} {statname}: " + " / ".join(f"g{g}: {np.mean(stats[g]):.2f}±{np.std(stats[g]):.2f}" for _, g, _ in states), "->", fn.name)


if __name__ == "__main__":
    main()
