"""Shared publication style for the paper's data figures (ACL two-column, Times text).

Colors are the Okabe-Ito colorblind-safe set with a fixed entity assignment (never cycled):
DAGFormer (ours) = vermillion, dense baseline = neutral gray, modular variant = blue,
MUDDFormer / per-layer routers = bluish green. Figure 1 (TikZ) uses the same hex values.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.dirname(os.path.abspath(__file__))

C_DAG, C_DENSE, C_MOD, C_MUDD = "#D55E00", "#6E6E6E", "#0072B2", "#009E73"
INK, MUTED = "#222222", "#6A6A6A"
FIG_FULL = (6.8, 2.6)      # ACL text width
FIG_SINGLE = (3.3, 2.5)    # ACL column width


def use_pub_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5, "legend.fontsize": 7.3,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.labelcolor": INK, "text.color": INK,
        "axes.grid": True, "grid.alpha": 0.15, "grid.linestyle": "-", "grid.linewidth": 0.6,
        "lines.linewidth": 1.6, "lines.markersize": 4.5,
        "legend.frameon": False, "legend.handletextpad": 0.4, "legend.borderaxespad": 0.2,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def panel_label(ax, text):
    """Panel letter only; the caption carries the description (no titles inside figures)."""
    ax.text(-0.02, 1.03, text, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300)
    print("wrote", name + ".pdf/.png")


def md_table(path, first_col):
    rows = []
    for line in open(path):
        if line.startswith("| " + first_col) or not line.startswith("|") or set(line.strip()) <= {"|", "-", " "}:
            continue
        rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    return rows


def md_col(path, name):
    hdr = [c.strip() for c in open(path).readline().strip().strip("|").split("|")]
    return hdr.index(name)
