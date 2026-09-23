"""Summarise and plot a prune-during-finetune sweep: baseline vs DAGFormer.

Reads every ``<ckpt_root>/<run>/summary.json`` + ``trajectory.json`` written by
scripts/prune_finetune.py, groups runs by (size, family, tag) and produces

    <out>/prune_pareto.png        domain NLL vs remaining backbone params
                                  (one panel per size; baseline vs dagformer)
    <out>/prune_trajectory.png    domain NLL over training for the matched
                                  pair at each sparsity, prune events marked
    <out>/prune_summary.md        the table behind the figures

Usage:
    python scripts/plot_prune_pareto.py \
        --ckpt-root /work/hdd/bfqt/xiaocong/dagformer_pruning/checkpoints \
        --out experiments/pruning/results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots 1 and 2 of the reference palette (fixed order, never cycled)
COLORS = {"baseline": "#2a78d6", "dagformer": "#eb6834"}
MARKERS = {"baseline": "o", "dagformer": "s"}
RUN_RE = re.compile(r"^(?P<size>\d+m)_(?P<family>baseline|dagformer)_(?P<domain>[a-z0-9]+)_(?P<tag>.+)$")


def load_runs(root: str) -> list[dict]:
    runs = []
    for summ in sorted(glob.glob(os.path.join(root, "*", "summary.json"))):
        run_dir = os.path.dirname(summ)
        m = RUN_RE.match(os.path.basename(run_dir))
        if not m:
            continue
        s = json.load(open(summ))
        traj = json.load(open(os.path.join(run_dir, "trajectory.json")))
        cfg = s["config"]
        extra = s.get("extra_params")
        if extra is None:                      # older runs: count the saved predictor/routing state
            extra = 0
            if s["model"] == "fourway":
                ck = torch.load(os.path.join(run_dir, "final", "checkpoint.pt"), map_location="cpu",
                                weights_only=False)
                extra = sum(v.numel() for v in ck["predictor_state_dict"].values()) + \
                    sum(v.numel() for v in ck.get("routing_state_dict", {}).values())
        runs.append({
            **m.groupdict(), "name": os.path.basename(run_dir),
            "target": cfg["target_sparsity"], "units": ",".join(cfg["prune_units"]),
            "importance": cfg["importance"],
            "params_total": s["block_params_total"], "params_remaining": s["block_params_remaining"],
            "sparsity": 1 - s["block_params_remaining"] / s["block_params_total"],
            "base_params_remaining": s["base_params_remaining"],
            "total_params_remaining": s["base_params_remaining"] + extra,
            "total_params_unpruned": s["base_params_total"] + extra,
            "extra_params": extra,
            "init_nll": s["initial_domain_nll"], "final_nll": s["final_domain_nll"],
            "final_general": s.get("final_general_nll"),
            "traj": traj,
        })
    return runs


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c3c2b7")
    ax.grid(True, axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=9)


def plot_pareto(runs: list[dict], out: str, xkey: str = "params_remaining",
                x0key: str = "params_total", xlabel: str = "transformer-block params remaining (M)") -> None:
    sizes = sorted({r["size"] for r in runs}, key=lambda s: int(s[:-1]))
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.2 * len(sizes), 3.6), squeeze=False)
    for ax, size in zip(axes[0], sizes):
        for fam in ("baseline", "dagformer"):
            pts = sorted([r for r in runs if r["size"] == size and r["family"] == fam
                          and r["units"] == "head,neuron" and r["importance"] == "taylor"
                          and not r["tag"].endswith("frozenpred")],
                         key=lambda r: r[xkey])
            if not pts:
                continue
            xs = [r[xkey] / 1e6 for r in pts]
            ys = [r["final_nll"] for r in pts]
            ax.plot(xs, ys, marker=MARKERS[fam], markersize=7, linewidth=2, color=COLORS[fam],
                    label=fam, markeredgecolor="#fcfcfb", markeredgewidth=1.5)
            for r, x, y in zip(pts, xs, ys):
                ax.annotate(f"{r['sparsity']:.0%}", (x, y), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=7.5, color="#52514e")
            # unpruned starting point
            ax.plot([pts[0][x0key] / 1e6], [pts[0]["init_nll"]], marker=MARKERS[fam],
                    markersize=7, color=COLORS[fam], markerfacecolor="none", linestyle="none")
        ax.set_title(f"{size}", fontsize=10, color="#0b0b0b", loc="left")
        ax.set_xlabel(xlabel, fontsize=9, color="#52514e")
        style(ax)
    axes[0][0].set_ylabel("domain eval NLL (lower is better)", fontsize=9, color="#52514e")
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle("Pruned during math finetuning. Hollow marker = before finetuning/pruning;\n"
                 "label = block param sparsity (embedding excluded)", fontsize=9, color="#52514e",
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_trajectory(runs: list[dict], out: str) -> None:
    keys = sorted({(r["size"], r["tag"]) for r in runs}, key=lambda k: (int(k[0][:-1]), k[1]))
    if not keys:
        return
    ncol = min(3, len(keys))
    nrow = (len(keys) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False)
    for ax, (size, tag) in zip(axes.flat, keys):
        for fam in ("baseline", "dagformer"):
            rr = [r for r in runs if r["size"] == size and r["tag"] == tag and r["family"] == fam]
            if not rr:
                continue
            tr = rr[0]["traj"]
            steps = [t["step"] for t in tr]
            nll = [t["eval/domain_nll"] for t in tr]
            ax.plot(steps, nll, linewidth=2, color=COLORS[fam], label=fam)
            post = [t for t in tr if t["tag"] == "post_prune"]
            ax.plot([t["step"] for t in post], [t["eval/domain_nll"] for t in post], linestyle="none",
                    marker="|", markersize=8, color=COLORS[fam])
        ax.set_title(f"{size}  {tag}", fontsize=10, loc="left", color="#0b0b0b")
        ax.set_xlabel("finetune step", fontsize=9, color="#52514e")
        style(ax)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    axes[0][0].set_ylabel("domain eval NLL", fontsize=9, color="#52514e")
    axes[0][0].legend(frameon=False, fontsize=9)
    fig.suptitle("Ticks mark evals taken right after a pruning step (before recovery)",
                 fontsize=9, color="#52514e", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def write_table(runs: list[dict], out: str) -> None:
    lines = ["| run | units | importance | target | block sparsity | block params remaining | total params remaining | NLL before | NLL final | general NLL final |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda r: (int(r["size"][:-1]), r["tag"], r["family"])):
        g = f"{r['final_general']:.4f}" if r["final_general"] is not None else "-"
        lines.append(f"| {r['name']} | {r['units']} | {r['importance']} | {r['target']} | "
                     f"{r['sparsity']:.3f} | {r['params_remaining']/1e6:.1f}M | {r['total_params_remaining']/1e6:.1f}M | {r['init_nll']:.4f} | "
                     f"{r['final_nll']:.4f} | {g} |")
    # matched-pair deltas
    pairs = defaultdict(dict)
    for r in runs:
        pairs[(r["size"], r["tag"])][r["family"]] = r
    lines += ["", "| size | tag | baseline final | dagformer final | delta (baseline - dagformer) |", "|---|---|---|---|---|"]
    for (size, tag), fams in sorted(pairs.items(), key=lambda kv: (int(kv[0][0][:-1]), kv[0][1])):
        if "baseline" in fams and "dagformer" in fams:
            b, d = fams["baseline"]["final_nll"], fams["dagformer"]["final_nll"]
            lines.append(f"| {size} | {tag} | {b:.4f} | {d:.4f} | {b - d:+.4f} |")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", default="/work/hdd/bfqt/xiaocong/dagformer_pruning/checkpoints")
    p.add_argument("--out", default="experiments/pruning/results")
    args = p.parse_args()
    runs = load_runs(args.ckpt_root)
    if not runs:
        raise SystemExit(f"no finished runs under {args.ckpt_root}")
    os.makedirs(args.out, exist_ok=True)
    plot_pareto(runs, os.path.join(args.out, "prune_pareto.png"))
    plot_pareto(runs, os.path.join(args.out, "prune_pareto_total.png"), xkey="total_params_remaining",
                x0key="total_params_unpruned",
                xlabel="total params remaining incl. embedding + predictor (M)")
    plot_trajectory(runs, os.path.join(args.out, "prune_trajectory.png"))
    write_table(runs, os.path.join(args.out, "prune_summary.md"))
    print(open(os.path.join(args.out, "prune_summary.md")).read())


if __name__ == "__main__":
    main()
