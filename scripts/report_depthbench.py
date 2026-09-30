"""Summarize completed native-checkpoint DepthBench diagnostics and plot them."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from eval_depthbench import write_json


COLORS = {"dense": "#0072B2", "dag": "#D55E00"}
NAMES = {"dense": "Dense", "dag": "DAGFormer"}
CORPORA = {"wikitext_test": "WikiText test", "dolma_eval": "Dolma evaluation cache"}
ARMS = {"bypass_dynamic": "Bypass, dynamic routing",
        "bypass_replay": "Bypass, frozen routing",
        "drop_source_dynamic": "Bypass + source removal, dynamic",
        "drop_source_replay": "Bypass + source removal, frozen"}


def stats(values, draws=2000):
    arr = np.asarray(values)
    if not np.isfinite(arr).all():
        raise ValueError("Statistic contains missing/nonfinite values")
    rng = np.random.default_rng(20260930)
    means = np.stack([arr[rng.integers(len(arr), size=len(arr))].mean(0) for _ in range(draws)])
    return {"mean": arr.mean(0).tolist(), "ci95": np.quantile(means, [.025, .975], axis=0).tolist()}


def masked_mean(values):
    finite = np.isfinite(values)
    count = finite.sum(0)
    return np.divide(np.nansum(values, axis=0), count,
                     out=np.full(values.shape[1:], np.nan), where=count > 0)


def summarize(a):
    L = a["lens"].shape[1] - 1
    s, l = np.indices((L, L))
    late = (s >= math.ceil(L/4)) & (l > s)
    adjacent = a["angular"][:, np.arange(L), np.arange(1, L+1)]
    result = dict(windows=len(a["nll"]), tokens=int(a["valid_tokens"].sum()),
                  nll=stats(a["nll"]), ppl=float(np.exp(a["nll"].mean())),
                  adjacent_angle=stats(adjacent), lens_ce=stats(a["lens"][:, :, 0]),
                  lens_kl=stats(a["lens"][:, :, 1]), lens_top5=stats(a["lens"][:, :, 2]),
                  last_quarter_readout_nll_gain=stats(a["lens"][:, 3*L//4, 0] - a["nll"]),
                  update_norm=stats(a["update_norm"]), branch_norm=stats(a["branch_norm"]), arms={})
    for arm in ARMS:
        if f"{arm}/skip_nll" not in a:
            continue
        delta = a[f"{arm}/skip_nll"] - a["nll"][:, None]
        matrix = masked_mean(a[f"{arm}/causal"])
        result["arms"][arm] = dict(
            skip_nll_increase=stats(delta),
            mean_late_skip_nll_increase=stats(delta[:, math.ceil(L/4):].mean(1)),
            last_quarter_skip_nll_increase=stats(delta[:, 3*L//4:].mean(1)),
            mean_late_causal=stats(a[f"{arm}/causal"][:, late].mean(1)),
            late_fraction_above_045=float((matrix[late] > .45).mean()),
            late_pairs=int(late.sum()),
            mean_late_raw_change=stats(a[f"{arm}/change_norm"][:, late].mean(1)),
            mean_late_branch_causal=stats(a[f"{arm}/branch_causal"][:, late].mean(1)),
            mean_late_raw_branch_change=stats(a[f"{arm}/branch_change_norm"][:, late].mean(1)))
    return result


def line(ax, x, values, label, color, marker=None, linestyle="-"):
    values = np.asarray(values)
    summary = stats(values)
    ax.plot(x, summary["mean"], color=color, marker=marker, ms=3.5,
            lw=1.6, label=label, linestyle=linestyle)
    ax.fill_between(x, *summary["ci95"], color=color, alpha=.13, linewidth=0)


def export(fig, path):
    for extension in ("png", "svg", "pdf"):
        target = path.with_suffix("." + extension)
        fig.savefig(target, dpi=180, bbox_inches="tight")
        if extension == "svg":
            target.write_text("\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n")
    plt.close(fig)


def figures(data, corpus, out):
    L = data["dense"]["lens"].shape[1] - 1
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.6), layout="constrained")
    x = np.arange(1, L+1)
    for kind, a in data.items():
        color, marker = COLORS[kind], "o" if kind == "dense" else "s"
        line(axes[0, 0], x, a["angular"][:, np.arange(L), x], NAMES[kind], color, marker)
        line(axes[0, 1], np.arange(L+1), a["lens"][:, :, 0], NAMES[kind], color, marker)
        line(axes[1, 0], x, a["bypass_dynamic/skip_nll"]-a["nll"][:, None], NAMES[kind], color, marker)
    dag = data["dag"]
    for arm, color, style in zip(ARMS, [COLORS["dag"], "#CC79A7", "#009E73", "#222222"], ["-", "--", "-.", ":"]):
        line(axes[1, 1], x, dag[f"{arm}/skip_nll"] - dag["nll"][:, None], ARMS[arm], color, linestyle=style)
    details = [
        ("(a) Representation change per block", "Block", "Angular distance / pi"),
        ("(b) Predictions decoded at each depth", "Depth state (0 = embedding)", "Readout NLL (nats/token)"),
        ("(c) Loss after bypassing each block", "Bypassed block", "NLL increase (nats/token)"),
        ("(d) DAG: effect of routing and source access", "Bypassed block", "NLL increase (nats/token)")]
    for ax, (title, xlabel, ylabel) in zip(axes.flat, details):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7.7, frameon=False)
    fig.suptitle(f"{CORPORA.get(corpus, corpus)}: matched step-9000 checkpoints · {len(dag['nll'])} × 1024-token windows", fontsize=12)
    export(fig, out / f"overview_{corpus}")

    panels = [("Dense: bypass", data["dense"]["bypass_dynamic/causal"]),
              ("DAG: bypass, dynamic", dag["bypass_dynamic/causal"]),
              ("DAG: bypass, frozen routing", dag["bypass_replay/causal"]),
              ("DAG: bypass + source removal", dag["drop_source_dynamic/causal"])]
    matrices = [masked_mean(a) for _, a in panels]
    vmax = max(np.nanmax(a) for a in matrices)
    fig, axes = plt.subplots(1, 4, figsize=(13.4, 3.7), layout="constrained")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#eeeeee")
    for ax, (title, _), matrix in zip(axes, panels, matrices):
        im = ax.imshow(matrix, vmin=0, vmax=vmax, cmap=cmap, origin="upper")
        ax.set(title=title, xlabel="Affected block", ylabel="Bypassed block")
        ticks = np.arange(0, L, 2)
        ax.set_xticks(ticks, ticks+1)
        ax.set_yticks(ticks, ticks+1)
    fig.colorbar(im, ax=axes, shrink=.7, label="Relative downstream update change")
    fig.suptitle(f"{CORPORA.get(corpus, corpus)}: causal dependencies (shared linear color scale)", fontsize=12)
    export(fig, out / f"causal_{corpus}")


def table_exports(data, corpus, out):
    rows = []
    for kind, a in data.items():
        L = a["lens"].shape[1]-1
        for block in range(L):
            rows.append(dict(corpus=corpus, model=kind, block=block+1,
                adjacent_angle=float(a["angular"][:, block, block+1].mean()),
                readout_nll=float(a["lens"][:, block+1, 0].mean()),
                readout_kl=float(a["lens"][:, block+1, 1].mean()),
                top5_overlap=float(a["lens"][:, block+1, 2].mean()),
                bypass_nll_increase=float((a["bypass_dynamic/skip_nll"][:, block]-a["nll"]).mean()),
                original_update_norm=float(a["update_norm"][:, block].mean())))
    with (out/f"layers_{corpus}.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--directory", type=Path, default=Path("experiments/results/depthbench_20260930"))
    args = ap.parse_args()
    out = args.directory
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9,
                         "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False})
    metadata = {kind: json.loads((out/f"metadata_{kind}.json").read_text()) for kind in NAMES}
    if not all(m["complete"] for m in metadata.values()):
        raise ValueError("Both models must finish before comparative reporting")
    if metadata["dense"]["checkpoint_step"] != metadata["dag"]["checkpoint_step"]:
        raise ValueError("Checkpoint steps do not match")
    if metadata["dense"]["protocol"]["caches"] != metadata["dag"]["protocol"]["caches"]:
        raise ValueError("Models used different caches")
    summary = {"metadata": metadata, "corpora": {}}
    for corpus in [entry["corpus"] for entry in metadata["dense"]["results"]]:
        data = {}
        for kind in NAMES:
            with np.load(out/f"{kind}_{corpus}.npz") as cache:
                data[kind] = {key: cache[key] for key in cache.files}
            assert int(data[kind]["completed"]) == len(data[kind]["nll"])
            assert np.allclose(data[kind]["lens"][:, -1, 0], data[kind]["nll"], atol=2e-6, rtol=0)
            assert np.all(data[kind]["lens"][:, -1, 1] == 0)
            assert np.all(data[kind]["lens"][:, -1, 2] == 1)
        assert np.array_equal(data["dense"]["valid_tokens"], data["dag"]["valid_tokens"])
        result = {kind: summarize(a) for kind, a in data.items()}
        result["paired_dag_minus_dense"] = {"nll": stats(data["dag"]["nll"]-data["dense"]["nll"]),
            "bypass_nll_increase": stats((data["dag"]["bypass_dynamic/skip_nll"]-data["dag"]["nll"][:, None])-
                                         (data["dense"]["bypass_dynamic/skip_nll"]-data["dense"]["nll"][:, None]))}
        L = data["dense"]["lens"].shape[1]-1
        s, l = np.indices((L, L))
        late = (s >= math.ceil(L/4)) & (l > s)
        result["paired_dag_minus_dense"]["mean_late_causal"] = stats(
            data["dag"]["bypass_dynamic/causal"][:, late].mean(1) -
            data["dense"]["bypass_dynamic/causal"][:, late].mean(1))
        result["dag"]["source_removal_minus_bypass_late_nll"] = stats(
            (data["dag"]["drop_source_dynamic/skip_nll"] -
             data["dag"]["bypass_dynamic/skip_nll"])[:, math.ceil(L/4):].mean(1))
        summary["corpora"][corpus] = result
        figures(data, corpus, out)
        table_exports(data, corpus, out)
    write_json(out/"summary.json", summary)
    for name, result in summary["corpora"].items():
        print(name)
        for kind in NAMES:
            a = result[kind]
            print(kind, "nll", a["nll"], "late bypass", a["arms"]["bypass_dynamic"]["mean_late_skip_nll_increase"],
                  "late causal", a["arms"]["bypass_dynamic"]["mean_late_causal"])


if __name__ == "__main__":
    main()
