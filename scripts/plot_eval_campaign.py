"""Plot the campaign's measured paired effects; no transcribed result values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {"pred": "#0072B2", "corr": "#E69F00", "both": "#CC79A7"}
MARKERS = {"pred": "o", "corr": "s", "both": "D"}
CHANNELS = {"pred": "Predictor", "corr": "Correction", "both": "Both channels"}
TASK_NAMES = {
    "lambada_openai": "LAMBADA", "hellaswag": "HellaSwag", "piqa": "PIQA",
    "arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge", "winogrande": "WinoGrande",
    "openbookqa": "OpenBookQA", "sciq": "SciQ", "boolq": "BoolQ*", "mathqa": "MathQA",
    "commonsense_qa": "CommonsenseQA*", "social_iqa": "SocialIQA", "wikitext": "WikiText",
    "gsm8k_bpb": "GSM8K question + answer",
}


def read(path):
    return json.loads(path.read_text())


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e4e4e4", linewidth=.6)
    ax.set_axisbelow(True)


def save(fig, out, name):
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(out / f"{name}.{suffix}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(name, flush=True)


def ordinary(root, out):
    pairs = read(root / "standard_matched/paired_summary.json")["pairs"]
    sizes = [s for s in ("75m", "150m", "300m") if s in pairs]
    tasks = list(pairs[sizes[-1]]["tasks"])
    accuracy = [t for t in tasks if pairs[sizes[-1]]["tasks"][t]["metric"].split(",")[0] != "bits_per_byte"]
    likelihood = [t for t in tasks if t not in accuracy]
    fig, axes = plt.subplots(2, 1, figsize=(9.3, 8.6), gridspec_kw={"height_ratios": [6, 1.5]})
    palette = ["#56B4E9", "#0072B2", "#222222"]
    markers = ["o", "s", "^"]
    for ax, group, factor in ((axes[0], accuracy, 100), (axes[1], likelihood, 1)):
        for i, size in enumerate(sizes):
            for y, task in enumerate(group):
                record = pairs[size]["tasks"][task]
                value = factor * record["delta_dagformer_better"]
                lo, hi = factor * np.array(record["paired_bootstrap_95ci"])
                ax.errorbar(value, y + (i - 1) * .21, xerr=[[value - lo], [hi - value]],
                            fmt=markers[i], color=palette[i], markersize=5, capsize=2, linewidth=1.1)
        ax.set_yticks(range(len(group)), [TASK_NAMES.get(t, t) for t in group])
        ax.invert_yaxis()
        ax.axvline(0, color="#777777", linewidth=.9, linestyle="--")
        ax.grid(axis="x", color="#e4e4e4", linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=.10)
    axes[0].set_title("(a) Accuracy endpoints", loc="left", fontweight="bold")
    axes[0].set_xlabel("Accuracy gain (percentage points)")
    axes[1].set_title("(b) Likelihood endpoints", loc="left", fontweight="bold")
    axes[1].set_xlabel("Reduction in bits per byte")
    handles = [Line2D([], [], marker=m, color=c, linestyle="none", label=s.upper())
               for s, m, c in zip(sizes, markers, palette)]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.6, 1.0))
    fig.suptitle("Paired benchmark differences: DAGFormer versus baseline", y=1.025, fontsize=13)
    fig.text(.03, -.02, "95% document-bootstrap intervals; fixed checkpoints; positive favors DAGFormer.\n"
             "* Strong label bias: all models are below the best constant-label score on these tasks.", fontsize=9)
    fig.tight_layout(h_pad=2.1, rect=[0, 0, 1, .965])
    save(fig, out, "ordinary_paired")


def routing(root, out):
    names = ["75m", "150m", "300m", "300m_step10500"]
    labels = ["75M\nstep 3000", "150M\nstep 6000", "300M\nstep 9000", "300M\nstep 10500"]
    records = [read(root / "routing_dependence" / f"{name}.json") for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
    configurations = [(axes[0], "pred_position/corr_dynamic", "Predictor → position table", "#0072B2", "o", 0),
                      (axes[1], "pred_dynamic/corr_position", "Correction → position table", "#E69F00", "s", -.07),
                      (axes[1], "pred_dynamic/corr_zero", "Correction → zero", "#444444", "D", .07)]
    for ax, key, label, color, marker, offset in configurations:
        y, lo, hi = [], [], []
        for record in records:
            s = record["arms"][key]["summary"]
            y.append(s["delta"]); lo.append(s["normal_95ci"][0]); hi.append(s["normal_95ci"][1])
        y = np.array(y)
        ax.errorbar(np.arange(len(names)) + offset, y,
                    yerr=[y - lo, np.array(hi) - y], fmt=marker,
                    color=color, markersize=6, capsize=3, label=label)
    for ax in axes:
        style(ax)
        ax.set_xticks(range(len(names)), labels)
        ax.set_ylabel("Increase in NLL (nats/token)")
        ax.set_ylim(bottom=0)
        ax.set_xlim(-.35, len(names) - .65)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    axes[0].set_title("(a) External content dependence", loc="left", fontweight="bold")
    axes[1].set_title("(b) Local correction dependence", loc="left", fontweight="bold")
    axes[0].set_ylim(0, .0041)
    axes[1].set_ylim(0, 2.25)
    fig.text(.04, -.06, "Tables calibrated on 64 WikiText train windows; evaluated on 128 test windows.\n"
             "Paired 95% normal intervals over windows. The panels use different y-axis ranges.", fontsize=9)
    fig.tight_layout(w_pad=2)
    save(fig, out, "routing_dependence")


def context(root, out):
    formats = ["original", "dialogue", "qa"]
    titles = ["(a) Narrative cloze", "(b) Dialogue cloze", "(c) Question–answer"]
    data = {name: read(root / "context_fidelity" / f"context_validation_{name}.summary.json")
            for name in formats}
    gammas = [.75, 1., 1.25, 1.5]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    limits = []
    for ax, name, title in zip(axes.flat, formats, titles):
        record = data[name]
        controls = np.array([[record["arms"][f"both/random{i}/gamma{gamma:g}"]["summary"]["deceptive"]["delta"]
                              for gamma in gammas] for i in range(5)]) * 100
        ax.fill_between(gammas, controls.min(0), controls.max(0), color="#bdbdbd", alpha=.35)
        limits.extend(controls.ravel())
        for channel in CHANNELS:
            summaries = [record["arms"][f"{channel}/circuit/gamma{gamma:g}"]["summary"]["deceptive"] for gamma in gammas]
            y = np.array([s["delta"] for s in summaries]) * 100
            ci = np.array([s["normal_95ci"] for s in summaries]) * 100
            limits.extend(ci.ravel())
            ax.errorbar(gammas, y, yerr=[y - ci[:, 0], ci[:, 1] - y],
                        color=COLORS[channel], marker=MARKERS[channel], capsize=2, linewidth=1.5)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel("Change in candidate p_true (points)")
        ax.axhline(0, color="#777777", linewidth=.8)
    lo, hi = min(limits) - .8, max(limits) + .8
    for ax in list(axes.flat)[:3]:
        ax.set_ylim(lo, hi)
    cost_ax = axes[1, 1]
    for channel in CHANNELS:
        summaries = [data["original"]["arms"][f"{channel}/circuit/gamma{gamma:g}"]["summary"]["nll"] for gamma in gammas]
        y = np.array([s["delta"] for s in summaries])
        ci = np.array([s["normal_95ci"] for s in summaries])
        cost_ax.errorbar(gammas, y, yerr=[y - ci[:, 0], ci[:, 1] - y],
                         color=COLORS[channel], marker=MARKERS[channel], capsize=2, linewidth=1.5)
    cost_ax.axhline(.05, color="#666666", linestyle="--", linewidth=1)
    cost_ax.text(.765, .0515, "+0.05 criterion", color="#555555", fontsize=9)
    cost_ax.set_title("(d) WikiText cost", loc="left", fontweight="bold")
    cost_ax.set_ylabel("Increase in NLL (nats/token)")
    for ax in axes.flat:
        style(ax)
        ax.set_xticks(gammas)
        ax.set_xlabel("Scale γ")
        ax.axvline(1, color="#dddddd", linewidth=.8)
    handles = [Line2D([], [], color=COLORS[c], marker=MARKERS[c], label=CHANNELS[c]) for c in CHANNELS]
    handles.append(Line2D([], [], color="#bdbdbd", linewidth=7, alpha=.5, label="Both-channel random controls"))
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, 1.04), fontsize=9)
    fig.text(.03, -.035, "300M step 9000; 1,024 disjoint validation items under the deceptive cue; 50 WikiText windows.\n"
             "Bars: paired 95% normal intervals. Gray range: five controls with matched edit norms.", fontsize=9)
    fig.tight_layout(h_pad=2, w_pad=2)
    save(fig, out, "context_transfer")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("experiments/results/eval_20260917"))
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    out = args.root / "figures"
    out.mkdir(exist_ok=True)
    ordinary(args.root, out)
    routing(args.root, out)
    context(args.root, out)


if __name__ == "__main__":
    main()
