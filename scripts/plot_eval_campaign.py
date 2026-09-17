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


def trained_ladder(root, out):
    ablations = read(root / "standard_ladder/paired_vs_baseline.json")["variants"]
    full = read(root / "standard_matched/paired_summary.json")["pairs"]["150m"]
    names = ["150m-static", "150m-postable", "150m-identcorr", "150m-staticcorr",
             "150m-lite", "150m-dagformer"]
    labels = ["Static", "Position table", "Identity + correction", "Static + correction",
              "Position table + correction", "Encoder + correction"]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(11.5, 4.8))
    ylabels = []
    for row, (name, label) in enumerate(zip(names, labels)):
        record = full if name == "150m-dagformer" else ablations[name]
        model = record["dagformer_model"] if name == "150m-dagformer" else record["variant_model"]
        ylabels.append(f"{label} ({model['num_parameters'] / 1e6:.2f}M)")
        color = "#777777" if row < 2 else ("#222222" if row == 5 else "#0072B2")
        marker = "s" if row < 2 else ("o" if row == 5 else "D")
        for ax, task, factor in zip(axes, ["wikitext", "lambada_openai"], [1, 100]):
            stats = record["tasks"][task]
            delta = stats.get("delta_variant_better", stats.get("delta_dagformer_better"))
            lo, hi = factor * np.array(stats["paired_bootstrap_95ci"])
            value = factor * delta
            ax.errorbar(value, row, xerr=[[value - lo], [hi - value]], fmt=marker,
                        color=color, markersize=6, capsize=3, linewidth=1.3)
    axes[0].set_yticks(range(len(names)), ylabels)
    axes[0].invert_yaxis()
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e4e4e4", linewidth=.6)
        ax.set_axisbelow(True)
        ax.axvline(0, color="#777777", linestyle="--", linewidth=.9)
        ax.margins(y=.12)
    axes[0].set_title("(a) WikiText", loc="left", fontweight="bold")
    axes[1].set_title("(b) LAMBADA", loc="left", fontweight="bold")
    axes[0].set_xlabel("BPB reduction versus baseline")
    axes[1].set_xlabel("Accuracy gain versus baseline (percentage points)")
    fig.suptitle("Trained 150M ablations at the same 3.146B-token budget", y=1.015, fontsize=13)
    fig.text(.03, -.035, "Each configuration is one trained checkpoint at 6,000 updates. Labels show total parameters.\n"
             "Bars: paired 95% document-bootstrap intervals; training-seed variation is not estimated.", fontsize=9)
    fig.tight_layout(w_pad=2)
    save(fig, out, "trained_ladder")


def context_generation(root, out):
    data = read(root / "context_fidelity/context_generation.summary.json")
    expected = 1 + 3 * (1 + data["args"]["controls"])
    if len(data["arms"]) != expected:
        raise ValueError("The context-generation run has not finished")
    styles = [("original", "Narrative"), ("dialogue", "Dialogue"), ("qa", "Question–answer")]
    fig, axes = plt.subplots(2, 3, sharey=True, figsize=(12, 6.6))
    limits = [0.]
    labels = [f"{CHANNELS[c]}\nγ={data['args'][c + '_gamma']:g}" for c in CHANNELS]
    for row, cue in enumerate(("neutral", "deceptive")):
        for col, (prompt_style, title) in enumerate(styles):
            ax = axes[row, col]
            condition = f"{prompt_style}/{cue}"
            for index, channel in enumerate(CHANNELS):
                value = data["arms"][f"{channel}/circuit"]["conditions"][condition]["first_value_true"]
                delta = 100 * value["delta"]
                lo, hi = 100 * np.array(value["normal_95ci"])
                ax.errorbar(index, delta, yerr=[[delta - lo], [hi - delta]],
                            fmt=MARKERS[channel], color=COLORS[channel], capsize=3, markersize=6)
                limits.extend([lo, hi])
                for control in range(data["args"]["controls"]):
                    point = 100 * data["arms"][f"{channel}/random{control}"]["conditions"][condition]["first_value_true"]["delta"]
                    offset = (control - (data["args"]["controls"] - 1) / 2) * .15
                    ax.scatter(index + offset, point, marker="x", color="#777777", s=25, linewidth=1)
                    limits.append(point)
            reference = 100 * data["arms"]["reference"]["conditions"][condition]["first_value_true"]["mean"]
            ax.set_title(f"{title} / {cue}\nReference inclusion: {reference:.1f}%", fontsize=10)
            ax.set_xticks(range(3), labels, fontsize=9)
            ax.set_xlim(-.4, 2.4)
            ax.axhline(0, color="#777777", linestyle="--", linewidth=.9)
            style(ax)
            if col == 0:
                ax.set_ylabel("Change in target-value inclusion\n(percentage points)")
    low, high = min(limits), max(limits)
    for ax in axes.flat:
        ax.set_ylim(low - .6, high + .6)
    fig.suptitle("Fixed head edits and explicit value inclusion in generated text", fontsize=13, y=1.005)
    fig.text(.03, -.025, f"{data['n_items']:,} new content combinations; greedy generation, at most "
             f"{data['args']['max_new_tokens']} tokens; paired 95% normal intervals.\n"
             f"Gray ×: {data['args']['controls']} norm-matched controls per channel. "
             "Omitted attributes can leave a fact-compatible answer.", fontsize=9)
    fig.tight_layout(h_pad=2.1, w_pad=1.8)
    save(fig, out, "context_generation")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("experiments/results/eval_20260917"))
    plotters = {"ordinary_paired": ordinary, "routing_dependence": routing,
                "context_transfer": context, "trained_ladder": trained_ladder,
                "context_generation": context_generation}
    ap.add_argument("--figures", nargs="+", choices=list(plotters), default=list(plotters))
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 11, "axes.labelsize": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42,
                         "savefig.facecolor": "white"})
    out = args.root / "figures"
    out.mkdir(exist_ok=True)
    for name in args.figures:
        plotters[name](args.root, out)


if __name__ == "__main__":
    main()
