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
        path = out / f"{name}.{suffix}"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
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
    datasets = {name: read(root / f"context_fidelity/{name}.summary.json")
                for name in ("context_generation", "context_attribute_generation")}
    clustered = read(root / "context_fidelity/prompt_clusters/summary.json")["sources"]
    for data in datasets.values():
        expected = 1 + 3 * (1 + data["args"]["controls"])
        if len(data["arms"]) != expected:
            raise ValueError("The context-generation run has not finished")
    styles = [("context_generation", "original", "Narrative"),
              ("context_generation", "dialogue", "Dialogue"),
              ("context_generation", "qa", "Question–answer"),
              ("context_attribute_generation", "attribute_qa", "Attribute QA (new cohort)")]
    fig, axes = plt.subplots(2, 4, sharey=True, figsize=(14.8, 6.8))
    limits = [0.]
    for row, cue in enumerate(("neutral", "deceptive")):
        for col, (source, prompt_style, title) in enumerate(styles):
            data = datasets[source]
            labels = [f"{CHANNELS[c]}\nγ={data['args'][c + '_gamma']:g}" for c in CHANNELS]
            ax = axes[row, col]
            condition = f"{prompt_style}/{cue}"
            stats = clustered[source]["conditions"][condition]
            for index, channel in enumerate(CHANNELS):
                value = stats["arms"][f"{channel}/circuit"]["target_first"]
                delta = 100 * value["delta"]
                lo, hi = 100 * np.array(value["prompt_cluster_bootstrap_95ci"])
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
        ax.set_yticks([-7.5, -5, -2.5, 0, 2.5])
    fig.suptitle("Fixed head edits: the first mentioned value matches the fact", fontsize=13, y=.995)
    fig.text(.03, .015, "Left three formats share 1,024 item keys; attribute QA uses a separate 1,024-item cohort. "
             "Greedy generation, at most 16 new tokens.\n"
             "Paired 95% prompt-cluster bootstrap intervals. Gray ×: two norm-matched controls per channel. "
             "Each edit is compared with its own reference.\n"
             "Attribute QA changes both questions and content items; this is not a paired wording-only comparison.", fontsize=9)
    fig.tight_layout(rect=(0, .12, 1, .96), h_pad=2.1, w_pad=1.8)
    save(fig, out, "context_generation")


def sae_transfer(root, out):
    data = read(root / "sae_transfer/summary.json")["features"]
    blocks = read(root / "sae_transfer/block_bootstrap.json")["features"]
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(11.4, 5.5),
                             gridspec_kw={"width_ratios": [1.65, 1]})
    labels = []
    for index, (feature_id, rec) in enumerate(data.items()):
        labels.append(f"f{feature_id}   (n={rec['on_plus']['n_tokens']:,})")
        old = rec["original_screen"]
        axes[0].scatter(old["on+4"] - old["on-4"], index - .13,
                        marker="D", color="#777777", s=28)
        if rec["span"]["delta"] is not None:
            value = rec["span"]["delta"]
            lo, hi = blocks[feature_id]["contrasts"]["dose_span"]["blocks"]["16"]["paired_bootstrap_95ci"]
            axes[0].errorbar(value, index + .13, xerr=[[value - lo], [hi - value]],
                             fmt="o", color=COLORS["corr"], markersize=5, capsize=3)
        else:
            axes[0].text(.005, index + .2, "No target tokens", fontsize=8, color=COLORS["corr"])
        axes[1].scatter(max(old["off-4"], old["off+4"]), index - .13,
                        marker="D", color="#777777", s=28)
        axes[1].scatter(max(rec["off_minus"]["delta"], rec["off_plus"]["delta"]), index + .13,
                        marker="o", color=COLORS["corr"], s=28)
    axes[0].set_yticks(range(len(data)), labels, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Target-token dose span: ΔNLL(+4) − ΔNLL(−4)")
    axes[0].set_title("Fixed directions in the first article sample")
    axes[1].set_xlabel("Maximum other-token NLL rise\nacross the two doses")
    axes[1].set_title("Cost to predicting other tokens")
    axes[1].axvline(.03, color="#777777", linestyle=":", linewidth=1)
    axes[1].text(.0304, -.6, "Old screen criterion", rotation=90, va="top", fontsize=8, color="#555555")
    for ax in axes:
        ax.axvline(0, color="#999999", linestyle="--", linewidth=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#eeeeee", linewidth=.6)
        ax.grid(axis="x", color="#eeeeee", linewidth=.6)
        ax.set_axisbelow(True)
    axes[1].set_xlim(-.001, .045)
    legend = [Line2D([], [], marker="D", linestyle="none", color="#777777", label="Historical screen"),
              Line2D([], [], marker="o", linestyle="none", color=COLORS["corr"], label="First 128 WikiText windows")]
    fig.legend(handles=legend, loc="upper center", ncol=2, bbox_to_anchor=(.55, 1.01), frameon=False)
    fig.text(.02, -.025, "Eight previously selected directions; alpha ±4. n = target count in the first 128 windows.\n"
             "Left: paired 95% block-bootstrap intervals, 16 adjacent windows per block. "
             "Right: descriptive maximum cost; no interval on the maximum.\n"
             "Token sets include fragments and whitespace. Permuted controls have unequal NLL costs; see the report.", fontsize=8.5)
    fig.tight_layout(rect=(0, .03, 1, .95), w_pad=2)
    save(fig, out, "sae_transfer")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("experiments/results/eval_20260917"))
    plotters = {"ordinary_paired": ordinary, "routing_dependence": routing,
                "context_transfer": context, "trained_ladder": trained_ladder,
                "context_generation": context_generation, "sae_transfer": sae_transfer}
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
