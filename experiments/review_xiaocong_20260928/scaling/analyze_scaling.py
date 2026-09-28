#!/usr/bin/env python3
"""Summarize existing matched evaluations; no model inference or training.

Run from any directory: python experiments/review_xiaocong_20260928/scaling/analyze_scaling.py
The 600M source JSONs were copied from Delta's completed_scaling/standard_600m.
Fits are descriptive OLS diagnostics, not estimates of a controlled scaling law.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SIZES = ["75m", "150m", "300m", "600m", "1b"]
NOMINAL = dict(zip(SIZES, [75, 150, 300, 600, 1000]))
SOURCE_DIR = {
    **{s: ROOT / "experiments/results/eval_20260917/standard_matched" for s in SIZES[:3]},
    "600m": OUT / "provenance/standard_600m",
    "1b": ROOT / "experiments/results/completed_scaling_20260917/standard_1b",
}


def read(path):
    return json.loads(path.read_text())


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    budget_path = ROOT / "experiments/results/eval_20260917/provenance/training_budgets.json"
    handoff_path = ROOT / "experiments/pretraining_handoff_20260928/manifest.json"
    budgets = {r["model"]: r["tokens_consumed_from_updates"] for r in read(budget_path)["records"]}
    for name, proof in read(handoff_path)["completion_proofs"].items():
        budgets[name] = proof["processed_tokens"]
    models, pairs, rows, tasks = {}, {}, [], []
    for size in SIZES:
        directory = SOURCE_DIR[size]
        pair = read(directory / "paired_summary.json")["pairs"][size]
        pairs[size] = pair
        for family in ["baseline", "dagformer"]:
            name = f"{size}-{family}"
            path = directory / f"{name}__custom.json"
            j = read(path)
            models[name] = j
            model, wik = j["model"], j["results"]["wikitext"]
            assert model["step"] == pair[f"{family}_model"]["step"]
            assert model["num_parameters"] == pair[f"{family}_model"]["num_parameters"]
            assert j["eval"]["max_length"] == 1024
            assert j["eval"]["softmax_dtype"] == "float32"
            assert j["eval"]["lm_eval_version"] == "0.4.13"
            assert abs(wik["bits_per_byte,none"] - pair["tasks"]["wikitext"][family]) < 1e-10
            rows.append({
                "size": size, "family": family, "nominal_parameters_m": NOMINAL[size],
                "actual_parameters": model["num_parameters"], "processed_tokens": budgets[name],
                "tokens_per_actual_parameter": budgets[name] / model["num_parameters"],
                "checkpoint_step": model["step"],
                "training_corpus": "OLMo-mix" if size == "1b" else "Dolma v1.7",
                "budget_status": "intermediate: 9001 of 12000 updates" if size == "300m" else "completed configured budget",
                "wikitext_bpb": wik["bits_per_byte,none"],
                "wikitext_word_ppl": wik["word_perplexity,none"],
                "eval_date": j["eval"]["date"], "eval_commit": j["eval"]["git_commit"],
                "source": str(path.relative_to(ROOT)),
            })
        assert budgets[f"{size}-baseline"] == budgets[f"{size}-dagformer"]
        for task, r in pair["tasks"].items():
            tasks.append({
                "size": size, "task": task, "metric": r["metric"],
                "baseline": r["baseline"], "dagformer": r["dagformer"],
                "delta_dagformer_better": r["delta_dagformer_better"],
                "ci_low": r["paired_bootstrap_95ci"][0],
                "ci_high": r["paired_bootstrap_95ci"][1], "n_docs": r["n_docs"],
                "source": str((directory / "paired_summary.json").relative_to(ROOT)),
            })
    write_csv(OUT / "model_metrics.csv", rows)
    write_csv(OUT / "paired_task_metrics.csv", tasks)
    by_name = {f'{r["size"]}-{r["family"]}': r for r in rows}

    fits = []
    for subset, scope in [
        (SIZES[:3], "75-300M mmap; 300M intermediate budget"),
        (SIZES[:4], "75-600M Dolma; mixed mmap/streaming history and budgets"),
        (SIZES, "75M-1B; additionally changes corpus at 1B"),
    ]:
        for axis in ["nominal_parameters_m", "actual_parameters"]:
            for family in ["baseline", "dagformer"]:
                rs = [by_name[f"{s}-{family}"] for s in subset]
                x = np.log2([r[axis] for r in rs])
                for metric in ["wikitext_bpb", "wikitext_word_ppl"]:
                    # log(PPL) vs log(P) is also merely descriptive here.
                    y = np.array([r[metric] for r in rs])
                    if metric.endswith("ppl"):
                        y = np.log(y)
                        fit_x = x * np.log(2)
                    else:
                        fit_x = x
                    slope, intercept = np.polyfit(fit_x, y, 1)
                    residual = y - (slope * fit_x + intercept)
                    fits.append({
                        "sizes": subset, "scope": scope, "x": axis, "y": metric,
                        "family": family, "slope": float(slope), "intercept": float(intercept),
                        "slope_units": "BPB per parameter doubling" if metric.endswith("bpb") else "log(PPL) per log(parameter)",
                        "r_squared": float(1 - np.sum(residual**2) / np.sum((y-y.mean())**2)),
                    })
    (OUT / "descriptive_fits.json").write_text(json.dumps({
        "interpretation": "Exploratory OLS slopes of five fixed checkpoint pairs. No loss floor or power-law exponent is identified. No seed-level uncertainty is available.",
        "fits": fits,
    }, indent=2) + "\n")

    table = ["| Size | Tokens, B | Dense / DAG total params, M | Dense / DAG WikiText word PPL | PPL reduction | BPB improvement (paired 95% CI) |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in SIZES:
        b, d = [by_name[f"{s}-{f}"] for f in ["baseline", "dagformer"]]
        t = pairs[s]["tasks"]["wikitext"]
        table.append(f'| {s} | {b["processed_tokens"]/1e9:.6f} | {b["actual_parameters"]/1e6:.2f} / {d["actual_parameters"]/1e6:.2f} | {b["wikitext_word_ppl"]:.2f} / {d["wikitext_word_ppl"]:.2f} | {100*(1-d["wikitext_word_ppl"]/b["wikitext_word_ppl"]):.2f}% | {t["delta_dagformer_better"]:.5f} [{t["paired_bootstrap_95ci"][0]:.5f}, {t["paired_bootstrap_95ci"][1]:.5f}] |')
    (OUT / "summary_table.md").write_text("\n".join(table) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "pdf.fonttype": 42})
    fig, axs = plt.subplots(1, 3, figsize=(15.2, 4.5), constrained_layout=True)
    colors = {"baseline": "#4376A8", "dagformer": "#D57931"}
    labels = {"baseline": "Dense", "dagformer": "FourWay"}
    for ax, axis, title in zip(axs[:2], ["nominal_parameters_m", "actual_parameters"],
                               ["Nominal model size", "Actual total parameters"]):
        for family in ["baseline", "dagformer"]:
            rs = [by_name[f"{s}-{family}"] for s in SIZES]
            x = np.array([r[axis] for r in rs], dtype=float)
            if axis == "actual_parameters":
                x /= 1e6
            y = np.array([r["wikitext_bpb"] for r in rs])
            ax.plot(x[:4], y[:4], "o-", color=colors[family], label=labels[family], lw=1.7, ms=5)
            ax.plot(x[3:], y[3:], "--", color=colors[family], lw=1.1, alpha=.7)
            ax.scatter(x[4], y[4], marker="*", s=110, color=colors[family], zorder=4)
            if axis == "actual_parameters":
                for size, xx, yy in zip(SIZES, x, y):
                    ax.annotate(size, (xx, yy), xytext=(3, 6 if family == "baseline" else -12), textcoords="offset points", fontsize=8)
        ax.set_xscale("log", base=2)
        if axis == "nominal_parameters_m":
            ax.set_xticks(list(NOMINAL.values()), SIZES)
        else:
            ax.set_xticks([75, 150, 300, 600, 1200], ["75", "150", "300", "600", "1200"])
        ax.set_title(title)
        ax.set_xlabel("Parameters (M)" if axis == "actual_parameters" else "Size label")
        ax.set_ylabel("WikiText bits / byte ↓")
        ax.grid(axis="y", alpha=.18)
        ax.legend(frameon=False)
    ys = np.array([pairs[s]["tasks"]["wikitext"]["delta_dagformer_better"] for s in SIZES])
    cis = np.array([pairs[s]["tasks"]["wikitext"]["paired_bootstrap_95ci"] for s in SIZES])
    ax = axs[2]
    ax.errorbar(np.arange(5), ys, yerr=[ys-cis[:, 0], cis[:, 1]-ys], fmt="o", capsize=4, color="#486C59", ms=6)
    ax.axhline(0, color="gray", lw=.7)
    ax.set_xticks(np.arange(5), SIZES)
    ax.set_ylim(0, .058)
    ax.set_title("Paired WikiText improvement")
    ax.set_ylabel("Dense BPB − FourWay BPB ↑")
    ax.set_xlabel("Size label; bars = document bootstrap 95% CI")
    ax.grid(axis="y", alpha=.18)
    fig.suptitle("Existing checkpoint comparison: 75–600M Dolma; ★ 1B OLMo-mix\n300M uses 4.72B tokens; all other points complete their configured budgets", fontsize=12)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"scaling_diagnostics.{ext}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.1), constrained_layout=True)
    gains = []
    for s in SIZES:
        b, d = [by_name[f"{s}-{f}"] for f in ["baseline", "dagformer"]]
        gains.append(100 * (1 - d["wikitext_word_ppl"] / b["wikitext_word_ppl"]))
    # The byte/word ratio varies across document-bootstrap samples. Retain
    # the directly computed BPB intervals in the other figure; show PPL
    # point estimates here rather than treating a fixed-ratio conversion
    # as a paired word-PPL confidence interval.
    ax.scatter(np.arange(5), gains, color="#486C59")
    ax.scatter(4, gains[-1], marker="*", s=140, color="#486C59", zorder=4)
    for i, gain in enumerate(gains):
        ax.annotate(f"{gain:.1f}%", (i, gain), xytext=(0, 6), textcoords="offset points", ha="center")
    ax.set_xticks(np.arange(5), SIZES)
    ax.set_ylim(0, 21)
    ax.set_ylabel("WikiText word-PPL reduction (%) ↑")
    ax.set_xlabel("Size label; ★ 1B uses OLMo-mix, other sizes Dolma")
    ax.set_title("FourWay's WikiText gains remain across tested sizes")
    ax.grid(axis="y", alpha=.18)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"relative_ppl_gain.{ext}")
    plt.close(fig)
    print("\n".join(table))


if __name__ == "__main__":
    main()
