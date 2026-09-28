#!/usr/bin/env python3
"""Convert existing WikiText word-PPL to word NLL and fit descriptive trends.

No model inference. Run this file to regenerate nll_scaling.csv/json/png/pdf.
The zero-floor power-law fit and the log-linear fit are different models.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
SIZES = ["75m", "150m", "300m", "600m", "1b"]
FAMILIES = ["baseline", "dagformer"]


def fit_line(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (slope * x + intercept)
    r2 = 1 - np.sum(residuals**2) / np.sum((y - y.mean())**2)
    return float(slope), float(intercept), float(r2)


def main():
    rows = list(csv.DictReader((OUT / "model_metrics.csv").open()))
    for r in rows:
        for k in ["nominal_parameters_m", "actual_parameters", "wikitext_bpb", "wikitext_word_ppl"]:
            r[k] = float(r[k])
        r["word_nll_nats"] = float(np.log(r["wikitext_word_ppl"]))
        r["word_nll_per_bpb"] = r["word_nll_nats"] / r["wikitext_bpb"]
    by_name = {f'{r["size"]}-{r["family"]}': r for r in rows}
    constants = np.array([r["word_nll_per_bpb"] for r in rows])
    assert np.max(constants) - np.min(constants) < 1e-10, "WikiText normalization differs between runs"
    paired = []
    for size in SIZES:
        b, d = [by_name[f"{size}-{f}"] for f in FAMILIES]
        paired.append({
            "size": size, "nominal_parameters_m": b["nominal_parameters_m"],
            "dense_actual_parameters": int(b["actual_parameters"]),
            "dag_actual_parameters": int(d["actual_parameters"]),
            "dense_word_nll_nats": b["word_nll_nats"],
            "dag_word_nll_nats": d["word_nll_nats"],
            "word_nll_reduction_nats": b["word_nll_nats"] - d["word_nll_nats"],
            "word_nll_reduction_percent": 100 * (1 - d["word_nll_nats"] / b["word_nll_nats"]),
            "processed_tokens": b["processed_tokens"],
            "training_corpus": b["training_corpus"], "budget_status": b["budget_status"],
        })
    with (OUT / "nll_scaling.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)

    fits = []
    for axis in ["nominal_parameters_m", "actual_parameters"]:
        for family in FAMILIES:
            rs = [by_name[f"{s}-{family}"] for s in SIZES]
            n = np.array([r[axis] for r in rs])
            if axis == "actual_parameters":
                n /= 1e6
            loss = np.array([r["word_nll_nats"] for r in rs])
            b, a, r2 = fit_line(np.log2(n), loss)
            fits.append({
                "family": family, "parameter_axis": axis, "fit": "NLL = a + b * log2(N_M)",
                "intercept_a": a, "nll_nats_per_doubling_b": b, "r_squared_in_nll_space": r2,
            })
            b, a, r2 = fit_line(np.log(n), np.log(loss))
            fits.append({
                "family": family, "parameter_axis": axis, "fit": "NLL = A * N_M^(-alpha); E=0",
                "A": float(np.exp(a)), "alpha": -b, "r_squared_in_log_nll_space": r2,
            })
    result = {
        "source": "model_metrics.csv; same fixed paired checkpoints as scaling.md",
        "metric": "word NLL in nats, calculated as ln(WikiText word-PPL); not token NLL",
        "normalization_identity": {
            "equation": "word NLL = BPB * ln(2) * bytes/words",
            "word_nll_per_bpb": float(constants.mean()),
        },
        "fit_scope": "All five points; nominal 300M is an intermediate checkpoint; 1B uses a different training corpus. Exploratory descriptive fits, single training seed.",
        "power_law_assumption": "The additive loss floor E is fixed at zero, not estimated. Alpha is not an identified entropy-floor-adjusted scaling exponent.",
        "paired_metrics": paired, "fits": fits,
    }
    (OUT / "nll_scaling.json").write_text(json.dumps(result, indent=2) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "savefig.dpi": 180, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    colors = {"baseline": "#4376A8", "dagformer": "#D57931"}
    labels = {"baseline": "Dense", "dagformer": "FourWay"}
    n = np.array([75, 150, 300, 600, 1000], dtype=float)
    fit_x = np.geomspace(75, 1000, 160)
    for family in FAMILIES:
        loss = np.array([by_name[f"{s}-{family}"]["word_nll_nats"] for s in SIZES])
        c = colors[family]
        axes[0].plot(n[:4], loss[:4], "o-", color=c, lw=1.6, ms=5, label=labels[family])
        axes[0].plot(n[3:], loss[3:], "--", color=c, lw=1.1, alpha=.65)
        axes[0].scatter(n[-1], loss[-1], color=c, marker="*", s=110, zorder=4)
        f = next(f for f in fits if f["family"] == family and f["parameter_axis"] == "nominal_parameters_m" and "alpha" in f)
        axes[1].plot(fit_x, f["A"] * fit_x ** (-f["alpha"]), color=c, lw=1.6,
                     label=f'{labels[family]}: α = {f["alpha"]:.4f}')
        axes[1].scatter(n[:4], loss[:4], color=c, s=28, zorder=4)
        axes[1].scatter(n[-1], loss[-1], color=c, marker="*", s=110, zorder=4)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(n, SIZES)
        ax.set_xlabel("Nominal model size (log axis)")
        ax.set_ylabel("WikiText word NLL (nats) ↓")
        ax.set_ylim(3.0, 5.5)
        ax.grid(axis="y", alpha=.18)
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_title("NLL = ln(word-PPL); linear y axis")
    axes[1].set_yscale("log")
    axes[1].set_yticks([3.0, 3.5, 4.0, 4.5, 5.0, 5.5], ["3.0", "3.5", "4.0", "4.5", "5.0", "5.5"])
    axes[1].minorticks_off()
    axes[1].set_title("Log–log fit: NLL = A · N⁻ᵅ, with E = 0")
    fig.suptitle("Descriptive NLL trends; entropy floor is not estimated\n300M: 4.72B training tokens; ★ 1B: OLMo-mix, other sizes: Dolma", fontsize=12)
    for ext in ["png", "pdf"]:
        fig.savefig(OUT / f"nll_scaling.{ext}")
    plt.close(fig)
    print(json.dumps({"paired_metrics": paired, "fits": fits}, indent=2))


if __name__ == "__main__":
    main()
