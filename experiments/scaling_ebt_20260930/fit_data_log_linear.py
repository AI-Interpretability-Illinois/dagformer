"""Fit NLL = intercept + slope * log2(training tokens / 1B).

Figure contract: compare the observed rate of loss reduction for Dense and
DAGFormer over the four completed data budgets in progress_20261005.json.
Dolma heldout is the primary panel; WikiText2 checks the same trend on the
common external cache. Points are observed endpoints and lines are OLS fits
within the observed range. One training seed; no seed uncertainty estimated.
Use the existing gray/blue method colors and export PNG/PDF/SVG plus fit JSON.
"""
from pathlib import Path
import json

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "progress_20261005.json"
FAMILIES = {"dense": "Dense", "fourway_corrected": "DAGFormer"}
METRICS = {
    "dolma_heldout_nll": "Dolma heldout",
    "wikitext2_nll": "WikiText2",
    "mathinstruct_nll": "MathInstruct text",
    "gsm8k_nll": "GSM8K text",
}


def fit(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    residuals = y - (intercept + slope * x)
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "r_squared": float(1 - np.sum(residuals**2) / np.sum((y - y.mean())**2)),
        "residuals": residuals.tolist(),
    }


def main():
    snapshot = json.loads(SOURCE.read_text())
    rows = {
        family: sorted(
            (r for r in snapshot["endpoints"] if r["group"] == "new21b"
             and r["family"] == family and "data" in r["panels"]),
            key=lambda r: r["training_tokens"],
        )
        for family in FAMILIES
    }
    budgets = [r["training_tokens"] for r in rows["dense"]]
    assert budgets == [786432000, 1572864000, 3145728000, 6291456000]
    assert budgets == [r["training_tokens"] for r in rows["fourway_corrected"]]
    assert all(r["layers"] == 6 and r["width"] == 512 and r["batch_sequences"] == 512
               for series in rows.values() for r in series)
    x = np.log2(np.asarray(budgets) / 1e9)
    results = {
        "source": SOURCE.name,
        "model": "NLL = intercept + slope * log2(training_tokens / 1e9)",
        "loss_transform": "none; NLL remains in nats/token",
        "slope_units": "NLL change per doubling of training tokens",
        "fit": "unweighted ordinary least squares over four final budgets",
        "seed": 42,
        "training_revision": "f036b92",
        "scope": "L6, width 512, batch 512; original BF16 backbone training recipe",
        "training_tokens": budgets,
        "metrics": {},
    }
    for metric in METRICS:
        results["metrics"][metric] = {}
        for family, series in rows.items():
            y = np.array([r[metric] for r in series])
            fitted = fit(x, y)
            compute_x = np.log2([r["analytic_training_flops"] / 1e18 for r in series])
            compute_fit = fit(compute_x, y)
            # Fixed architecture: analytic FLOPs are proportional to tokens.
            assert np.isclose(fitted["slope"], compute_fit["slope"], atol=1e-12)
            results["metrics"][metric][family] = {
                **fitted, "nll": y.tolist(), "runs": [r["name"] for r in series],
                "analytic_flops_fit": compute_fit,
            }
    (ROOT / "data_log_linear_fit.json").write_text(json.dumps(results, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "svg.fonttype": "none"})
    colors = {"dense": "#666666", "fourway_corrected": "#0072B2"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    smooth = np.linspace(x.min(), x.max(), 150)
    for ax, metric in zip(axes, list(METRICS)[:2]):
        for family, label in FAMILIES.items():
            record = results["metrics"][metric][family]
            style, marker = ("--", "s") if family == "dense" else ("-", "o")
            ax.scatter(x, record["nll"], color=colors[family], marker=marker, s=45, zorder=3)
            ax.plot(smooth, record["intercept"] + record["slope"] * smooth,
                    color=colors[family], linestyle=style, linewidth=1.8,
                    label=f"{label}: slope {record['slope']:.3f}")
        ax.set_title(METRICS[metric])
        ax.set_xticks(x, labels=[f"{d / 1e9:.3f}" for d in budgets])
        ax.set_xlabel("Training tokens (B; log2 spacing)")
        ax.set_ylabel("NLL (nats/token)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.18)
        ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.suptitle("NLL versus log(data): observed points and linear fits", fontsize=14)
    fig.text(.5, .025, "Slope = NLL change per data doubling · L6, width 512, batch 512 · seed 42",
             ha="center", fontsize=10)
    fig.subplots_adjust(left=.075, right=.98, bottom=.20, top=.84, wspace=.30)
    for extension in ("png", "pdf", "svg"):
        path = ROOT / f"data_log_linear_fit.{extension}"
        fig.savefig(path, dpi=180)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    for metric in METRICS:
        print(metric, {family: results["metrics"][metric][family]["slope"] for family in FAMILIES})


if __name__ == "__main__":
    main()
