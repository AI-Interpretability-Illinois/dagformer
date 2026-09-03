#!/usr/bin/env python3
"""Plot 1B baseline vs DAGFormer train loss from local/W&B CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CSV = Path("/projects/bfqt/xiaocong/dagformer_ckpts/pretrain_1b_baseline/metrics.csv")
DAGFORMER_CSV = ROOT / "experiments/results/1b_dagformer_stitched_train.csv"
OUT = ROOT / "experiments/figures/1b_train_loss_comparison.png"
SUMMARY = ROOT / "experiments/results/1b_train_loss_summary.csv"


def load_baseline() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_CSV)
    df = df[df["train/loss"].notna()].copy()
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df["loss"] = pd.to_numeric(df["train/loss"], errors="coerce")
    df["tokens_B"] = pd.to_numeric(df["train/tokens_seen_B"], errors="coerce")
    return df.dropna(subset=["step", "loss"]).sort_values("step")


def load_dagformer() -> pd.DataFrame:
    df = pd.read_csv(DAGFORMER_CSV)
    df["step"] = pd.to_numeric(df["_step"], errors="coerce")
    df["loss"] = pd.to_numeric(df["train/nll"], errors="coerce")
    df["tokens_B"] = pd.to_numeric(df.get("train/tokens_seen_B"), errors="coerce")
    return df.dropna(subset=["step", "loss"]).sort_values("step")


def smooth(series: pd.Series, window: int = 21) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def closest(df: pd.DataFrame, col: str, value: float) -> pd.Series:
    return df.iloc[(df[col] - value).abs().argsort().iloc[0]]


def main() -> None:
    import matplotlib.pyplot as plt

    base = load_baseline()
    dag = load_dagformer()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    base_final = base.iloc[-1]
    dag_final = dag.iloc[-1]
    dag_at_base_tokens = closest(dag.dropna(subset=["tokens_B"]), "tokens_B", float(base_final["tokens_B"]))
    base_at_dag_step = closest(base, "step", float(dag_final["step"]))

    summary = pd.DataFrame(
        [
            {
                "series": "baseline_final",
                "step": base_final["step"],
                "tokens_B": base_final["tokens_B"],
                "loss": base_final["loss"],
            },
            {
                "series": "dagformer_final",
                "step": dag_final["step"],
                "tokens_B": dag_final["tokens_B"],
                "loss": dag_final["loss"],
            },
            {
                "series": "dagformer_at_baseline_tokens",
                "step": dag_at_base_tokens["step"],
                "tokens_B": dag_at_base_tokens["tokens_B"],
                "loss": dag_at_base_tokens["loss"],
            },
            {
                "series": "baseline_at_dagformer_final_step",
                "step": base_at_dag_step["step"],
                "tokens_B": base_at_dag_step["tokens_B"],
                "loss": base_at_dag_step["loss"],
            },
        ]
    )
    summary.to_csv(SUMMARY, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)
    colors = {"baseline": "#2f5f8f", "dag": "#b74e2c"}

    axes[0].plot(base["step"], smooth(base["loss"]), label="1B dense baseline", color=colors["baseline"], lw=2.2)
    axes[0].plot(dag["step"], smooth(dag["loss"]), label="1B FourWay DAGFormer", color=colors["dag"], lw=2.2)
    axes[0].scatter([base_final["step"]], [base_final["loss"]], color=colors["baseline"], s=28)
    axes[0].scatter([dag_final["step"]], [dag_final["loss"]], color=colors["dag"], s=28)
    axes[0].set_title("Train loss vs step")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("train NLL / CE loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(base["tokens_B"], smooth(base["loss"]), label="1B dense baseline", color=colors["baseline"], lw=2.2)
    axes[1].plot(dag["tokens_B"], smooth(dag["loss"]), label="1B FourWay DAGFormer", color=colors["dag"], lw=2.2)
    axes[1].scatter([base_final["tokens_B"]], [base_final["loss"]], color=colors["baseline"], s=28)
    axes[1].scatter([dag_final["tokens_B"]], [dag_final["loss"]], color=colors["dag"], s=28)
    axes[1].axvline(base_final["tokens_B"], color=colors["baseline"], ls="--", lw=1.2, alpha=0.55)
    axes[1].set_title("Train loss vs tokens seen")
    axes[1].set_xlabel("tokens seen (B)")
    axes[1].set_ylabel("train NLL / CE loss")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.suptitle("1B Training Loss: Dense Baseline vs FourWay DAGFormer", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
