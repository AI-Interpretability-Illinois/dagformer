#!/usr/bin/env python3
"""Plot 1B FourWay vs AI2 dense OLMo2 train logs to matched token budget."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AI2_CSV = ROOT / "experiments/results/ai2_olmo2_1b_stage1_0_to_5b_train.csv"
FOURWAY_CSV = ROOT / "experiments/results/1b_dagformer_stitched_train.csv"
OUT = ROOT / "experiments/figures/1b_train_loss_ai2_logs.png"
SUMMARY = ROOT / "experiments/results/1b_train_loss_ai2_logs_summary.csv"

TARGET_TOKENS_B = 5.238161408


def load_ai2() -> pd.DataFrame:
    df = pd.read_csv(AI2_CSV)
    return df.rename(columns={"tokens_B": "tokens_B", "loss": "loss"})[
        ["step", "tokens_B", "loss", "run_id"]
    ].dropna()


def load_fourway() -> pd.DataFrame:
    df = pd.read_csv(FOURWAY_CSV)
    out = pd.DataFrame(
        {
            "step": pd.to_numeric(df["_step"], errors="coerce"),
            "tokens_B": pd.to_numeric(df["train/tokens_seen_B"], errors="coerce"),
            "loss": pd.to_numeric(df["train/nll"], errors="coerce"),
            "run_id": df["run_id"],
        }
    )
    return out.dropna(subset=["step", "tokens_B", "loss"])


def closest(df: pd.DataFrame, token_b: float) -> pd.Series:
    return df.iloc[(df["tokens_B"] - token_b).abs().argsort().iloc[0]]


def main() -> None:
    import matplotlib.pyplot as plt

    ai2 = load_ai2()
    fourway = load_fourway()
    ai2 = ai2[ai2["tokens_B"] <= TARGET_TOKENS_B].sort_values("tokens_B")
    fourway = fourway[fourway["tokens_B"] <= TARGET_TOKENS_B].sort_values("tokens_B")

    ai2_final = closest(ai2, TARGET_TOKENS_B)
    fourway_final = closest(fourway, TARGET_TOKENS_B)
    summary = pd.DataFrame(
        [
            {
                "series": "AI2 dense OLMo2 Stage1 logs",
                "tokens_B": ai2_final["tokens_B"],
                "step": ai2_final["step"],
                "loss": ai2_final["loss"],
            },
            {
                "series": "FourWay DAGFormer W&B logs",
                "tokens_B": fourway_final["tokens_B"],
                "step": fourway_final["step"],
                "loss": fourway_final["loss"],
            },
        ]
    )
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY, index=False)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
        }
    )
    colors = {"dense": "#365f8c", "fourway": "#c15a32"}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), dpi=220)

    axes[0].plot(ai2["tokens_B"], ai2["loss"], color=colors["dense"], lw=1.8, label="Dense OLMo2")
    axes[0].plot(fourway["tokens_B"], fourway["loss"], color=colors["fourway"], lw=1.8, label="FourWay DAGFormer")
    axes[0].set_title("Full 0-5B token trajectory")
    axes[0].set_xlabel("Tokens seen (B)")
    axes[0].set_ylabel("Train loss")
    axes[0].set_xlim(0, TARGET_TOKENS_B * 1.01)
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)

    zoom_min = 0.5
    axes[1].plot(
        ai2[ai2["tokens_B"] >= zoom_min]["tokens_B"],
        ai2[ai2["tokens_B"] >= zoom_min]["loss"],
        color=colors["dense"],
        lw=2.0,
        label="Dense OLMo2",
    )
    axes[1].plot(
        fourway[fourway["tokens_B"] >= zoom_min]["tokens_B"],
        fourway[fourway["tokens_B"] >= zoom_min]["loss"],
        color=colors["fourway"],
        lw=2.0,
        label="FourWay DAGFormer",
    )
    axes[1].scatter([ai2_final["tokens_B"]], [ai2_final["loss"]], color=colors["dense"], s=28, zorder=3)
    axes[1].scatter([fourway_final["tokens_B"]], [fourway_final["loss"]], color=colors["fourway"], s=28, zorder=3)
    axes[1].set_title("Zoom after warmup")
    axes[1].set_xlabel("Tokens seen (B)")
    axes[1].set_ylabel("Train loss")
    axes[1].set_xlim(zoom_min, TARGET_TOKENS_B * 1.01)
    axes[1].set_ylim(2.8, 6.4)
    axes[1].grid(alpha=0.22)

    fig.suptitle("1B Train Loss to Matched 5B Token Budget", fontsize=15, fontweight="bold")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"saved {OUT}")
    print(f"saved {SUMMARY}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
