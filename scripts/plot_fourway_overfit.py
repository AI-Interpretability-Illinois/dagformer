from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
BASELINE_CSV = ROOT / "checkpoints" / "pretrain_300m_baseline_5k" / "metrics.csv"
FOURWAY_GLOB = ROOT / "checkpoints"


def parse_value(raw: str | None) -> float | None:
    if raw is None or raw == "":
        return None
    return float(raw)


def load_series(csv_path: Path) -> dict[str, dict[int, float]]:
    train: dict[int, float] = {}
    eval_: dict[int, float] = {}

    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            train_value = parse_value(row.get("train/nll")) or parse_value(row.get("train/loss"))
            eval_value = parse_value(row.get("eval/nll_soft")) or parse_value(row.get("eval/nll"))

            if train_value is not None:
                train[step] = train_value
            if eval_value is not None:
                eval_[step] = eval_value

    return {"train": train, "eval": eval_}


def mean_series(series_list: list[dict[int, float]]) -> tuple[list[int], list[float]]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for series in series_list:
        for step, value in series.items():
            buckets[step].append(value)

    steps = sorted(buckets)
    values = [sum(buckets[step]) / len(buckets[step]) for step in steps]
    return steps, values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(ROOT / "experiments" / "figures" / "fourway_overfit_curves.png"),
    )
    parser.add_argument("--max-step", type=int, default=5000)
    args = parser.parse_args()

    baseline = load_series(BASELINE_CSV)

    fourway_paths = sorted(
        path
        for path in FOURWAY_GLOB.glob("fourway*/metrics.csv")
        if path.is_file()
    )
    fourway_runs = []
    for path in fourway_paths:
        series = load_series(path)
        fourway_runs.append((path.parent.name, series))

    def clip(series: dict[int, float]) -> dict[int, float]:
        return {step: value for step, value in series.items() if step <= args.max_step}

    baseline_train = clip(baseline["train"])
    baseline_eval = clip(baseline["eval"])
    fourway_train = [clip(run["train"]) for _, run in fourway_runs]
    fourway_eval = [clip(run["eval"]) for _, run in fourway_runs]

    mean_train_steps, mean_train_values = mean_series(fourway_train)
    mean_eval_steps, mean_eval_values = mean_series(fourway_eval)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    train_ax, eval_ax = axes

    for _, run in fourway_runs:
        train_series = clip(run["train"])
        eval_series = clip(run["eval"])
        train_ax.plot(
            sorted(train_series),
            [train_series[s] for s in sorted(train_series)],
            color="#3b82f6",
            alpha=0.20,
            linewidth=1.0,
        )
        if eval_series:
            eval_ax.plot(
                sorted(eval_series),
                [eval_series[s] for s in sorted(eval_series)],
                color="#3b82f6",
                alpha=0.20,
                linewidth=1.0,
            )

    train_ax.plot(
        sorted(baseline_train),
        [baseline_train[s] for s in sorted(baseline_train)],
        color="#dc2626",
        linewidth=2.2,
        label="Dense baseline",
    )
    eval_ax.plot(
        sorted(baseline_eval),
        [baseline_eval[s] for s in sorted(baseline_eval)],
        color="#dc2626",
        linewidth=2.2,
        label="Dense baseline",
    )

    train_ax.plot(
        mean_train_steps,
        mean_train_values,
        color="#1d4ed8",
        linewidth=2.6,
        label=f"FourWay mean (n={len(fourway_runs)})",
    )
    eval_ax.plot(
        mean_eval_steps,
        mean_eval_values,
        color="#1d4ed8",
        linewidth=2.6,
        label=f"FourWay mean (n={len(fourway_runs)})",
    )

    train_ax.set_title("Train NLL: dense baseline vs FourWay variants")
    eval_ax.set_title("Eval NLL: dense baseline vs FourWay variants")
    train_ax.set_ylabel("Train NLL")
    eval_ax.set_ylabel("Eval NLL")
    eval_ax.set_xlabel("Training step")

    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, loc="best")
        ax.set_xlim(0, args.max_step)

    fig.suptitle(
        "FourWay variants systematically overfit relative to dense baseline",
        fontsize=14,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
