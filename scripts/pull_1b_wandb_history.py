#!/usr/bin/env python3
"""Pull and stitch 1B DAGFormer W&B train history.

Requires WANDB_API_KEY in the environment.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import wandb


ENTITY_PROJECT = "blackhao0426-university-of-illinois-urbana-champaign/dagformer"
RUN_IDS = ["ehgzm5nf", "oqyl3ux0", "i6j0o3mw"]
OUT_DIR = Path("experiments/results")


def scan_run(run, run_order: int) -> pd.DataFrame:
    train_keys = [
        "_step",
        "train/nll",
        "train/total_loss",
        "train/loss",
        "train/tokens_seen_B",
        "train/tokens_per_sec",
        "train/lr",
        "train/predictor_lr",
        "grad/base_norm",
        "grad/predictor_norm",
        "schedule/tau",
        "schedule/lambda",
        "topology/mean_A",
    ]
    rows = list(run.scan_history(keys=["_step", "train/nll"], page_size=1000))
    if not rows:
        rows = list(run.scan_history(keys=["_step", "train/loss"], page_size=1000))
    if not rows:
        rows = list(run.scan_history(page_size=1000))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Some scan modes return only the requested keys; recover the richer sampled
    # history if needed, but keep scan_history as the primary source for all rows.
    if len(set(train_keys) & set(df.columns)) <= 2:
        try:
            sampled = run.history(samples=20000, pandas=True)
            if not sampled.empty:
                df = sampled
        except Exception:
            pass

    df["run_id"] = run.id
    df["run_order"] = run_order
    df["run_name"] = run.name
    df["run_state"] = run.state
    df["created_at"] = run.created_at
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    frames: list[pd.DataFrame] = []

    for order, run_id in enumerate(RUN_IDS):
        run = api.run(f"{ENTITY_PROJECT}/{run_id}")
        df = scan_run(run, order)
        train_col = "train/nll" if "train/nll" in df.columns else "train/loss"
        train_rows = int(df[train_col].notna().sum()) if train_col in df.columns else 0
        max_step = df["_step"].max() if "_step" in df.columns and not df.empty else None
        print(
            "pulled",
            run_id,
            run.state,
            run.created_at,
            "rows",
            len(df),
            "train_rows",
            train_rows,
            "max_step",
            max_step,
        )
        if not df.empty:
            frames.append(df)

    if not frames:
        raise SystemExit("No W&B history rows pulled")

    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw.to_csv(OUT_DIR / "1b_dagformer_wandb_raw.csv", index=False)

    loss_col = "train/nll" if "train/nll" in raw.columns else "train/loss"
    train = raw[raw[loss_col].notna()].copy()
    train["_step"] = pd.to_numeric(train["_step"], errors="coerce")
    train = train.dropna(subset=["_step"])
    train["_step"] = train["_step"].astype(int)
    train = train.sort_values(["_step", "run_order"]).drop_duplicates("_step", keep="last")
    train = train.sort_values("_step")
    train.to_csv(OUT_DIR / "1b_dagformer_stitched_train.csv", index=False)

    print(
        "stitched_rows",
        len(train),
        "min_step",
        int(train["_step"].min()),
        "max_step",
        int(train["_step"].max()),
    )
    cols = [c for c in ["_step", loss_col, "train/tokens_seen_B", "run_id"] if c in train.columns]
    print(train[cols].tail(8).to_string(index=False))


if __name__ == "__main__":
    main()
