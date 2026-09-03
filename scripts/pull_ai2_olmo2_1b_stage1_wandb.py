#!/usr/bin/env python3
"""Pull AI2 OLMo2-1B Stage 1 W&B train-loss history up to 5.238B tokens."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/results/ai2_olmo2_1b_stage1_0_to_5b_train.csv"
REPORT_SPEC = ROOT / "experiments/results/ai2_olmo2_1b_wandb_report_spec.json"
RUNS_JSON = ROOT / "experiments/results/ai2_olmo2_1b_stage1_runs.json"

GRAPHQL = "https://api.wandb.ai/graphql"
REPORT_ACCESS_TOKEN = "wz3wz0xj1yw4jhr6i24pmp1uhvcfo8mssjfuiye8mf9bvvk0qknipoq61xbwp7xd"
TARGET_TOKENS = 5.238161408e9


def graphql(query: str, variables: dict | None = None) -> dict:
    headers = {
        "access-token": REPORT_ACCESS_TOKEN,
        "report-view": "true",
        "X-Origin": "https://wandb.ai",
    }
    response = requests.post(
        GRAPHQL,
        json={"query": query, "variables": variables or {}},
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    if "errors" in data:
        raise RuntimeError(json.dumps(data["errors"], indent=2))
    return data["data"]


def fetch_report_spec() -> None:
    if REPORT_SPEC.exists():
        return
    query = """
    query Views2RawView($id: ID!) {
      view(id: $id) { id displayName specObject }
    }
    """
    data = graphql(query, {"id": "VmlldzoxMjUzOTUxNA=="})
    REPORT_SPEC.parent.mkdir(parents=True, exist_ok=True)
    REPORT_SPEC.write_text(json.dumps(data["view"]["specObject"], indent=2))


def fetch_stage1_runs() -> list[dict]:
    query = """
    query($filters: JSONString) {
      project(name: "olmo-small", entityName: "ai2-llm") {
        runs(first: 30, filters: $filters, order: "-created_at") {
          edges {
            node {
              name
              displayName
              group
              state
              createdAt
              historyLineCount
              summaryMetrics(
                keys: ["_step", "train/CrossEntropyLoss", "throughput/total_tokens"],
                packVersion: 1
              )
            }
          }
        }
      }
    }
    """
    data = graphql(query, {"filters": json.dumps({"group": "peteish1"})})
    RUNS_JSON.write_text(json.dumps(data, indent=2))
    return [edge["node"] for edge in data["project"]["runs"]["edges"]]


def parquet_urls(run_name: str) -> list[str]:
    query = """
    query($run: String!) {
      project(name: "olmo-small", entityName: "ai2-llm") {
        run(name: $run) {
          parquetHistory(liveKeys: ["_step", "train/CrossEntropyLoss", "throughput/total_tokens"]) {
            parquetUrls
          }
        }
      }
    }
    """
    data = graphql(query, {"run": run_name})
    return data["project"]["run"]["parquetHistory"]["parquetUrls"]


def read_run_history(run: dict) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for url in parquet_urls(run["name"]):
        frame = pd.read_parquet(url)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    keep = ["_step", "train/CrossEntropyLoss", "throughput/total_tokens"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df = df.rename(
        columns={
            "_step": "step",
            "train/CrossEntropyLoss": "loss",
            "throughput/total_tokens": "tokens",
        }
    )
    df["run_id"] = run["name"]
    df["run_name"] = run["displayName"]
    df["created_at"] = run["createdAt"]
    return df


def main() -> None:
    fetch_report_spec()
    runs = fetch_stage1_runs()

    # Only the first two run segments are needed for 0-5.238B tokens.
    needed = [run for run in runs if run["name"] in {"gfm5tift", "y8brg36d"}]
    needed.sort(key=lambda run: run["createdAt"])

    history = pd.concat([read_run_history(run) for run in needed], ignore_index=True)
    history = history.dropna(subset=["tokens", "loss"]).copy()
    history["tokens_B"] = history["tokens"] / 1e9
    history = history[history["tokens"] <= TARGET_TOKENS].sort_values(["tokens", "created_at"])
    # Later segments override duplicated token/step neighborhoods after resume.
    history = history.groupby("tokens", as_index=False).tail(1).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(OUT, index=False)
    print(f"saved {OUT}")
    print(
        history[["run_id", "step", "tokens_B", "loss"]]
        .agg({"tokens_B": ["min", "max"], "loss": ["min", "max"], "step": ["min", "max"]})
        .to_string()
    )
    print(history.tail(5)[["run_id", "step", "tokens_B", "loss"]].to_string(index=False))


if __name__ == "__main__":
    main()
