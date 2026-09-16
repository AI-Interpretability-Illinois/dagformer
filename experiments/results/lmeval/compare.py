#!/usr/bin/env python
"""Turn the per-model result JSONs into a baseline-vs-DAGFormer table.

    python compare.py                      # markdown to stdout
    python compare.py --write              # also comparison.md + comparison.csv
    python compare.py --dir reasoning --suite reasoning

Reads every ``<model>__<suite>.json`` written by ``run_eval.py``, pairs the two
families at each size, and reports the primary metric per task with its
standard error and the DAGFormer-minus-baseline delta (sign-corrected, so a
positive delta always means DAGFormer is better, including for bits-per-byte
where lower is better).

On significance: the two models are scored on the *same* documents, so the
per-model standard errors are conservative for a paired comparison — but the
per-document scores needed for a proper paired test are not kept for
log-likelihood tasks.  A delta is flagged ``*`` when it exceeds twice the
quadrature-combined standard error, which is the strict reading.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from models import checkpoint_step  # noqa: E402
from suites import higher_is_better, primary_metric  # noqa: E402


def sign_test_p(wins: int, losses: int) -> float:
    """Two-sided binomial p for `wins` of `wins+losses` deltas favouring one side.

    Each task's delta is individually inside its own error bars; what carries
    information is whether the *signs* line up across independent tasks. Ties
    are dropped, which is the standard sign test.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    extreme = max(wins, losses)
    tail = sum(math.comb(n, k) for k in range(extreme, n + 1)) / 2**n
    return min(1.0, 2 * tail)


def load_results(directory: Path, suite: str | None) -> list[dict]:
    payloads = []
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        if "model" not in payload or "results" not in payload:
            continue
        if suite and payload.get("eval", {}).get("suite") != suite:
            continue
        payload["_path"] = str(path)
        payloads.append(payload)
    return payloads


def score(payload: dict, task: str) -> tuple[str | None, float | None, float | None]:
    """(metric key, value, stderr) for ``task``, or (None, None, None) if absent."""
    metrics = payload.get("results", {}).get(task)
    if not metrics:
        return None, None, None
    key = primary_metric(task, metrics)
    if key is None:
        return None, None, None
    stderr = metrics.get(key.replace(",", "_stderr,", 1))
    value = metrics.get(key)
    return key, (None if value is None else float(value)), (
        None if not isinstance(stderr, (int, float)) else float(stderr)
    )


def fmt(value: float | None, stderr: float | None) -> str:
    if value is None:
        return "—"
    if stderr is None or math.isnan(stderr):
        return f"{value:.4f}"
    return f"{value:.4f} ± {stderr:.4f}"


def delta_cell(task: str, key: str, base: tuple, dag: tuple) -> tuple[str, float | None]:
    """DAGFormer-minus-baseline, sign-corrected so positive = DAGFormer better."""
    _, b_val, b_err = base
    _, d_val, d_err = dag
    if b_val is None or d_val is None:
        return "—", None
    raw = d_val - b_val
    signed = raw if higher_is_better(key) else -raw
    combined = math.sqrt((b_err or 0.0) ** 2 + (d_err or 0.0) ** 2)
    mark = "*" if combined > 0 and abs(raw) > 2 * combined else ""
    return f"{signed:+.4f}{mark}", signed


def build_tables(payloads: list[dict]) -> tuple[list[str], dict]:
    by_size: dict[str, dict[str, dict]] = {}
    for payload in payloads:
        model = payload["model"]
        by_size.setdefault(model["size"], {})[model["family"]] = payload

    def size_key(size: str) -> float:
        text = size.lower()
        scale = 1000.0 if text.endswith("b") else 1.0
        try:
            return float(text.rstrip("mb")) * scale
        except ValueError:
            return float("inf")

    tasks: list[str] = []
    for payload in payloads:
        for task in payload.get("results", {}):
            if task not in tasks:
                tasks.append(task)

    lines: list[str] = []
    rows: list[dict] = []
    wins = {"dagformer": 0, "baseline": 0, "tie": 0}

    for size in sorted(by_size, key=size_key):
        families = by_size[size]
        base = families.get("baseline")
        dag = families.get("dagformer")
        lines.append(f"\n### {size}\n")
        if base is None or dag is None:
            only = base or dag
            lines.append(f"_only `{only['model']['name']}` evaluated at this size_\n")
            lines.append("| task | metric | " + only["model"]["family"] + " |")
            lines.append("|---|---|---|")
            for task in tasks:
                key, val, err = score(only, task)
                if key is None:
                    continue
                lines.append(f"| {task} | `{key}` | {fmt(val, err)} |")
                rows.append({
                    "size": size, "task": task, "metric": key,
                    only["model"]["family"]: val, f"{only['model']['family']}_stderr": err,
                })
            continue

        lines.append("| task | metric | baseline | dagformer | Δ (dagformer better = +) |")
        lines.append("|---|---|---|---|---|")
        for task in tasks:
            b = score(base, task)
            d = score(dag, task)
            key = b[0] or d[0]
            if key is None:
                continue
            cell, signed = delta_cell(task, key, b, d)
            lines.append(f"| {task} | `{key}` | {fmt(b[1], b[2])} | {fmt(d[1], d[2])} | {cell} |")
            rows.append({
                "size": size, "task": task, "metric": key,
                "baseline": b[1], "baseline_stderr": b[2],
                "dagformer": d[1], "dagformer_stderr": d[2],
                "delta_dagformer_better": signed,
            })
            if signed is None:
                continue
            wins["dagformer" if signed > 0 else "baseline" if signed < 0 else "tie"] += 1

    return lines, {"rows": rows, "wins": wins, "tasks": tasks, "by_size": by_size}


def render(payloads: list[dict]) -> tuple[str, list[dict]]:
    if not payloads:
        return "no result JSONs found — run run_eval.py first", []

    lines = ["# DAGFormer vs baseline — lm-evaluation-harness", ""]
    first = payloads[0]["eval"]
    lines += [
        f"- suite: `{first.get('suite')}` · lm-eval {first.get('lm_eval_version')} · "
        f"context {first.get('max_length')} tokens",
        f"- generative tasks limited to {first.get('gen_limit')} docs "
        f"at {first.get('gen_num_fewshot')}-shot; log-likelihood tasks use the full split "
        f"(limit {first.get('limit')})",
        "- `*` = |Δ| exceeds twice the quadrature-combined standard error",
        "- for `bits_per_byte` lower is better, so Δ is sign-corrected: + always favours DAGFormer",
        "",
        "| model | params | train steps | eval seconds | peak GPU GB |",
        "|---|---|---|---|---|",
    ]
    steps: dict[str, int | None] = {}
    for payload in payloads:
        model = payload["model"]
        # Older result files predate the `step` field; read it off the checkpoint.
        step = model.get("step") or checkpoint_step(model["checkpoint"])
        steps[model["name"]] = step
        seconds = sum((payload["eval"].get("seconds") or {}).values())
        lines.append(
            f"| {model['name']} | {model['num_parameters']:,} | {step} | "
            f"{seconds:.0f} | {payload['eval'].get('peak_gpu_gb')} |"
        )

    body, meta = build_tables(payloads)

    # Call out pairs trained for different numbers of steps — the comparison is
    # only iso-token when they match.
    mismatched = []
    for size, families in meta["by_size"].items():
        if len(families) < 2:
            continue
        pair = {f: steps.get(p["model"]["name"]) for f, p in families.items()}
        if len(set(pair.values())) > 1:
            mismatched.append(
                f"{size}: " + ", ".join(f"{f} {v} steps" for f, v in sorted(pair.items()))
            )
    if mismatched:
        lines += [
            "",
            "> **Not step-matched:** " + "; ".join(mismatched) + ". "
            "The shorter-trained model saw proportionally fewer tokens, so a win there "
            "is a lower bound and a loss is not conclusive.",
        ]

    lines += body
    wins = meta["wins"]
    p_value = sign_test_p(wins["dagformer"], wins["baseline"])
    lines += [
        "",
        f"**Head-to-head:** DAGFormer better on {wins['dagformer']} of "
        f"{sum(wins.values())} paired task-size cells, baseline on {wins['baseline']}"
        + (f", {wins['tie']} tied" if wins["tie"] else "")
        + f" (sign test over the {wins['dagformer'] + wins['baseline']} non-tied cells, "
        f"p = {p_value:.3f}).",
        "",
        "Individual multiple-choice deltas at this scale are inside their own error bars; "
        "the sign agreement across tasks is what carries the information. Note the cells "
        "are not fully independent — the same model pair is scored on every task.",
    ]
    return "\n".join(lines), meta["rows"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=HERE / "reasoning")
    p.add_argument("--suite", default=None, help="only include runs of this suite")
    p.add_argument("--write", action="store_true", help="write comparison.md and comparison.csv")
    args = p.parse_args(argv)

    payloads = load_results(args.dir, args.suite)
    markdown, rows = render(payloads)
    print(markdown)

    if args.write and rows:
        (args.dir / "comparison.md").write_text(markdown + "\n")
        fields = sorted({k for row in rows for k in row})
        with (args.dir / "comparison.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.dir / 'comparison.md'} and {args.dir / 'comparison.csv'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
