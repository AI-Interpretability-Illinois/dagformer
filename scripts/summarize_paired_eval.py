"""Paired document uncertainty for saved baseline/DAGFormer lm-eval results.

Consumes --save-likelihood-samples output. Positive differences always favor
DAGFormer. Confidence intervals resample documents, not training runs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "experiments/results/lmeval"))
from suites import primary_metric


def read_samples(path, filter_name):
    samples = {}
    with Path(path).open() as file:
        for line in file:
            row = json.loads(line)
            if row["filter"] == filter_name:
                samples[row["doc_id"]] = row
    return samples


def paired_difference(base, dag, metric, draws=10000, seed=0):
    """Return the original aggregation and a paired document-bootstrap CI."""
    metric_name, filter_name = metric.split(",", 1)
    if set(base) != set(dag):
        raise ValueError("The two sample files contain different document IDs")
    ids = sorted(base)
    for doc in ids:
        if base[doc].get("doc_hash") != dag[doc].get("doc_hash"):
            raise ValueError(f"Document content differs for ID {doc}")
        if base[doc].get("prompt_hash") != dag[doc].get("prompt_hash"):
            raise ValueError(f"Evaluation prompt differs for ID {doc}")
    n = len(ids)
    rng = np.random.default_rng(seed)
    if metric_name == "bits_per_byte":
        b = np.array([base[i][metric_name] for i in ids], dtype=float)
        d = np.array([dag[i][metric_name] for i in ids], dtype=float)
        if not np.array_equal(b[:, 1], d[:, 1]):
            raise ValueError("Byte denominators differ between models")
        byte_count = b[:, 1]
        numerator = (d[:, 0] - b[:, 0]) / math.log(2)
        delta = numerator.sum() / byte_count.sum()
        boot = []
        for start in range(0, draws, 128):
            indexes = rng.integers(n, size=(min(128, draws - start), n))
            boot.extend((numerator[indexes].sum(1) / byte_count[indexes].sum(1)).tolist())
        extra = {"baseline": float(-b[:, 0].sum() / byte_count.sum() / math.log(2)),
                 "dagformer": float(-d[:, 0].sum() / byte_count.sum() / math.log(2)),
                 "bytes": int(byte_count.sum())}
    elif metric_name in ("acc", "acc_norm", "exact_match"):
        b = np.array([base[i][metric_name] for i in ids], dtype=float)
        d = np.array([dag[i][metric_name] for i in ids], dtype=float)
        difference = d - b
        counts = np.array([(difference == -1).sum(), (difference == 0).sum(),
                           (difference == 1).sum()])
        if counts.sum() != n:
            raise ValueError("Accuracy sample scores must be binary")
        sampled = rng.multinomial(n, counts / n, size=draws)
        boot = (sampled[:, 2] - sampled[:, 0]) / n
        delta = difference.mean()
        discordant = int(counts[0] + counts[2])
        extra = {"baseline": float(b.mean()), "dagformer": float(d.mean()),
                 "dag_only_correct": int(counts[2]), "base_only_correct": int(counts[0]),
                 "mcnemar_exact_p": float(binomtest(int(counts[2]), discordant).pvalue)
                 if discordant else 1.0}
    else:
        raise ValueError(f"Unsupported paired metric {metric}")
    return {"n_docs": n, "metric": metric,
            "delta_dagformer_better": float(delta),
            "paired_bootstrap_95ci": np.quantile(boot, [0.025, 0.975]).tolist(),
            **extra}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()
    pairs = {}
    for path in sorted(args.dir.glob("*__*.json")):
        payload = json.loads(path.read_text())
        if "model" not in payload or "results" not in payload:
            continue
        model = payload["model"]
        pairs.setdefault(model["size"], {})[model["family"]] = payload
    result = {"protocol": {"bootstrap": "paired resampling of evaluation documents",
                            "draws": args.draws, "seed": args.seed,
                            "scope": "evaluation-sample uncertainty for fixed trained checkpoints",
                            "multiplicity": "intervals and p-values are unadjusted"}, "pairs": {}}
    lines = ["# Paired ordinary evaluation", "",
             "Positive differences favor DAGFormer. Intervals resample the same documents",
             "for both models; they do not measure variation across training seeds.", "",
             "| Size | Task | Baseline | DAGFormer | Difference | Paired 95% CI |",
             "|---|---|---:|---:|---:|---|"]
    for size, families in sorted(pairs.items(), key=lambda item: float(item[0].rstrip("m"))):
        if not {"baseline", "dagformer"}.issubset(families):
            continue
        base, dag = families["baseline"], families["dagformer"]
        rec = {"baseline_model": base["model"], "dagformer_model": dag["model"],
               "tasks": {}}
        for task in base["results"]:
            if task not in dag["results"]:
                continue
            metric = primary_metric(task, base["results"][task])
            filter_name = metric.split(",", 1)[1]
            b = read_samples(args.dir / "samples" / f"{base['model']['name']}__{task}.jsonl", filter_name)
            d = read_samples(args.dir / "samples" / f"{dag['model']['name']}__{task}.jsonl", filter_name)
            stats = paired_difference(b, d, metric, args.draws, args.seed)
            for family in ("baseline", "dagformer"):
                reported = families[family]["results"][task][metric]
                if not math.isclose(stats[family], reported, abs_tol=1e-8):
                    raise ValueError(f"Sample aggregation disagrees with {family}/{task}")
            rec["tasks"][task] = stats
            lo, hi = stats["paired_bootstrap_95ci"]
            lines.append(f"| {size} | {task} | {stats['baseline']:.4f} | "
                         f"{stats['dagformer']:.4f} | {stats['delta_dagformer_better']:+.4f} | "
                         f"[{lo:+.4f}, {hi:+.4f}] |")
        result["pairs"][size] = rec
    (args.dir / "paired_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.dir / "paired_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
