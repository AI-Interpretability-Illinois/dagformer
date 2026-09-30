"""Collect observed scaling endpoints and render six panels, leaving missing data empty."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(path, default=None):
    return json.loads(Path(path).read_text()) if Path(path).exists() else default


def collect(manifests, legacy):
    rows, states = [], {}
    collected = {}
    for path in (ROOT / "experiments/scaling").glob("collected_*.json"):
        collected.update({r["name"]: r for r in read_json(path)["runs"]})
    for pair in read_json(ROOT / "experiments/scaling_ebt_20260930/pr_review.json")["pairs"]:
        for prefix, family in (("dense", "dense"), ("dag", "fourway_corrected")):
            record = collected[pair[f"{prefix}_run"]]
            rows.append({"name": record["name"], "family": family, "group": pair["corpus"],
                         "source": "PR #3", "panels": ["parameters", "compute"],
                         "total_params": pair[f"{prefix}_total_params"],
                         "training_tokens": pair[f"{prefix}_recorded_tokens"],
                         "analytic_training_flops": record["train_flops_total"],
                         "wikitext2_nll": pair[f"{prefix}_wikitext_token_nll"]})
    for manifest_path in manifests:
        manifest_path = Path(manifest_path)
        manifest = read_json(manifest_path)
        # Supports both live roots and rsync copies of remote metadata.
        root = manifest_path.parent
        for run in manifest["runs"]:
            meta = root / "runs" / run["name"]
            status = read_json(meta / "status.json", {"status": "pending"})
            evaluation = read_json(meta / "evaluation.json")
            entry = {"name": run["name"], "panels": run["panels"],
                     "status": status["status"], "host": status.get("host"),
                     "job": status.get("slurm_job"), "planned_tokens": run["tokens"],
                     "layers": run["layers"], "width": run["width"],
                     "batch_sequences": run["batch_sequences"],
                     "optimizer_updates": run["optimizer_updates"]}
            curve = meta / "metrics.csv"
            if curve.exists():
                logs = list(csv.DictReader(curve.open()))
                tokens = [r for r in logs if r.get("train/tokens_seen_B")]
                if tokens:
                    entry["last_tokens"] = int(float(tokens[-1]["train/tokens_seen_B"]) * 1e9)
                    entry["last_step"] = int(tokens[-1]["step"])
            prior = states.get(run["name"])
            if prior is None or (prior["status"] == "pending" and entry["status"] != "pending"):
                states[run["name"]] = entry
            if evaluation is None or not (meta / "DONE.json").exists():
                continue
            if any(r["name"] == run["name"] for r in rows):
                continue
            states[run["name"]] = entry
            rows.append({"name": run["name"], "family": run["family"], "group": "new21b",
                         "source": str(meta / "evaluation.json"), "panels": run["panels"],
                         "layers": run["layers"], "width": run["width"],
                         "batch_sequences": run["batch_sequences"],
                         **{k: evaluation[k] for k in ("total_params", "non_embedding_params",
                                                      "training_tokens", "analytic_training_flops")},
                         **{f"{k}_nll": v["nll"] for k, v in evaluation["evals"].items()}})
    if legacy:
        for path in Path(legacy).glob("*/evaluation.json"):
            result = read_json(path)
            routed = "dagformer" in path.parent.name
            size = path.parent.name.split("-")[0]
            # Continuation corpora differ from each other and from the PR ladder.
            group = "olmomix20b" if size == "1b" else "dolma600m_continuation"
            rows.append({"name": path.parent.name, "family": "fourway_corrected" if routed else "dense",
                         "group": group, "source": str(path), "panels": ["parameters", "compute"],
                         **{k: result[k] for k in ("total_params", "non_embedding_params",
                                                   "training_tokens", "analytic_training_flops")},
                         **{f"{k}_nll": v["nll"] for k, v in result["evals"].items()}})
    return rows, list(states.values())


def plot(rows, states, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), layout="constrained")
    panels = [("parameters", "total_params", 1e6, "Total parameters (M)"),
              ("compute", "analytic_training_flops", 1e18, "Analytic training FLOPs (×10¹⁸)"),
              ("data", "training_tokens", 1e9, "Training tokens (B)"),
              ("depth", "layers", 1, "Layers"),
              ("width", "width", 1, "Hidden width"),
              ("batch", "batch_sequences", 1, "Sequences per update")]
    colors = {"dense": "#666666", "fourway_corrected": "#0072B2"}
    labels = {"dense": "Dense", "fourway_corrected": "DAGFormer"}
    markers = {"shared12b": "o", "timan12b": "s", "delta21b": "^",
               "dolma600m_continuation": "D", "olmomix20b": "P", "new21b": "o"}
    for ax, (axis, key, divisor, xlabel) in zip(axes.flat, panels):
        selected = [r for r in rows if axis in r["panels"] and r.get("wikitext2_nll") is not None]
        if axis == "compute":
            selected += [r for r in rows if "data" in r["panels"] and r.get("wikitext2_nll") is not None]
        for group in dict.fromkeys(r["group"] for r in selected):
            for family in colors:
                points = sorted((r for r in selected if r["group"] == group and r["family"] == family), key=lambda r: r[key])
                if not points:
                    continue
                ax.plot([r[key] / divisor for r in points], [r["wikitext2_nll"] for r in points],
                        color=colors[family], marker=markers[group], markersize=6,
                        linestyle="--" if family == "dense" else "-", linewidth=1.3)
        n = sum(axis in s["panels"] and s["status"] == "complete" for s in states)
        total = sum(axis in s["panels"] for s in states)
        ax.set_title(axis.capitalize() + (f" · {n}/{total} complete" if total else " · existing ladder"))
        ax.set(xlabel=xlabel, ylabel="WikiText2 NLL (nats/token)")
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
        if not selected:
            ax.text(.5, .5, "Training pending / in progress", ha="center", va="center", transform=ax.transAxes)
            ax.set_yticks([])
            values = sorted({s["planned_tokens" if key == "training_tokens" else key] / divisor
                             for s in states if axis in s["panels"]})
            if values:
                pad = (max(values) - min(values)) * .08
                ax.set_xlim(min(values) - pad, max(values) + pad)
                ax.set_xticks(values, labels=[f"{v:g}" for v in values])
        if axis in ("parameters", "compute"):
            ax.set_xscale("log")
    family_handles = [Line2D([], [], color=colors[f], label=labels[f], linestyle="--" if f == "dense" else "-") for f in colors]
    used = set(r["group"] for r in rows)
    group_handles = [Line2D([], [], color="black", marker=m, linestyle="none", label=g)
                     for g, m in markers.items() if g in used]
    fig.legend(handles=family_handles + group_handles, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Six-axis scaling: observed endpoints\nSeparate lines for each training corpus; first-seed new runs", fontsize=13)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"six_axes.{extension}", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", action="append", required=True)
    p.add_argument("--legacy")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    rows, states = collect(args.manifest, args.legacy)
    payload = {"updated_utc": datetime.now(timezone.utc).isoformat(), "runs": states, "endpoints": rows}
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with (output / "endpoints.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    plot(rows, states, output)
    counts = {s: sum(r["status"] == s for r in states) for s in sorted({r["status"] for r in states})}
    print(json.dumps({"status_counts": counts, "endpoints": len(rows), "out": str(output)}))


if __name__ == "__main__":
    main()
