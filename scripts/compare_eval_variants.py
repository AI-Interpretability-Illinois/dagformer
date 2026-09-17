"""Pair an ablation/intervention with an existing model on identical documents."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from summarize_paired_eval import paired_difference, primary_metric, read_samples


def load_evaluations(directory):
    result = []
    for path in sorted(directory.glob("*__*.json")):
        payload = json.loads(path.read_text())
        if "model" in payload and "results" in payload:
            result.append(payload)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-dir", required=True, type=Path)
    ap.add_argument("--variant-dir", required=True, type=Path)
    ap.add_argument("--reference-family", default="dagformer")
    ap.add_argument("--out-stem", help="output filename stem, useful for several references in one variant directory")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()
    references = {r["model"]["size"]: r for r in load_evaluations(args.reference_dir)
                  if r["model"]["family"] == args.reference_family}
    output = {"protocol": "paired document bootstrap; positive differences favor the variant; fixed checkpoints; unadjusted intervals",
              "draws": args.draws, "seed": args.seed, "variants": {}}
    lines = [f"# Paired comparison against {args.reference_family}", "",
             f"Reference results: `{args.reference_dir}`.", "",
             "Positive differences favor the variant. Intervals resample documents, "
             "not training seeds. BPB decreases and accuracy increases are positive.", "",
             "| Variant | Task | Reference | Variant | Difference | Paired 95% interval |",
             "|---|---|---:|---:|---:|---|"]
    for candidate in load_evaluations(args.variant_dir):
        model = candidate["model"]
        if model["size"] not in references:
            continue
        reference = references[model["size"]]
        rec = {"reference_model": reference["model"], "variant_model": model, "tasks": {}}
        for task in reference["results"]:
            if task not in candidate["results"]:
                continue
            metric = primary_metric(task, reference["results"][task])
            filter_name = metric.split(",", 1)[1]
            ref_samples = read_samples(args.reference_dir / "samples" /
                                       f"{reference['model']['name']}__{task}.jsonl", filter_name)
            cand_samples = read_samples(args.variant_dir / "samples" /
                                        f"{model['name']}__{task}.jsonl", filter_name)
            stats = paired_difference(ref_samples, cand_samples, metric, args.draws, args.seed)
            for label, source in (("baseline", reference), ("dagformer", candidate)):
                if not math.isclose(stats[label], source["results"][task][metric], abs_tol=1e-8):
                    raise ValueError(f"Sample aggregation disagrees with {source['model']['name']}/{task}")
            rename = {"baseline": "reference", "dagformer": "variant",
                      "delta_dagformer_better": "delta_variant_better",
                      "dag_only_correct": "variant_only_correct", "base_only_correct": "reference_only_correct"}
            stats = {rename.get(k, k): v for k, v in stats.items()}
            rec["tasks"][task] = stats
            lo, hi = stats["paired_bootstrap_95ci"]
            lines.append(f"| {model['name']} | {task} | {stats['reference']:.4f} | "
                         f"{stats['variant']:.4f} | {stats['delta_variant_better']:+.4f} | "
                         f"[{lo:+.4f}, {hi:+.4f}] |")
        output["variants"][model["name"]] = rec
    stem = args.out_stem or "paired_vs_" + args.reference_family
    (args.variant_dir / (stem + ".json")).write_text(json.dumps(output, indent=2) + "\n")
    (args.variant_dir / (stem + ".md")).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
