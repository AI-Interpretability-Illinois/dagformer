"""Compare the trained 150M routing variants on identical natural and copy inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_context_fidelity import paired_summary


def formatted(stats, scale=1.):
    lo, hi = stats["normal_95ci"]
    return f"{scale * stats['delta']:+.4f} [{scale * lo:+.4f}, {scale * hi:+.4f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, required=True,
                    help="Campaign directory containing routing_dependence and ordinary results")
    args = ap.parse_args()
    root = args.results / "routing_dependence"
    full_path = root / "150m.json"
    full = json.loads(full_path.read_text())
    reference = full["arms"]["pred_dynamic/corr_dynamic"]
    variants = {"dense": root / "dense_150m.json"}
    variants.update({name: root / f"ladder_150m_{name}.json"
                     for name in ("static", "postable", "identcorr", "staticcorr", "lite")})
    output = {"protocol_source": str(full_path), "reference": "full 150M DAGFormer",
              "difference": "variant minus full DAGFormer; lower NLL and higher accuracy are better",
              "variants": {}}
    periods = full["args"]["periods"]
    lines = ["# Copy behavior of the trained 150M routing ladder", "",
             "All models use the same 128 WikiText test windows and 16 synthetic sequences",
             "at each repetition period. Token blocks are sampled from the same training-token",
             "marginal. Copy accuracy scores teacher-forced predictions in the second half",
             "of each sequence, following the original routing-dependence protocol.", "",
             "All checkpoints were trained for 6,000 updates / 3.146B tokens. They have",
             "different parameter counts; this is neither a matched-FLOPs nor a speed test.",
             "Each cell's difference is variant minus full DAGFormer. Intervals are unadjusted",
             "paired normal intervals across the same windows or synthetic sequences, not",
             "uncertainty across training seeds.", "",
             "| Model | Parameters | Natural-text NLL | NLL difference [95% interval] |",
             "|---|---:|---:|---|"]
    metadata = json.loads((args.results / "standard_matched" / "150m-dagformer__custom.json").read_text())["model"]
    lines.append(f"| full DAGFormer | {metadata['num_parameters']:,} | {np.mean(reference['nll']):.6f} | reference |")
    copy_lines = ["", "| Model | Period | Copy accuracy | Accuracy difference (points) [95% interval] | Copy NLL difference [95% interval] |",
                  "|---|---:|---:|---|---|"]
    for period in periods:
        copy_lines.append(f"| full DAGFormer | {period} | {100 * np.mean(reference['copy'][str(period)]['accuracy']):.3f}% | reference | reference |")
    for name, path in variants.items():
        raw = json.loads(path.read_text())
        if Path(raw["args"]["routing_result"]).resolve() != full_path.resolve():
            raise ValueError(f"{name}: different reference-input specification")
        family = "baseline" if name == "dense" else name
        folder = "standard_matched" if name == "dense" else "standard_ladder"
        expected = json.loads((args.results / folder / f"150m-{family}__custom.json").read_text())["model"]
        if raw["num_parameters"] != expected["num_parameters"]:
            raise ValueError(f"{name}: parameters disagree with the ordinary-evaluation model")
        if Path(raw["args"]["ckpt"]).resolve() != Path(expected["checkpoint"]).resolve():
            raise ValueError(f"{name}: checkpoint differs from the ordinary-evaluation model")
        if len(raw["nll"]) != len(reference["nll"]):
            raise ValueError(f"{name}: different natural-text sample count")
        rec = {"source": str(path), "num_parameters": raw["num_parameters"],
               "natural_nll": paired_summary(raw["nll"], reference["nll"]), "copy": {}}
        lines.append(f"| {name} | {raw['num_parameters']:,} | {np.mean(raw['nll']):.6f} | {formatted(rec['natural_nll'])} |")
        for period in periods:
            key = str(period)
            stats = {}
            for metric in ("accuracy", "nll"):
                if len(raw["copy"][key][metric]) != len(reference["copy"][key][metric]):
                    raise ValueError(f"{name}: different period-{period} sample count")
                stats[metric] = paired_summary(raw["copy"][key][metric], reference["copy"][key][metric])
            rec["copy"][key] = stats
            copy_lines.append(f"| {name} | {period} | {100 * np.mean(raw['copy'][key]['accuracy']):.3f}% | "
                              f"{formatted(stats['accuracy'], 100)} | {formatted(stats['nll'])} |")
        output["variants"][name] = rec
    lines += copy_lines + ["", "Static and identity labels describe the external predictor. Variants with",
                           "corrections still have input-dependent routing through the local hidden states.", ""]
    (root / "copy_ladder.json").write_text(json.dumps(output, indent=2) + "\n")
    (root / "copy_ladder.md").write_text("\n".join(lines).rstrip() + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
