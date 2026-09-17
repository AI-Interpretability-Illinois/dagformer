"""Compare candidate ratios with full-vocabulary probabilities and cue effects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_context_fidelity import paired_summary


def metrics(values):
    return {"candidate_p_true": values["p_true"],
            "candidate_accuracy": values["correct"],
            "whole_vocab_p_true": np.exp(values["logp_true"])}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arm", default="both/circuit/gamma1.25")
    args = ap.parse_args()
    output = {"arm": args.arm,
              "protocol": "paired item differences; whole-vocabulary probability is exp(logp_true), then averaged; candidate accuracy is argmax among candidate values only",
              "sources": {}}
    lines = ["# Context metric diagnostics", "",
             f"Fixed edit: `{args.arm}`. Whole-vocabulary probability averages the",
             "per-item probabilities, not the exponentiated mean log probability.", "",
             "| Source | Cue | Metric | Reference | Edited | Difference | Paired 95% interval |",
             "|---|---|---|---:|---:|---:|---|"]
    for path in args.inputs:
        raw = json.loads(path.read_text())
        ref = raw["reference"]["cloze"]
        edited = raw["arms"][args.arm]["cloze"]
        rec = {"source": str(path), "n_items": len(raw["items"]), "cues": {}}
        for cue in ref:
            r, e = metrics(ref[cue]), metrics(edited[cue])
            rec["cues"][cue] = {}
            for metric, values in e.items():
                stats = paired_summary(values, r[metric])
                stats["reference_mean"] = float(np.mean(r[metric]))
                rec["cues"][cue][metric] = stats
                lo, hi = stats["normal_95ci"]
                lines.append(f"| {path.stem.removeprefix('context_validation_')} | {cue} | {metric} | "
                             f"{stats['reference_mean']:.6g} | {stats['mean']:.6g} | "
                             f"{stats['delta']:+.6g} | [{lo:+.6g}, {hi:+.6g}] |")
        if {"neutral", "deceptive"}.issubset(ref):
            ref_gap = np.array(ref["deceptive"]["p_true"]) - ref["neutral"]["p_true"]
            edit_gap = np.array(edited["deceptive"]["p_true"]) - edited["neutral"]["p_true"]
            rec["deceptive_minus_neutral"] = paired_summary(edit_gap, ref_gap)
            rec["deceptive_minus_neutral"]["reference_mean"] = float(ref_gap.mean())
        output["sources"][path.stem] = rec
    lines += ["", "Intervals describe variation over these paired content items, not training seeds.",
              "Candidate accuracy is not unrestricted next-token or generated-answer accuracy.",
              "In a QA prompt, articles or other opening words may precede the factual value;",
              "a low whole-vocabulary probability at the first position makes the candidate",
              "ratio an incomplete measure of answer retrieval. See the separate generation evaluation.", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n")
    args.out.with_suffix(".md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
