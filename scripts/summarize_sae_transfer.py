"""Summarize fixed SAE-direction transfer, including paired dose contrasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def span_summary(minus, plus, draws, seed):
    counts = np.asarray(plus["window_token_count"])
    if not np.array_equal(counts, minus["window_token_count"]):
        raise ValueError("Dose arms must score exactly the same target tokens")
    sums = np.asarray(plus["window_delta_sum"]) - np.asarray(minus["window_delta_sum"])
    if counts.sum() == 0:
        return {"delta": None, "paired_bootstrap_95ci": None}
    indexes = np.random.default_rng(seed).integers(len(counts), size=(draws, len(counts)))
    den = counts[indexes].sum(1)
    keep = den > 0
    values = sums[indexes[keep]].sum(1) / den[keep]
    return {"delta": float(sums.sum() / counts.sum()),
            "paired_bootstrap_95ci": np.quantile(values, [.025, .975]).tolist()}


def effect(value):
    if value["delta"] is None:
        return "no target tokens"
    lo, hi = value["paired_bootstrap_95ci"]
    return f"{value['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--old-screen", type=Path, required=True)
    ap.add_argument("--tokenizer", required=True)
    args = ap.parse_args()
    payload = json.loads(args.input.read_text())
    if not payload["complete"]:
        raise ValueError("The SAE transfer run has not finished")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    old = {str(r["feature"]): r for r in json.loads(args.old_screen.read_text())}
    lines = ["# Transfer of the eight previously selected SAE directions", "",
             "These are the fixed directions used in the historical large-generation check.",
             "No direction or token set was selected on the new WikiText test windows.",
             "Edits act on the correction channel at all positions; alpha units match the",
             "original mean active feature coefficient. NLL changes are token-weighted.",
             "Intervals resample the 128 windows in paired form, with 10,000 draws; they",
             "are unadjusted and do not cover training-seed or feature-selection uncertainty.", "",
             "Negative target-token ΔNLL means that those gold tokens become more probable.",
             "The signed dose span is ΔNLL(+4) − ΔNLL(−4), calculated from paired windows.", "",
             "| Feature | Target tokens | Target count | ΔNLL at −4 [95% CI] | ΔNLL at +4 [95% CI] | Dose span [95% CI] |",
             "|---|---|---:|---|---|---|"]
    secondary = ["", "## Original screen, other tokens and permuted controls", "",
                 "Controls permute each direction within every layer and Q/K/V/R stream.",
                 "They preserve coefficient values and norms. The five-control span range",
                 "is descriptive and is not a confidence interval. Cross-corpus differences",
                 "also change which target tokens occur and their contexts.", "",
                 "| Feature | Original target ΔNLL −4 / +4 | New other-token ΔNLL −4 / +4 | Five permuted target-span range |",
                 "|---|---|---|---|"]
    summary = {"source": str(args.input), "old_screen": str(args.old_screen), "features": {}}
    for feature_id, rec in payload["features"].items():
        arms = rec["arms"]
        minus, plus = arms["feature/alpha-4"], arms["feature/alpha4"]
        span = span_summary(minus["on_rule"], plus["on_rule"], 10000, payload["args"]["seed"])
        tokens = tokenizer.convert_ids_to_tokens(rec["rule"][1])
        controls = []
        for index in range(payload["args"]["controls"]):
            value = span_summary(arms[f"random{index}/alpha-4"]["on_rule"],
                                 arms[f"random{index}/alpha4"]["on_rule"], 10000, payload["args"]["seed"])
            controls.append(value)
        valid = [c["delta"] for c in controls if c["delta"] is not None]
        bounds = [min(valid), max(valid)] if valid else None
        summary["features"][feature_id] = {"tokens": tokens, "on_minus": minus["on_rule"],
            "on_plus": plus["on_rule"], "off_minus": minus["off_rule"], "off_plus": plus["off_rule"],
            "span": span, "control_spans": controls, "control_span_range": bounds,
            "original_screen": old[feature_id]}
        token_text = ", ".join(tokens).replace("|", "\\|")
        lines.append(f"| {feature_id} | {token_text} | {plus['on_rule']['n_tokens']} | "
                     f"{effect(minus['on_rule'])} | {effect(plus['on_rule'])} | {effect(span)} |")
        bound_text = f"[{bounds[0]:+.4f}, {bounds[1]:+.4f}]" if bounds else "no target tokens"
        historical = old[feature_id]
        secondary.append(f"| {feature_id} | {historical['on-4']:+.4f} / {historical['on+4']:+.4f} | "
                         f"{minus['off_rule']['delta']:+.4f} / {plus['off_rule']['delta']:+.4f} | {bound_text} |")
    output = args.input.parent
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "README.md").write_text("\n".join(lines + secondary) + "\n")
    print("\n".join(lines + secondary))


if __name__ == "__main__":
    main()
