"""Render fixed-copy-head effects, measured controls and pattern-break costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_context_fidelity import paired_summary


def effect(stats, scale=1):
    lo, hi = stats["normal_95ci"]
    return f"{scale*stats['delta']:+.3f} [{scale*lo:+.3f}, {scale*hi:+.3f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    args = ap.parse_args()
    source = json.loads(args.input.read_text())
    if not source["complete"]:
        raise ValueError("Copy-head transfer evaluation is incomplete")
    settings = source["args"]
    reference = source["arms"]["reference"]
    summary = {"source": str(args.input), "protocol": source["protocol"],
               "reference": reference["summary"], "named_arms": {}}
    lines = ["# Transfer of the six historical copy heads", "",
             "The head set is fixed from the original localization: " + ", ".join(source["named_heads"]) + ".",
             "No heads are reselected on these inputs. Q/K head deviations are scaled in",
             "the predictor, correction, or both channels. Controls use the same number",
             "of other heads per layer, matching edit norms separately per layer/token.", "",
             f"Each period uses {settings['n_sequences']} new {settings['seq_len']}-token sequences, formed by repeating",
             "blocks sampled from the WikiText training-token marginal. They are synthetic",
             "token sequences, not contiguous natural text. Accuracy is teacher-forced",
             "next-token accuracy beginning at the first token of the second block.",
             "The historical localization used a Dolma-token marginal and a different seed;",
             "absolute reference accuracies across the two runs are not directly paired.",
             "Natural-text NLL uses WikiText validation windows. Brackets are paired normal",
             "95% intervals over sequences/windows, unadjusted for multiple comparisons.", "",
             "## Unedited reference", "",
             "| Period | Accuracy after first block | Accuracy over second half |",
             "|---|---:|---:|"]
    for p in settings["periods"]:
        first = reference["summary"][f"period{p}/after_first_block/accuracy"]["mean"]
        second = reference["summary"][f"period{p}/second_half/accuracy"]["mean"]
        lines.append(f"| {p} | {100*first:.2f}% | {100*second:.2f}% |")
    lines += ["", f"Reference natural-text NLL: {reference['summary']['natural_nll']['mean']:.6f}.", "",
              "## Named-head accuracy changes", "",
              "Accuracy changes are percentage points. Gamma below one weakens head differences;",
              "gamma above one amplifies them. All tested doses are shown.", "",
              "| Channel | Gamma | " + " | ".join(f"Period {p}" for p in settings["periods"]) + " | Natural NLL rise |",
              "|---|---:|" + "---|" * (len(settings["periods"]) + 1)]
    control_lines = ["# Named heads and fixed random controls", "",
                     "The range describes the measured random-head effects, not uncertainty",
                     "over all possible control circuits. The final column gives the smallest",
                     "and largest lower/upper bounds among direct paired named-minus-control",
                     "intervals. Individual comparisons are retained in summary.json. The NLL",
                     "columns show whether norm matching also gives similar capability costs.", "",
                     "| Channel | Gamma | Period | Named change (points) | Control change range | Paired difference interval envelope | Named NLL rise | Control NLL rise range |",
                     "|---|---:|---:|---:|---|---|---:|---|"]
    tail = ["", "## Repetition interrupted by new tokens", "",
            "After three copies of a 128-token block, the remaining tokens are independent",
            "samples from the same marginal. True next-token NLL measures adaptation to the",
            "new tail. False-lag probability and argmax rates refer to the token 128 positions",
            "back, only where it differs from the actual target. Higher false-lag scores",
            "indicate more copying of a now-wrong token, not better prediction.", "",
            f"Unedited tail NLL is {reference['summary']['pattern_break/nll']['mean']:.4f}; "
            f"false-lag probability is {100 * reference['summary']['pattern_break/false_lag_probability']['mean']:.3f}% "
            f"and false-lag argmax rate is {100 * reference['summary']['pattern_break/false_lag_argmax']['mean']:.3f}%.", "",
            "| Channel | Gamma | Tail NLL change | False-lag probability change (points) | False-lag argmax change (points) |",
            "|---|---:|---|---|---|"]
    for channel in ("pred", "corr", "both"):
        for gamma in settings["gammas"]:
            key = f"{channel}/named@{gamma:g}"
            arm = source["arms"][key]
            rec = {"summary": arm["summary"], "paired_named_minus_control": {}}
            control_costs = []
            for i in range(settings["controls"]):
                control = source["arms"][f"{channel}/random{i}@{gamma:g}"]
                control_costs.append(control['summary']['natural_nll']['delta'])
                rec["paired_named_minus_control"][str(i)] = {
                    metric: paired_summary(values, control["values"][metric])
                    for metric, values in arm["values"].items()}
            summary["named_arms"][key] = rec
            cells = [effect(arm["summary"][f"period{p}/after_first_block/accuracy"], 100)
                     for p in settings["periods"]]
            lines.append(f"| {channel} | {gamma:g} | " + " | ".join(cells)
                         + " | " + effect(arm["summary"]["natural_nll"]) + " |")
            for p in settings["periods"]:
                metric = f"period{p}/after_first_block/accuracy"
                values = [source["arms"][f"{channel}/random{i}@{gamma:g}"]["summary"][metric]["delta"]
                          for i in range(settings["controls"])]
                intervals = [v[metric]["normal_95ci"] for v in rec["paired_named_minus_control"].values()]
                lower, upper = min(v[0] for v in intervals), max(v[1] for v in intervals)
                control_lines.append(f"| {channel} | {gamma:g} | {p} | {100*arm['summary'][metric]['delta']:+.3f} | "
                                     f"[{100*min(values):+.3f}, {100*max(values):+.3f}] | [{100*lower:+.3f}, {100*upper:+.3f}] | "
                                     f"{arm['summary']['natural_nll']['delta']:+.4f} | "
                                     f"[{min(control_costs):+.4f}, {max(control_costs):+.4f}] |")
            tail.append(f"| {channel} | {gamma:g} | {effect(arm['summary']['pattern_break/nll'])} | "
                        f"{effect(arm['summary']['pattern_break/false_lag_probability'], 100)} | "
                        f"{effect(arm['summary']['pattern_break/false_lag_argmax'], 100)} |")
    lines += tail + ["", "[Measured random controls](control_comparisons.md) and all per-sequence",
                     "values remain available. These are fixed-checkpoint interventions, not",
                     "a test of what separately retrained architectures can learn.", ""]
    output = args.input.parent
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "README.md").write_text("\n".join(lines))
    (output / "control_comparisons.md").write_text("\n".join(control_lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
