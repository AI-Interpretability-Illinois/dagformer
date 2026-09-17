"""Summarize fixed PR domain directions, transfer items, controls and NLL cost."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_domain_streams import paired_bootstrap


def interval(stats):
    lo, hi = stats["paired_bootstrap_95ci"]
    return f"{stats['delta']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--nll-limit", type=float, default=.05)
    args = ap.parse_args()
    source = json.loads(args.input.read_text())
    if not source["complete"]:
        raise ValueError("Wait for all domain-stream arms to finish")
    settings = source["args"]
    summary = {"source": str(args.input), "protocol": source["protocol"],
               "nll_limit": args.nll_limit, "channels": {}}
    lines = ["# Domain-code direction: stream and content transfer", "",
             "This exploratory follow-up keeps the PR #2 training-half directions and circuit",
             f"masks fixed. It tests {len(settings['streams'])} stream conditions at "
             f"{len(settings['lams'])} additive doses",
             f"({', '.join(f'{v:g}' for v in settings['lams'])}), with "
             f"{settings['controls']} within-layer/stream permuted controls fixed across doses.",
             "The new 32 content pairs reuse the original instruction templates. The original",
             "test set has eight content items; paraphrases are averaged within each item.", "",
             "The endpoint is mean per-token code log probability minus prose log probability.",
             "It does not measure generated code correctness. Natural-text NLL uses 50 WikiText",
             f"validation windows (reference NLL {np.mean(source['reference_nll']):.6f}). "
             "Intervals are unadjusted paired content/window bootstrap",
             "intervals, without training-seed uncertainty.", "",
             "## Reference behavior", "",
             "| Channel | Content set | Neutral score | Positive instruction | Negative instruction | Instruction gap |",
             "|---|---|---:|---:|---:|---:|"]
    for channel, data in source["channels"].items():
        ref = data["reference"]
        gaps = {}
        for name, vals in ref.items():
            means = {key: float(np.mean(v)) for key, v in vals.items()}
            gaps[name] = means["pos"] - means["neg"]
            lines.append(f"| {channel} | {name} | {means['neutral']:.4f} | {means['pos']:.4f} | "
                         f"{means['neg']:.4f} | {gaps[name]:.4f} |")
        summary["channels"][channel] = {"instruction_gaps": gaps, "circuit_arms": {},
                                        "largest_tested_transfer_shift_below_nll_limit": {}}

    lines += ["", "## All circuit doses", "",
              f"Each bracket is a paired 95% interval. The control range contains {settings['controls']}",
              "measured transfer shifts at the same dose, not a confidence interval.", "",
              "| Channel / stream | Dose | Original shift | New-content shift | New-control shift range | NLL rise |",
              "|---|---:|---|---|---|---|"]
    candidates = ["# Candidate likelihood components", "",
                  "These are changes from the neutral reference. A larger code-minus-prose",
                  "score can arise from either increasing code likelihood or decreasing prose",
                  "likelihood. Code-candidate preference is a two-candidate scoring endpoint.", "",
                  "| Channel / stream | Dose | Content set | Code logp change | Prose logp change | Code preference change (points) |",
                  "|---|---:|---|---:|---:|---:|"]
    controls_table = ["# Direct circuit-versus-control comparisons", "",
                      "All differences use the same content items. Intervals resample items,",
                      "not the population of possible random control directions. The last two",
                      "columns give each intervention's natural-text NLL rise against the same",
                      "reference. Equal direction norms need not imply equal capability costs.", "",
                      "| Channel / stream | Dose | Control | New-content score difference [95% interval] | Circuit NLL rise | Control NLL rise |",
                      "|---|---:|---|---|---:|---:|"]
    for channel, data in source["channels"].items():
        out = summary["channels"][channel]
        for key, arm in data["arms"].items():
            if arm["control"] != "circuit":
                continue
            stream, lam = arm["stream"], arm["lam"]
            others = {tag: data["arms"][f"{stream}/{tag}@{lam:g}"]
                      for tag in (f"permuted_{i}" for i in range(settings["controls"]))}
            comparisons = {}
            for tag, control in others.items():
                comparisons[tag] = {}
                for name in arm["behavior"]:
                    comparisons[tag][name] = paired_bootstrap(
                        arm["behavior"][name]["per_unit_delta"],
                        control["behavior"][name]["per_unit_delta"],
                        settings["draws"], settings["seed"])
                controls_table.append(f"| {channel}/{stream} | {lam:g} | {tag} | "
                                      f"{interval(comparisons[tag]['new_content'])} | "
                                      f"{arm['nll']['delta']:+.4f} | {control['nll']['delta']:+.4f} |")
            vals = [c["behavior"]["new_content"]["delta"] for c in others.values()]
            components = {}
            for name, behavior in arm["behavior"].items():
                ref = data["reference_candidates"][name]["neutral"]
                components[name] = {k: behavior[k] - ref[k] for k in ref}
                v = components[name]
                candidates.append(f"| {channel}/{stream} | {lam:g} | {name} | "
                                  f"{v['mean_logp_code']:+.4f} | {v['mean_logp_prose']:+.4f} | "
                                  f"{100*v['code_candidate_win_rate']:+.2f} |")
            out["circuit_arms"][key] = {
                "behavior": arm["behavior"], "nll": arm["nll"],
                "candidate_changes": components, "paired_circuit_minus_controls": comparisons,
                "control_nll": {tag: control['nll'] for tag, control in others.items()},
                "new_control_range": [min(vals), max(vals)],
                "fraction_of_instruction_gap": {
                    name: value["delta"] / out["instruction_gaps"][name]
                    if out["instruction_gaps"][name] else None
                    for name, value in arm["behavior"].items()}}
            lines.append(f"| {channel}/{stream} | {lam:g} | {interval(arm['behavior']['original_test'])} | "
                         f"{interval(arm['behavior']['new_content'])} | [{min(vals):+.4f}, {max(vals):+.4f}] | "
                         f"{interval(arm['nll'])} |")
        for stream in settings["streams"]:
            eligible = {k: v for k, v in out["circuit_arms"].items()
                        if k.startswith(stream + "/") and v["nll"]["delta"] <= args.nll_limit}
            if eligible:
                best = max(eligible, key=lambda k: eligible[k]["behavior"]["new_content"]["delta"])
                out["largest_tested_transfer_shift_below_nll_limit"][stream] = best
    lines += ["", "## Descriptive grid maxima", "",
              f"For each channel and stream, the following picks the largest new-content shift",
              f"among tested circuit doses whose **mean** NLL rise is at most {args.nll_limit:g}.",
              "Selection uses these outcomes, so this is exploratory and intervals are not",
              "adjusted for selecting a maximum. Passing the mean-cost criterion does not mean",
              "its entire uncertainty interval passes the criterion.", "",
              "| Channel / stream | Selected dose | New-content shift | Fraction of instruction gap | NLL rise |",
              "|---|---:|---:|---:|---:|"]
    for channel, out in summary["channels"].items():
        for stream, key in out["largest_tested_transfer_shift_below_nll_limit"].items():
            val = out["circuit_arms"][key]
            frac = val["fraction_of_instruction_gap"]["new_content"]
            lines.append(f"| {channel}/{stream} | {key.rsplit('@', 1)[1]} | "
                         f"{val['behavior']['new_content']['delta']:+.4f} | "
                         f"{100*frac:.2f}% | {val['nll']['delta']:+.4f} |")
    lines += ["", "[Candidate likelihood components](candidate_components.md) distinguish changes",
              "to the code and prose scores. [Direct control comparisons](control_comparisons.md)",
              "retain paired item uncertainty and NLL costs for all fixed controls. The input JSON",
              "retains every circuit/control arm and per-item/window changes.", ""]
    output = args.input.parent
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, content in (("README.md", lines), ("candidate_components.md", candidates),
                          ("control_comparisons.md", controls_table)):
        (output / name).write_text("\n".join(content) + "\n")
    print("\n".join(lines[-20:]))


if __name__ == "__main__":
    main()
