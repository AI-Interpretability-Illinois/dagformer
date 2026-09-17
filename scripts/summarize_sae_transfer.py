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
            "n_tokens": int(counts.sum()), "n_windows": int((counts > 0).sum()),
            "valid_bootstrap_draws": int(keep.sum()),
            "paired_bootstrap_95ci": np.quantile(values, [.025, .975]).tolist()}


def paired_span_contrast(minus, plus, control_minus, control_plus, draws, seed):
    """Compare two signed dose spans while retaining all four arms' pairing."""
    contrasts = []
    counts = np.asarray(plus["window_token_count"])
    for arm, control in ((minus, control_minus), (plus, control_plus)):
        if not all(np.array_equal(counts, value["window_token_count"])
                   for value in (arm, control)):
            raise ValueError("Feature and control arms must score the same target tokens")
        contrasts.append({"window_token_count": counts,
                          "window_delta_sum": np.asarray(arm["window_delta_sum"])
                          - np.asarray(control["window_delta_sum"])})
    return span_summary(*contrasts, draws, seed)


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
             "The count column gives target tokens / windows containing target tokens.",
             "Bootstrap draws with no target tokens are omitted; valid-draw counts are in JSON.", "",
             "Token labels are raw tokenizer pieces, including whitespace and fragments.",
             "They are not a semantic annotation of what each feature represents.", "",
             "Negative target-token ΔNLL means that those gold tokens become more probable.",
             "The signed dose span is ΔNLL(+4) − ΔNLL(−4), calculated from paired windows.", "",
             "| Feature | Target tokens | Tokens / windows | ΔNLL at −4 [95% CI] | ΔNLL at +4 [95% CI] | Dose span [95% CI] |",
             "|---|---|---:|---|---|---|"]
    secondary = ["", "## Original screen, other tokens and permuted controls", "",
                 payload["protocol"]["controls"] + ".",
                 "They preserve coefficient values and norms. The five-control span range",
                 "is descriptive and is not a confidence interval. Cross-corpus differences",
                 "also change which target tokens occur and their contexts.", "",
                 "| Feature | Original target ΔNLL −4 / +4 | New other-token ΔNLL −4 / +4 | Five permuted target-span range |",
                 "|---|---|---|---|"]
    control_costs = ["", "## Language-model cost of the controls", "",
                     "The permutations preserve direction norms, not language-model capability.",
                     "Their other-token NLL costs must be considered alongside target effects.",
                     "A feature/control contrast at unequal damage does not isolate semantic",
                     "specificity at a fixed capability cost. Ranges below contain the five",
                     "measured controls, not confidence intervals.", "",
                     "| Feature | Control other-token ΔNLL range at −4 | Range at +4 |",
                     "|---|---|---|"]
    summary = {"source": str(args.input), "old_screen": str(args.old_screen), "features": {}}
    decomposition = ["", "## Shared residual and Q/K/V components", "",
                     "The same feature direction is restricted to shared R or Q/K/V coordinates.",
                     "The two component effects need not add because the network is nonlinear.", "",
                     "| Feature | Component | Target ΔNLL −4 | Target ΔNLL +4 | Other-token ΔNLL −4 / +4 |",
                     "|---|---|---|---|---|"]
    token_lines = ["# Individual tokens within the fixed SAE target sets", "",
                   "These diagnostics retain all tokens in each pre-existing rule. Labels are",
                   "raw tokenizer pieces. Counts and unadjusted paired window-bootstrap",
                   "intervals show when an aggregate effect is driven by a common piece,",
                   "such as whitespace, rather than every token in the set.", "",
                   "| Feature | Component | Token | Occurrences | ΔNLL at −4 | ΔNLL at +4 |",
                   "|---|---|---|---:|---|---|"]
    has_token_details = False
    control_comparisons = ["# Paired feature-minus-control dose spans", "",
                           "Each contrast is (feature +4 minus feature -4) minus",
                           "(control +4 minus control -4), paired across the same windows.",
                           "A positive value means a larger signed dose response, not higher",
                           "language-model accuracy. Intervals are unadjusted window bootstrap",
                           "intervals; control NLL costs are reported in the main table.", "",
                           "| Feature | Control | Signed span difference [95% CI] |",
                           "|---|---|---|"]
    for feature_id, rec in payload["features"].items():
        arms = rec["arms"]
        minus, plus = arms["feature/alpha-4"], arms["feature/alpha4"]
        span = span_summary(minus["on_rule"], plus["on_rule"], 10000, payload["args"]["seed"])
        tokens = tokenizer.convert_ids_to_tokens(rec["rule"][1])
        controls = []
        span_contrasts = []
        for index in range(payload["args"]["controls"]):
            value = span_summary(arms[f"random{index}/alpha-4"]["on_rule"],
                                 arms[f"random{index}/alpha4"]["on_rule"], 10000, payload["args"]["seed"])
            controls.append(value)
            contrast = paired_span_contrast(
                minus["on_rule"], plus["on_rule"],
                arms[f"random{index}/alpha-4"]["on_rule"],
                arms[f"random{index}/alpha4"]["on_rule"],
                10000, payload["args"]["seed"])
            span_contrasts.append(contrast)
            control_comparisons.append(f"| {feature_id} | random{index} | {effect(contrast)} |")
        valid = [c["delta"] for c in controls if c["delta"] is not None]
        bounds = [min(valid), max(valid)] if valid else None
        off_ranges = {}
        for alpha in (-4, 4):
            values = [arms[f"random{i}/alpha{alpha}"]["off_rule"]["delta"]
                      for i in range(payload["args"]["controls"])]
            off_ranges[str(alpha)] = [min(values), max(values)]
        summary["features"][feature_id] = {"tokens": tokens, "on_minus": minus["on_rule"],
            "on_plus": plus["on_rule"], "off_minus": minus["off_rule"], "off_plus": plus["off_rule"],
            "span": span, "control_spans": controls, "control_span_range": bounds,
            "paired_feature_minus_control_spans": span_contrasts,
            "control_off_rule_ranges": off_ranges,
            "original_screen": old[feature_id]}
        token_text = ", ".join(tokens).replace("|", "\\|")
        lines.append(f"| {feature_id} | {token_text} | {plus['on_rule']['n_tokens']} / {plus['on_rule']['n_windows']} | "
                     f"{effect(minus['on_rule'])} | {effect(plus['on_rule'])} | {effect(span)} |")
        bound_text = f"[{bounds[0]:+.4f}, {bounds[1]:+.4f}]" if bounds else "no target tokens"
        historical = old[feature_id]
        secondary.append(f"| {feature_id} | {historical['on-4']:+.4f} / {historical['on+4']:+.4f} | "
                         f"{minus['off_rule']['delta']:+.4f} / {plus['off_rule']['delta']:+.4f} | {bound_text} |")
        control_costs.append(f"| {feature_id} | [{off_ranges['-4'][0]:+.4f}, {off_ranges['-4'][1]:+.4f}] | "
                             f"[{off_ranges['4'][0]:+.4f}, {off_ranges['4'][1]:+.4f}] |")
        if payload["args"].get("stream_decomposition"):
            summary["features"][feature_id]["components"] = {}
            for component in ("r_only", "qkv_only"):
                low, high = arms[f"{component}/alpha-4"], arms[f"{component}/alpha4"]
                summary["features"][feature_id]["components"][component] = {"minus": low, "plus": high}
                decomposition.append(f"| {feature_id} | {component} | {effect(low['on_rule'])} | "
                                     f"{effect(high['on_rule'])} | {low['off_rule']['delta']:+.4f} / {high['off_rule']['delta']:+.4f} |")
        for component in ("feature", "r_only", "qkv_only"):
            if f"{component}/alpha-4" not in arms or "by_token" not in arms[f"{component}/alpha-4"]:
                continue
            has_token_details = True
            for token_id, low in arms[f"{component}/alpha-4"]["by_token"].items():
                high = arms[f"{component}/alpha4"]["by_token"][token_id]
                label = tokenizer.convert_ids_to_tokens(int(token_id)).replace("|", "\\|")
                token_lines.append(f"| {feature_id} | {component} | {label} | {low['n_tokens']} | "
                                   f"{effect(low)} | {effect(high)} |")
    output = args.input.parent
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sections = lines + secondary + control_costs + (decomposition if payload["args"].get("stream_decomposition") else [])
    sections += ["", "[Direct paired span contrasts](control_comparisons.md) compare the feature",
                 "with each control while retaining the pairing of all four dose arms."]
    if payload["args"].get("stream_decomposition"):
        sections[2:2] = ["This mechanism follow-up reuses the same 128 WikiText test windows as",
                         "the initial transfer run. Whole-head controls and the R/Q/K/V split",
                         "were added after inspecting that run; this is not a second corpus replication.", ""]
    if has_token_details:
        sections += ["", "[Individual-token diagnostics](individual_tokens.md) retain occurrence counts",
                     "and paired intervals within each fixed target set."]
    (output / "README.md").write_text("\n".join(sections) + "\n")
    (output / "control_comparisons.md").write_text("\n".join(control_comparisons) + "\n")
    if has_token_details:
        (output / "individual_tokens.md").write_text("\n".join(token_lines) + "\n")
    print("\n".join(sections))


if __name__ == "__main__":
    main()
