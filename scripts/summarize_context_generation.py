"""Separate target-value inclusion, alternative mentions and attribute omission."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

from eval_context_fidelity import evaluation_prompt, paired_summary


def mentioned(text, word):
    return bool(word and re.search(r"\b" + re.escape(word) + r"\b", text, re.IGNORECASE))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    raw = json.loads(args.input.read_text())
    expected = 1 + 3 * (1 + raw["args"]["controls"])
    if len(raw["arms"]) != expected:
        raise ValueError(f"Generation run is incomplete: {len(raw['arms'])}/{expected} arms")
    items = raw["items"]
    result = {"source": str(args.input), "protocol": {
        "endpoint": "literal target-value inclusion in the first 16 generated tokens",
        "omission": "no candidate value mentioned; a response may still name the correct object",
        "alternatives": "literal mentions, without resolving negation or quoted speech",
        "examples": "first three reference object-only omissions and alternative-first outputs, plus first two added/removed target-value mentions by item index, for each condition"},
        "arms": {}}
    result["protocol"]["endpoint"] = f"literal target-value inclusion in at most {raw['args']['max_new_tokens']} generated tokens"
    lines = ["# Context generation: value inclusion and omission", "",
             f"Greedy continuations contain at most {raw['args']['max_new_tokens']} new tokens. The endpoint asks whether",
             "the first mentioned candidate value is the value stated in the fact. It",
             "counts omission as a failure to include the target value. This is not a",
             "semantic judgment that the answer is false: for an iron ring, saying",
             "only ‘a ring’ omits the material while remaining compatible with the fact.", "",
             "Candidate alternatives are scored by literal mention. The diagnostic also",
             "checks whether an alternative appears anywhere after a correct first value;",
             "it does not resolve negation, quotation or conversational intent.", "",
             "| Arm | Condition | Target first | Alternative first | No candidate | Correct object named but value omitted | Any alternative anywhere |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for arm_name, arm in raw["arms"].items():
        result["arms"][arm_name] = {}
        for condition, values in arm["values"].items():
            by_category = defaultdict(list)
            rows = []
            for index, (item, text) in enumerate(zip(items, values["generations"], strict=True)):
                target = int(mentioned(text, item["true"]))
                alternative = int(any(mentioned(text, word) for word in item["alts"]))
                no_value = int(not values["any_value"][index])
                row = {"target_first": int(values["first_value_true"][index]),
                       "alternative_first": int(values["any_value"][index] - values["first_value_true"][index]),
                       "no_candidate": no_value,
                       "object_without_value": int(no_value and mentioned(text, item["obj"])),
                       "target_anywhere": target, "alternative_anywhere": alternative,
                       "target_and_alternative": int(target and alternative)}
                rows.append(row)
                by_category[item["cat"]].append(row)

            def counts(records):
                return {"n": len(records), **{key: sum(row[key] for row in records) for key in records[0]}}

            rec = {"all": counts(rows), "categories": {name: counts(vals) for name, vals in by_category.items()}}
            result["arms"][arm_name][condition] = rec
            c = rec["all"]
            keys = ("target_first", "alternative_first", "no_candidate", "object_without_value", "alternative_anywhere")
            lines.append(f"| {arm_name} | {condition} | " + " | ".join(f"{100*c[k]/c['n']:.2f}%" for k in keys) + " |")
    examples = ["# Deterministic context-generation examples", "",
                "Reference omissions and both-channel edit changes are selected by item index.", ""]
    reference = raw["arms"]["reference"]["values"]
    edited = raw["arms"]["both/circuit"]["values"]
    for condition, values in reference.items():
        examples += [f"## {condition}", ""]
        omitted = [i for i, item in enumerate(items)
                   if not values["any_value"][i] and mentioned(values["generations"][i], item["obj"])][:3]
        alternative_first = [i for i in range(len(items))
                             if values["any_value"][i] and not values["first_value_true"][i]][:3]
        added = [i for i in range(len(items)) if not values["first_value_true"][i]
                 and edited[condition]["first_value_true"][i]][:2]
        removed = [i for i in range(len(items)) if values["first_value_true"][i]
                   and not edited[condition]["first_value_true"][i]][:2]
        for label, indexes in (("Object named, attribute omitted", omitted),
                               ("Reference names an alternative first", alternative_first),
                               ("Target-value mention added by edit", added),
                               ("Target-value mention removed by edit", removed)):
            for index in indexes:
                item = items[index]
                style, cue = condition.split("/")
                examples += [f"### Item {index}: {label}", "", f"Prompt: {evaluation_prompt(item, cue, style)}", "",
                             f"Target value: `{item['true']}`; object: `{item['obj']}`.", "",
                             "Reference:", "```text", values["generations"][index].rstrip(), "```", ""]
                if label != "Object named, attribute omitted":
                    examples += ["Both-channel edit:", "```text", edited[condition]["generations"][index].rstrip(), "```", ""]
    lines += ["", "[Deterministically selected examples](context_generation_examples.md) retain the full short continuations.", "",
              "Counts by category and all edit/control arms are retained in the companion JSON.", ""]
    comparisons = {}
    control_lines = ["# Direct comparison with the fixed random controls", "",
                     "Differences compare the circuit and control on the same content items.",
                     "Intervals are unadjusted paired normal intervals over items. They do not",
                     "estimate variation across the population of possible random circuits.", "",
                     "| Channel | Condition | Control | Inclusion difference (points) | Paired 95% interval |",
                     "|---|---|---|---:|---|"]
    for channel in ("pred", "corr", "both"):
        for control in range(raw["args"]["controls"]):
            name = f"{channel}/random{control}"
            comparisons[name] = {}
            for condition, values in raw["arms"][f"{channel}/circuit"]["values"].items():
                stats = paired_summary(values["first_value_true"], raw["arms"][name]["values"][condition]["first_value_true"])
                comparisons[name][condition] = stats
                lo, hi = stats["normal_95ci"]
                control_lines.append(f"| {channel} | {condition} | {control} | {100*stats['delta']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    result["paired_circuit_minus_control"] = comparisons
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "context_generation_diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.out_dir / "context_generation_diagnostics.md").write_text("\n".join(lines))
    (args.out_dir / "context_generation_examples.md").write_text("\n".join(examples))
    (args.out_dir / "context_generation_control_pairs.md").write_text("\n".join(control_lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
