"""Audit rendered-prompt repetitions and cluster uncertainty in context generation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from eval_context_fidelity import evaluation_prompt
from interp_liar_cloze import build_items
from summarize_context_generation import mentioned


def vectors(items, values):
    first = np.asarray(values["first_value_true"], dtype=float)
    any_value = np.asarray(values["any_value"], dtype=float)
    target = np.asarray([mentioned(s, it["true"]) for it, s in zip(items, values["generations"], strict=True)])
    alternative = np.asarray([any(mentioned(s, v) for v in it["alts"])
                              for it, s in zip(items, values["generations"], strict=True)])
    return {"target_first": first, "alternative_first": any_value - first,
            "no_candidate": 1. - any_value,
            "target_without_alternative": (target & ~alternative).astype(float),
            "alternative_anywhere": alternative.astype(float)}


def cluster_stats(variant, reference, group_ids, counts, draws, denominators):
    diff = variant - reference
    sums = np.bincount(group_ids, weights=diff, minlength=len(counts))
    sampled = sums[draws].sum(1) / denominators
    return {"reference": float(reference.mean()), "variant": float(variant.mean()),
            "delta": float(diff.mean()), "prompt_cluster_bootstrap_95ci": np.quantile(sampled, [.025, .975]).tolist(),
            "equal_unique_prompt_delta": float((sums / counts).mean())}


def cell(rec):
    lo, hi = rec["prompt_cluster_bootstrap_95ci"]
    return f"{100 * rec['delta']:+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()
    discovery = build_items(240, seed=0)
    item_key = lambda it: (it['fact'], it['ask'], it['a'], it['b'])
    old_keys = {item_key(it) for it in discovery}
    old_facts = {it['fact'] for it in discovery}
    output = {"protocol": {
        "status": "posthoc robustness and endpoint diagnostics; primary target-first results are retained",
        "clusters": "identical rendered prompt strings, separately within each condition",
        "intervals": "paired prompt-cluster bootstrap, preserving the original item-weighted point estimate by resampling cluster sums and counts",
        "equal_unique_prompt_delta": "additional point estimate giving every distinct rendered prompt equal weight",
        "heldout_scope": "original item keys include the receiver; distinct item keys do not guarantee distinct facts or neutral QA prompts",
        "secondary_endpoint": "target_without_alternative is literal target mention with no listed alternative anywhere; it does not resolve negation, quotations, or separate statements about another speaker",
        "selection": "all named edits and both measured controls; no selection by effect direction"},
        "draws": args.draws, "seed": args.seed, "sources": {}}
    lines = ["# Rendered-prompt clusters and generation endpoints", "",
             "The original item key includes the receiver. Neutral QA prompts omit that",
             "field, so different sampled items can render as identical prompts. This",
             "posthoc check resamples identical-prompt groups together, retaining the",
             "original item-weighted point estimates. JSON also gives equally weighted",
             "unique-prompt point estimates. Intervals are unadjusted and do not represent",
             "variation across training seeds or all possible control circuits.", "",
             "The cohorts exclude original discovery item keys, not every repeated fact.",
             "New attribute questions also use a new template; differences between cohorts",
             "are not a paired estimate of changing only the question wording.", "",
             "| Source | Condition | Item rows | Unique prompts | Largest repeated group |",
             "|---|---|---:|---:|---:|"]
    effects = ["# Named edits under prompt-cluster uncertainty", "",
               "All differences are percentage points from the unedited model. The primary",
               "endpoint remains target-first. Target-without-alternative and alternative-anywhere",
               "are secondary literal-mention diagnostics; they do not grade semantic truth.", "",
               "| Source | Condition | Arm | Target first | Alternative first | No candidate | Target without alternative | Alternative anywhere |",
               "|---|---|---|---|---|---|---|---|"]
    controls = ["# Named-minus-control comparisons with prompt clusters", "",
                "Intervals resample rendered prompts for each fixed control, not the population",
                "of possible control directions. Differences are percentage points.", "",
                "| Source | Condition | Channel | Control | Target first | Target without alternative |",
                "|---|---|---|---|---|---|"]
    for path in args.inputs:
        raw = json.loads(path.read_text())
        expected = 1 + 3 * (1 + raw["args"]["controls"])
        if len(raw["arms"]) != expected:
            raise ValueError(f"Incomplete input: {path}")
        items = raw["items"]
        keys = [item_key(it) for it in items]
        source = {"path": str(path), "n_items": len(items), "unique_item_keys": len(set(keys)),
                  "original_discovery_item_key_overlap": len(set(keys) & old_keys),
                  "unique_facts": len({it['fact'] for it in items}),
                  "original_discovery_fact_overlap": len({it['fact'] for it in items} & old_facts),
                  "item_rows_with_discovery_fact": sum(it['fact'] in old_facts for it in items),
                  "conditions": {}}
        for condition in raw["arms"]["reference"]["values"]:
            style, cue = condition.split('/')
            prompts = [evaluation_prompt(it, cue, style) for it in items]
            _, group_ids = np.unique(prompts, return_inverse=True)
            counts = np.bincount(group_ids)
            draws = np.random.default_rng(args.seed).integers(len(counts), size=(args.draws, len(counts)))
            denominators = counts[draws].sum(1)
            all_vectors = {name: vectors(items, arm['values'][condition]) for name, arm in raw['arms'].items()}
            ref = all_vectors['reference']
            rec = {"unique_prompts": len(counts), "max_multiplicity": int(counts.max()),
                   "group_size_histogram": dict(Counter(map(int, counts))),
                   "reference_means": {k: float(v.mean()) for k, v in ref.items()},
                   "arms": {}, "paired_circuit_minus_control": {}}
            for name, values in all_vectors.items():
                if name == 'reference':
                    continue
                rec['arms'][name] = {metric: cluster_stats(v, ref[metric], group_ids, counts, draws, denominators)
                                     for metric, v in values.items()}
                if name.endswith('/circuit'):
                    effects.append(f"| {path.stem} | {condition} | {name} | "
                                   + " | ".join(cell(v) for v in rec['arms'][name].values()) + " |")
            for channel in ('pred', 'corr', 'both'):
                for index in range(raw['args']['controls']):
                    name = f'{channel}/random{index}'
                    rec['paired_circuit_minus_control'][name] = {
                        metric: cluster_stats(all_vectors[f'{channel}/circuit'][metric], all_vectors[name][metric],
                                              group_ids, counts, draws, denominators)
                        for metric in ('target_first', 'target_without_alternative')}
                    values = rec['paired_circuit_minus_control'][name]
                    controls.append(f"| {path.stem} | {condition} | {channel} | {index} | "
                                    + " | ".join(cell(v) for v in values.values()) + " |")
            source['conditions'][condition] = rec
            lines.append(f"| {path.stem} | {condition} | {len(items)} | {len(counts)} | {counts.max()} |")
        output['sources'][path.stem] = source
    lines += ["", "[All named endpoints](named_endpoints.md) include losses, gains and uncertain differences.",
              "[Direct control comparisons](control_comparisons.md) retain the two fixed controls.",
              "The companion JSON records overlap with original discovery facts, all arms,",
              "and the change from giving each unique prompt equal weight.", ""]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in (("README.md", lines), ("named_endpoints.md", effects), ("control_comparisons.md", controls)):
        (args.out_dir/name).write_text('\n'.join(text).rstrip()+'\n')
    (args.out_dir/'summary.json').write_text(json.dumps(output, indent=2)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
