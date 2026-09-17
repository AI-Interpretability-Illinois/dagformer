"""Compare cached and uncached generation on the same saved document subset."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from summarize_generation_audit import repeated_fourgram_fraction, response_text
from summarize_paired_eval import paired_difference, read_samples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--variant", required=True, type=Path)
    args = ap.parse_args()
    reference = json.loads(args.reference.read_text())
    variant = json.loads(args.variant.read_text())
    for key in ("checkpoint", "step", "num_parameters"):
        if reference["model"][key] != variant["model"][key]:
            raise ValueError(f"Checkpoint mismatch: {key}")
    for key in ("max_length", "max_gen_toks", "gen_num_fewshot", "fewshot_seed",
                "gen_batch_size", "tokenizer", "lm_eval_version"):
        if reference["eval"][key] != variant["eval"][key]:
            raise ValueError(f"Generation setting differs: {key}")
    if not reference["eval"]["kv_cache"] or variant["eval"]["kv_cache"]:
        raise ValueError("Expected a cached reference and uncached variant")
    result = {"reference": str(args.reference), "variant": str(args.variant),
              "protocol": "same saved prompts and documents; cached reference restricted to uncached document IDs; positive differences favor uncached generation; paired document bootstrap, 10000 draws, seed 20260917",
              "metrics": {}}
    for filter_name in ("strict-match", "flexible-extract"):
        path = lambda p, d: p.parent / "samples" / f"{d['model']['name']}__gsm8k.jsonl"
        cached = read_samples(path(args.reference, reference), filter_name)
        uncached = read_samples(path(args.variant, variant), filter_name)
        cached = {key: cached[key] for key in uncached}
        stats = paired_difference(cached, uncached, "exact_match," + filter_name,
                                  draws=10000, seed=20260917)
        if not math.isclose(stats["dagformer"], variant["results"]["gsm8k"]["exact_match," + filter_name]):
            raise ValueError("Uncached scores disagree with saved samples")
        rename = {"baseline": "cached", "dagformer": "uncached",
                  "delta_dagformer_better": "delta_uncached_better",
                  "dag_only_correct": "uncached_only_correct",
                  "base_only_correct": "cached_only_correct"}
        result["metrics"][filter_name] = {rename.get(k, k): v for k, v in stats.items()}
        if filter_name == "flexible-extract":
            ids = sorted(uncached)
            original_text = [response_text(cached[i]["resps"]) for i in ids]
            new_text = [response_text(uncached[i]["resps"]) for i in ids]
            result["response_comparison"] = {
                "n": len(ids),
                "changed_response_count": sum(a != b for a, b in zip(original_text, new_text)),
                "changed_extracted_count": sum(cached[i]["filtered_resps"] != uncached[i]["filtered_resps"] for i in ids),
                "cached_mean_repeated_fourgram_fraction": float(np.mean([repeated_fourgram_fraction(t) for t in original_text])),
                "uncached_mean_repeated_fourgram_fraction": float(np.mean([repeated_fourgram_fraction(t) for t in new_text])),
                "changed_response_doc_ids": [i for i, a, b in zip(ids, original_text, new_text) if a != b]}
    lines = ["# KV-cache generation check", "",
             "The same 300M baseline checkpoint is evaluated with and without KV caching.",
             "The cached full-test run is restricted to the uncached run's document IDs.",
             "Document and prompt hashes, checkpoint, and generation settings must match.", "",
             "| Endpoint | Cached | Uncached | Difference (percentage points) | Paired 95% interval |",
             "|---|---:|---:|---:|---|"]
    for name, stats in result["metrics"].items():
        lo, hi = stats["paired_bootstrap_95ci"]
        lines.append(f"| {name} | {100*stats['cached']:.2f}% | {100*stats['uncached']:.2f}% | "
                     f"{100*stats['delta_uncached_better']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    response = result["response_comparison"]
    lines += ["", f"Full response text changes on {response['changed_response_count']}/{response['n']} documents; "
              f"the extracted answer changes on {response['changed_extracted_count']}.",
              f"Mean repeated whitespace-token 4-gram fraction is {response['cached_mean_repeated_fourgram_fraction']:.3f} "
              f"with caching and {response['uncached_mean_repeated_fourgram_fraction']:.3f} without.", "",
              "Intervals describe this document subset, without training-seed uncertainty.", ""]
    out = args.variant.parent
    (out / "cache_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "README.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
