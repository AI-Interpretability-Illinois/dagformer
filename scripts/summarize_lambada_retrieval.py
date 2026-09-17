"""Posthoc LAMBADA stratification by exact answer-token occurrence in context."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from lm_eval.api.model import TemplateLM
from transformers import AutoTokenizer

from summarize_paired_eval import paired_difference, read_samples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--variants", required=True, nargs="+", type=Path)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    reference = read_samples(args.reference, "none")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    encoder = SimpleNamespace(backend="causal", tok_encode=lambda t: tokenizer.encode(t, add_special_tokens=False))
    strata = {"answer_seen": {}, "answer_not_seen": {}}
    truncated = 0
    for doc_id, sample in reference.items():
        context, continuation = sample["arguments"][0]
        prefix, answer = TemplateLM._encode_pair(encoder, context, continuation)
        keep = args.max_length + 1 - len(answer)
        if keep < 1 or not answer:
            raise ValueError("Invalid answer/context lengths")
        truncated += len(prefix) > keep
        prefix = prefix[-keep:]
        seen = any(prefix[i:i + len(answer)] == answer for i in range(len(prefix) - len(answer) + 1))
        strata["answer_seen" if seen else "answer_not_seen"][doc_id] = sample
    result = {"reference": str(args.reference),
              "protocol": "posthoc analysis motivated by observed head-edit LAMBADA gains; exact answer-token sequence in retained context, using lm_eval's causal context/continuation encoding; this is an occurrence proxy, not a semantic retrieval annotation",
              "tokenizer": args.tokenizer, "max_length": args.max_length,
              "n_contexts_truncated": truncated,
              "stratum_doc_ids": {k: sorted(v) for k, v in strata.items()}, "variants": {}}
    lines = ["# LAMBADA: does the answer already occur in context?", "",
             "This posthoc diagnostic was prompted by the observed LAMBADA gains after fixed",
             "head editing. The grouping is determined from the saved prompt and gold answer,",
             "before using each document's change in accuracy. ‘Seen’ means the exact answer",
             "token sequence occurs in the retained input context. It does not include semantic",
             "equivalents or changes in capitalization/tokenization. Encoding and left truncation",
             "match the evaluation harness. Intervals are unadjusted paired document bootstraps.", "",
             "| Variant | Answer occurrence | Documents | Unedited accuracy | Edited accuracy | Difference (points) | Paired 95% interval |",
             "|---|---|---:|---:|---:|---:|---|"]
    for path in args.variants:
        variant = read_samples(path, "none")
        if set(variant) != set(reference):
            raise ValueError("LAMBADA document sets differ")
        name = path.name.removesuffix("__lambada_openai.jsonl")
        result["variants"][name] = {}
        for group, refs in strata.items():
            vals = {i: variant[i] for i in refs}
            stats = paired_difference(refs, vals, "acc,none", draws=10000, seed=20260917)
            rename = {"baseline": "reference", "dagformer": "variant",
                      "delta_dagformer_better": "delta_variant_better",
                      "dag_only_correct": "variant_only_correct", "base_only_correct": "reference_only_correct"}
            stats = {rename.get(k, k): v for k, v in stats.items()}
            stats["first_gain_doc_ids"] = [i for i in sorted(refs) if vals[i]["acc"] > refs[i]["acc"]][:3]
            stats["first_loss_doc_ids"] = [i for i in sorted(refs) if vals[i]["acc"] < refs[i]["acc"]][:3]
            result["variants"][name][group] = stats
            lo, hi = stats["paired_bootstrap_95ci"]
            lines.append(f"| {name} | {group} | {len(refs)} | {100*stats['reference']:.2f}% | "
                         f"{100*stats['variant']:.2f}% | {100*stats['delta_variant_better']:+.2f} | "
                         f"[{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ["", "The JSON retains group membership and the first gain/loss document IDs.",
              "An association with repeated answers does not identify which computation",
              "produced a correct prediction or establish a causal mediation mechanism.", ""]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.out.with_suffix(".md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
