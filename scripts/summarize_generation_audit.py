"""Report saved GSM8K extraction scores and mechanical generation diagnostics."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

import numpy as np


def normalize_gold(value):
    for pattern in (",", r"\$", r"(?s).*#### ", r"\.$"):
        value = re.sub(pattern, "", value)
    return value.strip().lower()


def response_text(value):
    while isinstance(value, list):
        value = value[0] if value else ""
    return value if isinstance(value, str) else ""


def repeated_fourgram_fraction(text):
    words = text.split()
    grams = [tuple(words[i:i + 4]) for i in range(len(words) - 3)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, type=Path)
    args = ap.parse_args()
    result = {"protocol": {
        "scores": "unchanged upstream GSM8K strict-match and flexible-extract scores",
        "flexible_extract": "last numeric regex match, including numbers in repeated or irrelevant text",
        "constant_answer": "most frequent gold answer in this evaluated split; diagnostic, not a trained model",
        "repetition": "one minus distinct whitespace-token 4-grams divided by total 4-grams; descriptive only",
        "examples": "first three flexible matches, flexible nonmatches and strict matches in document-ID order, where available"},
        "models": {}}
    lines = ["# GSM8K generation audit", "",
             "Scores below retain the upstream extraction rules. Flexible extraction",
             "uses the last matched number; it does not validate the derivation or answer units.", "",
             "| Model | Documents | Strict match | Flexible match | Best constant answer | Mean repeated 4-gram fraction |",
             "|---|---:|---:|---:|---|---:|"]
    examples = ["# Deterministic GSM8K example sample", "",
                "For each completed model: the first three flexible matches, flexible nonmatches",
                "and strict matches by document ID, where available. Excerpts are limited to 400 characters; full",
                "responses remain in the ignored sample JSONLs and the Delta artifact mirror.", ""]
    for path in sorted(args.dir.glob("*__gen*.json")):
        payload = json.loads(path.read_text())
        if "gsm8k" not in payload.get("results", {}):
            continue
        name = payload["model"]["name"]
        sample_path = args.dir / "samples" / f"{name}__gsm8k.jsonl"
        samples = [json.loads(line) for line in sample_path.read_text().splitlines()]
        grouped = {f: sorted([r for r in samples if r["filter"] == f], key=lambda r: r["doc_id"])
                   for f in ("strict-match", "flexible-extract")}
        scores = {f: float(np.mean([r["exact_match"] for r in rows])) for f, rows in grouped.items()}
        for f, score in scores.items():
            if not math.isclose(score, payload["results"]["gsm8k"]["exact_match," + f], abs_tol=1e-12):
                raise ValueError(f"Saved samples disagree with reported {name}/{f}")
        rows = grouped["flexible-extract"]
        gold = Counter(normalize_gold(r["target"]) for r in rows)
        mode, count = gold.most_common(1)[0]
        repeats = [repeated_fourgram_fraction(response_text(r["resps"])) for r in rows]
        rec = {"model": payload["model"], "eval": payload["eval"], "n": len(rows),
               "scores": scores, "correct_counts": {f: sum(r["exact_match"] for r in v) for f, v in grouped.items()},
               "most_common_gold_answer": mode, "constant_answer_accuracy": count / len(rows),
               "gold_answer_counts": dict(gold), "mean_repeated_fourgram_fraction": float(np.mean(repeats)),
               "flexible_matches_without_strict_match": sum(r["exact_match"] > s["exact_match"]
                                                             for r, s in zip(rows, grouped["strict-match"]))}
        result["models"][name] = rec
        lines.append(f"| {name} | {len(rows)} | {100*scores['strict-match']:.2f}% | "
                     f"{100*scores['flexible-extract']:.2f}% | {mode}: {100*count/len(rows):.2f}% | "
                     f"{np.mean(repeats):.3f} |")
        examples += [f"## {name}", ""]
        selections = [("flexible match", [r for r in rows if r["exact_match"]][:3]),
                      ("flexible nonmatch", [r for r in rows if not r["exact_match"]][:3]),
                      ("strict match", [r for r in grouped["strict-match"] if r["exact_match"]][:3])]
        for label, chosen in selections:
            for row in chosen:
                examples += [f"### Document {row['doc_id']} — {label}", "",
                             f"Question: {row['doc']['question']}", "",
                             f"Gold final answer: `{normalize_gold(row['target'])}`. "
                             f"Extracted: `{row['filtered_resps'][0]}`.", "",
                             "```text", response_text(row["resps"])[:400].rstrip(), "```", ""]
    lines += ["", "The constant-answer diagnostic uses gold frequencies from the same evaluated documents.",
              "Repetition is a mechanical text statistic, not a correctness judgment.",
              "[Deterministically selected examples](generation_examples.md) show the continuations behind these metrics.", ""]
    (args.dir / "generation_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.dir / "generation_audit.md").write_text("\n".join(lines))
    (args.dir / "generation_examples.md").write_text("\n".join(examples))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
