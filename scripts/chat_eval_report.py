"""Judged chat eval, step 3: win rates and scores from the judge's verdicts.

Pairwise: each question is judged twice (routed as A, then dense as A). The routed model scores
1 / 0.5 / 0 per order for win / tie / loss; its per-question score is the mean over the orders,
and the win rate is the mean over questions (ties count half). The W/T/L counts use the MT-Bench
rule: a win or loss only if both orders agree, otherwise a tie. 95% CIs are a bootstrap over
questions (paired: both models' answers to the same question stay together).
Single: mean 1-10 score per model; routed - dense difference with a paired bootstrap CI.

    python scripts/chat_eval_report.py [--root experiments/chat_eval]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import yaml

CATS = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]


def load_jsonl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def boot_ci(vals: list[float], n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    k = len(vals)
    means = sorted(sum(vals[rng.randrange(k)] for _ in range(k)) / k for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="experiments/chat_eval")
    args = ap.parse_args()
    spec = yaml.safe_load(open(os.path.join(args.root, "models.yaml")))
    pair = load_jsonl(os.path.join(args.root, "judgments/pairwise.jsonl"))
    single = load_jsonl(os.path.join(args.root, "judgments/single.jsonl"))
    verdict = {(r["model_a"], r["model_b"], r["question_id"]): r["verdict"] for r in pair}
    cat_of = {r["question_id"]: r["category"] for r in pair + single}
    out, lines = {"comparisons": [], "models": {}}, []

    lines += ["# Judged chat eval (MT-Bench first turns, LLM judge)", "",
              "Judge: Qwen3-Coder-30B-A3B-Instruct, greedy, MT-Bench judge prompts (math, reasoning and coding "
              "with the GPT-4 reference answer). Answers: greedy, up to 256 new tokens, ChatML as in SFT. "
              "Pairwise verdicts in both answer orders; win rate = routed wins + half the ties, per question "
              "averaged over the two orders; W/T/L counts a win or loss only when both orders agree. "
              "95% CI: bootstrap over the 80 questions. Single runs of every model.", "",
              "## Pairwise: routed vs dense", "",
              "| SFT data | routed | dense | setting | win rate [95% CI] | W / T / L | order-consistent | judged A |",
              "|---|---|---|---|---|---|---|---|"]
    for c in spec["comparisons"]:
        r, d = c["routed"], c["dense"]
        per_q, wtl, consistent, a_votes, n_votes = {}, [0, 0, 0], 0, 0, 0
        for qid in sorted({k[2] for k in verdict if k[:2] == (r, d)}):
            v1, v2 = verdict.get((r, d, qid)), verdict.get((d, r, qid))   # v1: routed is A; v2: routed is B
            s = []
            if v1 and v1 != "error":
                s.append({"A": 1.0, "C": 0.5, "B": 0.0}[v1]); a_votes += v1 == "A"; n_votes += 1
            if v2 and v2 != "error":
                s.append({"B": 1.0, "C": 0.5, "A": 0.0}[v2]); a_votes += v2 == "A"; n_votes += 1
            if not s:
                continue
            per_q[qid] = sum(s) / len(s)
            if len(s) == 2:
                consistent += s[0] == s[1]
                wtl[0 if s == [1.0, 1.0] else 2 if s == [0.0, 0.0] else 1] += 1
        if not per_q:
            continue
        vals = [per_q[q] for q in sorted(per_q)]
        wr, (lo, hi) = sum(vals) / len(vals), boot_ci(vals)
        by_cat = {cat: (sum(per_q[q] for q in per_q if cat_of[q] == cat) /
                        max(1, sum(1 for q in per_q if cat_of[q] == cat))) for cat in CATS}
        sft = "Alpaca-Dolly" if r.endswith("alpaca_dolly") else "SmolTalk"
        lines.append(f"| {sft} | {r.rsplit('_', 2)[0]} | {d.rsplit('_', 2)[0]} | {c.get('note', '')} | "
                     f"{wr:.3f} [{lo:.3f}, {hi:.3f}] | {wtl[0]} / {wtl[1]} / {wtl[2]} | "
                     f"{consistent}/{len(per_q)} | {a_votes / max(1, n_votes):.2f} |")
        out["comparisons"].append({"routed": r, "dense": d, "note": c.get("note"), "n": len(per_q),
                                   "win_rate": wr, "ci95": [lo, hi], "wtl": wtl,
                                   "order_consistent": consistent, "frac_A": a_votes / max(1, n_votes),
                                   "by_category": by_cat})
    lines += ["", "Per category (routed win rate, 10 questions each, so +-0.15 is noise):", "",
              "| routed vs dense | " + " | ".join(CATS) + " |", "|---|" + "---|" * len(CATS)]
    for c in out["comparisons"]:
        lines.append(f"| {c['routed']} vs {c['dense']} | " +
                     " | ".join(f"{c['by_category'][k]:.2f}" for k in CATS) + " |")

    scores = defaultdict(dict)
    for r in single:
        if r["score"] is not None:
            scores[r["model"]][r["question_id"]] = r["score"]
    ans_stats = {}
    for m in spec["models"]:
        rows = load_jsonl(os.path.join(args.root, "answers", f"{m['name']}.jsonl"))
        if rows:
            ans_stats[m["name"]] = (sum(x["n_new"] for x in rows) / len(rows),
                                    sum(x["stopped"] for x in rows) / len(rows))
    lines += ["", "## Single-answer scores (1-10)", "",
              "| model | mean score | n | mean answer tokens | ended on <\\|im_end\\|> |", "|---|---|---|---|---|"]
    for m in spec["models"]:
        n = m["name"]
        if n in scores:
            s = list(scores[n].values())
            out["models"][n] = {"mean_score": sum(s) / len(s), "n": len(s),
                                "answer_tokens": ans_stats.get(n, (None, None))[0],
                                "stopped_frac": ans_stats.get(n, (None, None))[1]}
            at, st = ans_stats.get(n, (float("nan"), float("nan")))
            lines.append(f"| {n} | {sum(s) / len(s):.2f} | {len(s)} | {at:.0f} | {st:.2f} |")
    lines += ["", "Paired score difference, routed - dense:", "", "| routed vs dense | diff [95% CI] |", "|---|---|"]
    for c in spec["comparisons"]:
        r, d = c["routed"], c["dense"]
        common = sorted(set(scores.get(r, {})) & set(scores.get(d, {})))
        if common:
            diff = [scores[r][q] - scores[d][q] for q in common]
            lo, hi = boot_ci(diff)
            lines.append(f"| {r} vs {d} | {sum(diff) / len(diff):+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    with open(os.path.join(args.root, "RESULTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(args.root, "results.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
