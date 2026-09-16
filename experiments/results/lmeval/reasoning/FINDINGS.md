# Reasoning eval: DAGFormer vs baseline (2026-09-16)

Produced by `run_eval.py --model all --suite reasoning --gen-limit 200 --save-samples`
on one A100; full per-task numbers in `comparison.md` / `comparison.csv`.

## 1. GSM8K exact-match is at the floor for every model — as expected

| size | baseline (flexible-extract) | dagformer |
|---|---|---|
| 75M | 0.010 | 0.025 |
| 150M | 0.010 | 0.005 |
| 300M | 0.025 | 0.015 |
| 600M baseline | 0.020 | — |

n=200, so one correct answer is 0.005. These are all indistinguishable from
zero and from each other, in both directions — chain-of-thought arithmetic does
not exist at 75M-600M / 12B tokens. The generations are fluent and unrelated to
the question (`samples/*.jsonl`). **Do not report these as a comparison.**

## 2. The reasoning signal is in GSM8K chain-of-thought bits-per-byte

`gsm8k_bpb` (custom task, `../tasks/gsm8k_bpb.yaml`) scores the *gold* solution
text under the model, so it has no floor. DAGFormer wins at every size — and by
**~1.7× the margin it wins on wikitext**, which is the interesting part: the
routing advantage is larger on math-reasoning text than on general text.

| size | Δ wikitext BPB | Δ GSM8K-CoT BPB | ratio |
|---|---|---|---|
| 75M | 0.0459 (3.3%) | 0.0805 (4.4%) | 1.76× |
| 150M | 0.0438 (3.7%) | 0.0741 (4.7%) | 1.69× |
| 300M | 0.0424 (4.0%) | 0.0705 (5.0%) | 1.66× |

(Δ = baseline − dagformer, lower BPB is better.) The wikitext column is a
control run through this same harness (`../wikitext_control/`), not the number
from the shared README — though it reproduces it to four decimals for six of
seven checkpoints. The exception is `300m-baseline`: 1.0649 here vs 1.0715 in
`/work/hdd/bfqt/shared/dagformer-models/README.md`. Worth chasing before the
wikitext number is quoted anywhere, since the checkpoints and task are the same.

The 300M pair is also **not step-matched** — DAGFormer stopped at 9000 steps vs
the baseline's 12000 (25% fewer tokens) — so the 300M gap is a lower bound.

## 3. Multiple-choice reasoning: consistent direction, no single significant task

DAGFormer is ahead on 18 of 25 non-tied task-size cells (2 tied), sign test
p = 0.043. No individual delta clears twice its combined standard error; the
per-task effects (~+0.01 to +0.025 accuracy) are smaller than what ~1-3k
evaluation documents can resolve. The cells are not independent (same model
pair across tasks), so read p = 0.043 as "the direction is consistent", not as
a per-task result.

Largest consistent movers: winogrande (+0.007/+0.025/+0.009 across sizes),
arc_easy (+0.000/+0.011/+0.015), mathqa (+0.007/+0.021/+0.010).
commonsense_qa is flat everywhere and sits at chance (~0.20 on 5 choices) for
every model including the 600M baseline — it carries no information at this
scale and could be dropped from the suite.

## Suggested next steps

- Train a 600M DAGFormer: the 600M baseline is the only unpaired point, and the
  BPB trend would be much stronger with a fourth size.
- Re-run 300M DAGFormer at 12000 steps to make that pair iso-token.
- If gsm8k EM is needed for a paper, it needs a model an order of magnitude
  larger; at this scale report `gsm8k_bpb` and say why.
