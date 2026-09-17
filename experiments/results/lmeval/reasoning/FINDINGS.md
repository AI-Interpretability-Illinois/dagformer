# Reasoning eval: DAGFormer vs baseline (2026-09-16)

Produced by `run_eval.py --model all --suite reasoning --gen-limit 200 --save-samples`
on one A100; full per-task numbers in `comparison.md` / `comparison.csv`.
The [September 17 campaign](../../eval_20260917/README.md) adds a step-matched
300M pair, full-test generation, answer-conditional likelihood and paired
document uncertainty. The [PR audit](../../../PROJECT_SYNC_2026-09-17.md)
records corrections to the earlier interpretation; the numbers below remain
the historical 200-item run.

## 1. GSM8K exact-match is at the floor for every model — as expected

| size | baseline (flexible-extract) | dagformer |
|---|---|---|
| 75M | 0.010 | 0.025 |
| 150M | 0.010 | 0.005 |
| 300M | 0.025 | 0.015 |
| 600M baseline | 0.020 | — |

n=200, so one matching extracted answer is 0.005. These scores are low, and
numeric extraction does not validate the generated reasoning. They do not
establish an absence of arithmetic ability at these scales or a minimum model
size needed to learn it. Training-token budgets also differ by checkpoint;
12B is the source corpus size, not the budget of every model. Compare paired
uncertainty and inspect outputs; the new campaign expands this check to the
full 1,319-item split and retains repeated or irrelevant numeric matches.

## 2. Likelihood of GSM8K question-and-solution text

`gsm8k_bpb` (custom task, `../tasks/gsm8k_bpb.yaml`) scores the full string
`Question: {question}\nAnswer: {answer}` with rolling likelihood. It includes
the question and gold solution; it is not answer-only conditional likelihood
or a measure of generated reasoning correctness. DAGFormer has lower BPB at
each size. The absolute BPB reduction is 1.66–1.76 times the Wikitext reduction;
the relative percentage reduction is 1.26–1.33 times as large.

| size | Δ wikitext BPB | Δ GSM8K-CoT BPB | ratio |
|---|---|---|---|
| 75M | 0.0459 (3.3%) | 0.0805 (4.4%) | 1.76× |
| 150M | 0.0438 (3.7%) | 0.0741 (4.7%) | 1.69× |
| 300M | 0.0424 (4.0%) | 0.0705 (5.0%) | 1.66× |

(Δ = baseline − dagformer, lower BPB is better.) The wikitext column is a
control run through this same harness (`../wikitext_control/`). The apparent
300M-baseline discrepancy is a checkpoint-label error in the shared README:
`matched_mmap/dense_300m_mmap_s9000.json` records 1.0714925 at **step 9000**;
the new control records 1.0648552 at **step 12000**. Those are different
checkpoints, not conflicting measurements of the same checkpoint. Small
differences also exist at other sizes, so the claim of six exact four-decimal
matches should not be used.

The 300M pair is also **not step-matched** — DAGFormer stopped at 9000 steps vs
the baseline's 12000. With the recorded training batch sizes this is 25% fewer
tokens. This is a budget mismatch; calling its gap a lower bound would require
an unverified monotonicity assumption about further training.

## 3. Multiple-choice reasoning: descriptive direction of the old results

DAGFormer is ahead on **14 of 19 non-tied multiple-choice task-size cells**
(5 losses, 2 ties; 21 cells total), descriptive sign-test p = **0.064**.
The previously reported 18/25 and p = 0.043 combine all task types, including
three GSM8K BPB comparisons and three GSM8K generation comparisons.
The original comparison used combined marginal standard errors. That does
not account for within-document pairing, so it does not establish that
these sample sizes cannot resolve the observed effects. The new campaign
retains per-document outputs and reports paired intervals. The cells are
also not independent (the same model pair across tasks), so read p = 0.064
as a descriptive count, not as a per-task result.

Largest consistent movers: winogrande (+0.007/+0.025/+0.009 across sizes),
arc_easy (+0.000/+0.011/+0.015), mathqa (+0.007/+0.021/+0.010).
commonsense_qa is flat near 0.20 in this format. The new campaign traces this
to almost constant A predictions and separately scores answer text without
option letters. The failure is therefore reported with its prompt/scoring
convention rather than interpreted as a scale-wide absence of task information.

## Suggested next steps

- Train a 600M DAGFormer: the 600M baseline is the only unpaired point, and the
  BPB trend would be much stronger with a fourth size.
- The retained 300M baseline step-9000 checkpoint has now been evaluated
  against DAGFormer step 9000 at the same processed-token budget; see the
  September 17 campaign.
- Keep generated GSM8K accuracy separate from joint or conditional-answer
  BPB. These runs do not determine how much additional scale, training or
  data specialization would be needed to improve generated solutions.
