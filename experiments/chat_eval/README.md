# Judged chat eval of the instruction-tuned models (phase 2 of the SFT evaluation)

The lm-eval suites (`experiments/results/lmeval/sft/`) measure multiple-choice accuracy and bits per
byte, which an instruction-tuned 300M/1B model barely moves. This eval asks the question those suites
cannot: given the same instruction tuning, does DAGFormer answer instructions better than its dense
baseline, as judged by a stronger LLM?

## Protocol

- **Prompts:** the 80 MT-Bench first turns (8 categories x 10; `data/mt_bench_question.jsonl` from
  FastChat `llm_judge`), single-turn only, since the models are 1024-token-context SFT models.
- **Answers** (`scripts/chat_eval_generate.py`, Delta, `scripts/slurm/chat_eval_generate.slurm`):
  each SFT export answers in the ChatML format it was tuned on, greedy, up to 256 new tokens.
- **Judge** (`scripts/chat_eval_judge.py`, timan1, vLLM): Qwen3-Coder-30B-A3B-Instruct, greedy, the
  MT-Bench judge prompts (pairwise "pair-v2", single-answer "single-v1"; math, reasoning and coding
  questions with the GPT-4 reference answers in `data/mt_bench_reference_gpt4.jsonl`).
  - Pairwise, every comparison in `models.yaml`, in both answer orders to cancel position bias.
  - A 1-10 score for every answer.
- **Report** (`scripts/chat_eval_report.py` -> `RESULTS.md`, `results.json`):
  - Win rate of the routed model (ties count half; per question averaged over the two orders).
  - W/T/L under the MT-Bench rule: a win or loss only if both orders agree.
  - Per-category win rates.
  - Mean scores, and the paired routed - dense score difference.
  - 95% CIs from a bootstrap over questions.

## Comparisons (`models.yaml`), each for Alpaca-Dolly and SmolTalk SFT

| routed | dense | what it tests |
|---|---|---|
| 300M fourway_corrected | 300M dense | main method, equal tokens |
| 300M fourway_modular | 300M dense | modular variant, equal tokens |
| 1B fourway_corrected, 5B tokens | 1B dense, 5B tokens | equal tokens |
| 1B fourway_corrected, 5B tokens | 1B dense, 10B tokens | about equal training compute (86 vs 81 EFLOP) |
| 1B fourway_corrected, 10B tokens | 1B dense, 10B tokens | equal tokens; export lands ~2026-10-03, re-run steps 1-3 then |

## Running it

```
sbatch scripts/slurm/chat_eval_generate.slurm                 # Delta: answers -> experiments/chat_eval/answers/
# timan1 (GPU grant from dagformer-ce; vLLM env and weights from the swe-memory setup):
rsync experiments/chat_eval/{models.yaml,data,answers} scripts/chat_eval_judge.py timan1:/srv/local/xy51/chat_eval/
ssh timan1 'setsid nohup /srv/local/xy51/chat_eval/run_judge.sh < /dev/null > /dev/null 2>&1 &'
rsync timan1:/srv/local/xy51/chat_eval/judgments/ experiments/chat_eval/judgments/
python scripts/chat_eval_report.py
```

All three steps resume: finished answer files and already-judged (model, question, order) keys are skipped.

## Caveats

- Models at this size often loop or repeat under greedy decoding, and both families do it. The judge
  compares two weak answers, so absolute scores sit near the bottom of the scale; the pairwise win rate
  is the number to read.
- Single judge, single run per model, 80 prompts: a win rate's 95% CI is roughly +-0.07, and the
  per-category rates (10 prompts each) are only indicative.
- Qwen3-Coder is a coding-tuned model used as a general judge because it is the strongest model we can
  serve locally; it is the same judge for every comparison.

## Results (2026-10-01; 300M and 1B 5B-token models, full tables in `RESULTS.md`)

All 1280 pairwise verdicts and 960 scores parse. The first pass used a 512-token judge budget, which
cut off 145 judgments (mostly math, coding and extraction, where the judge works the problem itself).
Those 145 were re-judged with 2048 tokens (`--retry-errors`).

| SFT data | DAGFormer | dense | win rate [95% CI] | decisive W / L |
|---|---|---|---|---|
| Alpaca-Dolly | 300M corrected | 300M | 0.569 [0.500, 0.637] | 16 / 10 |
| Alpaca-Dolly | 300M modular | 300M | 0.588 [0.519, 0.656] | 19 / 10 |
| Alpaca-Dolly | 1B corrected, 5B tok | 1B, 5B tok | 0.584 [0.506, 0.662] | 26 / 12 |
| Alpaca-Dolly | 1B corrected, 5B tok | 1B, 10B tok | 0.559 [0.478, 0.641] | 23 / 17 |
| SmolTalk | 300M corrected | 300M | 0.691 [0.631, 0.750] | 27 / 2 |
| SmolTalk | 300M modular | 300M | 0.575 [0.509, 0.641] | 19 / 6 |
| SmolTalk | 1B corrected, 5B tok | 1B, 5B tok | 0.588 [0.516, 0.659] | 23 / 8 |
| SmolTalk | 1B corrected, 5B tok | 1B, 10B tok | 0.597 [0.519, 0.672] | 27 / 10 |

- **DAGFormer is preferred in all 8 comparisons.** The 95% CI excludes 0.5 in 6 of them. The two
  exceptions are both Alpaca-Dolly: 300M corrected sits exactly at the bound, and 1B 5B against
  dense 10B does not exclude it.
- **The largest effect is 300M fourway_corrected after SmolTalk:** 0.69, with 27 decisive wins
  against 2 losses.
- **At about equal training compute** (1B DAGFormer at 5B tokens against 1B dense at 10B), DAGFormer
  is still preferred: 0.60 after SmolTalk (significant) and 0.56 after Alpaca-Dolly (not).
- **Absolute scores sit near 2/10 for every model,** as expected at this size. The paired score
  difference agrees with the win rates after SmolTalk (+0.21 to +0.40, all CIs above 0). After
  Alpaca-Dolly it is significant only for 300M corrected (+0.24).
- **The judge has a strong second-position bias:** it picks answer A in only 28-39% of verdicts.
  Judging both orders cancels it, but only 30-45 of 80 questions get the same verdict in both orders,
  hence the many ties.
