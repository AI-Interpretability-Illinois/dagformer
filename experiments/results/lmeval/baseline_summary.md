# Downstream benchmark — baselines (lm-evaluation-harness 0.4.9.1)

Eval set: 0-shot loglikelihood on full datasets (no `--limit`). Wikitext is word-level perplexity.

All accuracies as %. Chance = HellaS 25, PIQA 50, ARC-E 25, Winogr 50, OBQA 25, SciQ 25, BoolQ 50.

| Model | Tokens | LAMBADA acc | LAMBADA ppl | HellaSwag | PIQA | ARC-E | Winogr | OBQA | SciQ | BoolQ | WT ppl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 75M baseline | 1.6B | 0.6 | 17921 | 24.5 | 49.9 | 27.3 | 50.2 | 27.4 | 27.4 | 41.1 | 475.8 |
| 150M baseline | 3.1B | 14.1 | 730.6 | 25.8 | 53.5 | 29.5 | 53.0 | 27.2 | 39.8 | 43.1 | 135.4 |
| 300M baseline | 6.3B | 27.2 | 81.5 | 28.9 | 61.7 | 36.4 | 50.5 | 26.4 | 54.0 | 58.5 | 58.3 |
| 1B baseline (ours) | 1.2B | 23.2 | 126.2 | 28.3 | 61.6 | 35.5 | 51.8 | 27.6 | 53.3 | 62.0 | 72.4 |
| OLMo2-1B ing1 | 5.0B | 63.6 | 5.3 | 68.3 | 76.6 | 69.4 | 62.9 | 39.6 | 94.6 | 66.4 | 14.3 |
| OLMo2-1B ing2 | 5.0B | 64.1 | 5.2 | 68.6 | 75.4 | 70.5 | 64.9 | 41.0 | 94.7 | 62.8 | 14.3 |
| OLMo2-1B ing3 | 5.0B | 64.1 | 5.1 | 67.9 | 75.7 | 71.2 | 64.3 | 41.0 | 94.9 | 64.3 | 14.3 |

Chance row: | _chance_ | — | — | — | 25 | 50 | 25 | 50 | 25 | 25 | 50 | — |
