# Judged chat eval (MT-Bench first turns, LLM judge)

Judge: Qwen3-Coder-30B-A3B-Instruct, greedy, MT-Bench judge prompts (math, reasoning and coding with the GPT-4 reference answer). Answers: greedy, up to 256 new tokens, ChatML as in SFT. Pairwise verdicts in both answer orders; win rate = routed wins + half the ties, per question averaged over the two orders; W/T/L counts a win or loss only when both orders agree. 95% CI: bootstrap over the 80 questions. Single runs of every model.

## Pairwise: routed vs dense

| SFT data | routed | dense | setting | win rate [95% CI] | W / T / L | order-consistent | judged A |
|---|---|---|---|---|---|---|---|
| Alpaca-Dolly | 300m_corrected | 300m_dense | 300M, equal tokens | 0.569 [0.500, 0.637] | 16 / 54 / 10 | 38/80 | 0.28 |
| Alpaca-Dolly | 300m_modular | 300m_dense | 300M, equal tokens | 0.588 [0.519, 0.656] | 19 / 51 / 10 | 40/80 | 0.29 |
| Alpaca-Dolly | 1b_corrected_5b | 1b_dense_5b | 1B, 5B tokens each | 0.584 [0.506, 0.662] | 26 / 42 / 12 | 45/80 | 0.38 |
| Alpaca-Dolly | 1b_corrected_5b | 1b_dense_10b | 1B, about equal training compute | 0.559 [0.478, 0.641] | 23 / 40 / 17 | 45/80 | 0.38 |
| Alpaca-Dolly | 1b_corrected_10b | 1b_dense_10b | 1B, 10B tokens each | 0.559 [0.484, 0.634] | 19 / 47 / 14 | 39/80 | 0.35 |
| SmolTalk | 300m_corrected | 300m_dense | 300M, equal tokens | 0.691 [0.631, 0.750] | 27 / 51 / 2 | 36/80 | 0.33 |
| SmolTalk | 300m_modular | 300m_dense | 300M, equal tokens | 0.575 [0.509, 0.641] | 19 / 55 / 6 | 30/80 | 0.38 |
| SmolTalk | 1b_corrected_5b | 1b_dense_5b | 1B, 5B tokens each | 0.588 [0.516, 0.659] | 23 / 49 / 8 | 34/80 | 0.38 |
| SmolTalk | 1b_corrected_5b | 1b_dense_10b | 1B, about equal training compute | 0.597 [0.519, 0.672] | 27 / 43 / 10 | 42/80 | 0.39 |
| SmolTalk | 1b_corrected_10b | 1b_dense_10b | 1B, 10B tokens each | 0.644 [0.569, 0.716] | 30 / 41 / 9 | 42/80 | 0.42 |
| Alpaca-Dolly | 300m_corrected_seed2 | 300m_dense_seed2 | 300M, equal tokens, SFT seed 2 | 0.606 [0.541, 0.672] | 19 / 53 / 8 | 38/80 | 0.38 |
| Alpaca-Dolly | 300m_modular_seed2 | 300m_dense_seed2 | 300M, equal tokens, SFT seed 2 | 0.603 [0.534, 0.672] | 20 / 52 / 8 | 39/80 | 0.31 |
| SmolTalk | 300m_corrected_seed2 | 300m_dense_seed2 | 300M, equal tokens, SFT seed 2 | 0.641 [0.569, 0.709] | 30 / 45 / 5 | 43/80 | 0.42 |
| SmolTalk | 300m_modular_seed2 | 300m_dense_seed2 | 300M, equal tokens, SFT seed 2 | 0.575 [0.503, 0.644] | 20 / 51 / 8 | 30/80 | 0.36 |

Per category (routed win rate, 10 questions each, so +-0.15 is noise):

| routed vs dense | writing | roleplay | reasoning | math | coding | extraction | stem | humanities |
|---|---|---|---|---|---|---|---|---|
| 300m_corrected_alpaca_dolly vs 300m_dense_alpaca_dolly | 0.72 | 0.57 | 0.47 | 0.40 | 0.60 | 0.55 | 0.75 | 0.47 |
| 300m_modular_alpaca_dolly vs 300m_dense_alpaca_dolly | 0.80 | 0.53 | 0.42 | 0.50 | 0.50 | 0.72 | 0.55 | 0.68 |
| 1b_corrected_5b_alpaca_dolly vs 1b_dense_5b_alpaca_dolly | 0.55 | 0.60 | 0.42 | 0.45 | 0.42 | 0.62 | 0.75 | 0.85 |
| 1b_corrected_5b_alpaca_dolly vs 1b_dense_10b_alpaca_dolly | 0.42 | 0.82 | 0.50 | 0.28 | 0.53 | 0.65 | 0.53 | 0.75 |
| 1b_corrected_10b_alpaca_dolly vs 1b_dense_10b_alpaca_dolly | 0.40 | 0.75 | 0.42 | 0.33 | 0.65 | 0.65 | 0.53 | 0.75 |
| 300m_corrected_smol_smoltalk vs 300m_dense_smol_smoltalk | 0.82 | 0.82 | 0.53 | 0.80 | 0.65 | 0.57 | 0.60 | 0.72 |
| 300m_modular_smol_smoltalk vs 300m_dense_smol_smoltalk | 0.68 | 0.62 | 0.35 | 0.50 | 0.55 | 0.57 | 0.70 | 0.62 |
| 1b_corrected_5b_smol_smoltalk vs 1b_dense_5b_smol_smoltalk | 0.78 | 0.62 | 0.55 | 0.53 | 0.42 | 0.57 | 0.70 | 0.53 |
| 1b_corrected_5b_smol_smoltalk vs 1b_dense_10b_smol_smoltalk | 0.82 | 0.62 | 0.50 | 0.38 | 0.60 | 0.68 | 0.65 | 0.53 |
| 1b_corrected_10b_smol_smoltalk vs 1b_dense_10b_smol_smoltalk | 0.80 | 0.60 | 0.78 | 0.55 | 0.50 | 0.65 | 0.57 | 0.70 |
| 300m_corrected_seed2_alpaca_dolly vs 300m_dense_seed2_alpaca_dolly | 0.55 | 0.47 | 0.68 | 0.60 | 0.60 | 0.60 | 0.72 | 0.62 |
| 300m_modular_seed2_alpaca_dolly vs 300m_dense_seed2_alpaca_dolly | 0.90 | 0.57 | 0.57 | 0.53 | 0.42 | 0.65 | 0.70 | 0.47 |
| 300m_corrected_seed2_smol_smoltalk vs 300m_dense_seed2_smol_smoltalk | 0.78 | 0.68 | 0.50 | 0.68 | 0.65 | 0.47 | 0.50 | 0.88 |
| 300m_modular_seed2_smol_smoltalk vs 300m_dense_seed2_smol_smoltalk | 0.53 | 0.60 | 0.50 | 0.78 | 0.62 | 0.57 | 0.50 | 0.50 |

## Single-answer scores (1-10)

| model | mean score | n | mean answer tokens | ended on <\|im_end\|> |
|---|---|---|---|---|
| 300m_dense_alpaca_dolly | 1.91 | 80 | 109 | 0.74 |
| 300m_corrected_alpaca_dolly | 2.15 | 80 | 109 | 0.72 |
| 300m_modular_alpaca_dolly | 1.95 | 80 | 114 | 0.71 |
| 1b_dense_5b_alpaca_dolly | 2.06 | 80 | 103 | 0.74 |
| 1b_corrected_5b_alpaca_dolly | 2.14 | 80 | 122 | 0.75 |
| 1b_dense_10b_alpaca_dolly | 2.19 | 80 | 111 | 0.71 |
| 1b_corrected_10b_alpaca_dolly | 2.33 | 80 | 122 | 0.70 |
| 300m_dense_smol_smoltalk | 2.12 | 80 | 210 | 0.34 |
| 300m_corrected_smol_smoltalk | 2.52 | 80 | 210 | 0.35 |
| 300m_modular_smol_smoltalk | 2.34 | 80 | 208 | 0.38 |
| 1b_dense_5b_smol_smoltalk | 2.06 | 80 | 210 | 0.34 |
| 1b_corrected_5b_smol_smoltalk | 2.35 | 80 | 200 | 0.40 |
| 1b_dense_10b_smol_smoltalk | 2.12 | 80 | 204 | 0.38 |
| 1b_corrected_10b_smol_smoltalk | 2.52 | 80 | 210 | 0.38 |
| 300m_dense_seed2_alpaca_dolly | 1.99 | 80 | 117 | 0.66 |
| 300m_corrected_seed2_alpaca_dolly | 2.14 | 80 | 112 | 0.71 |
| 300m_modular_seed2_alpaca_dolly | 2.01 | 80 | 107 | 0.74 |
| 300m_dense_seed2_smol_smoltalk | 2.16 | 80 | 211 | 0.31 |
| 300m_corrected_seed2_smol_smoltalk | 2.45 | 80 | 199 | 0.41 |
| 300m_modular_seed2_smol_smoltalk | 2.40 | 80 | 206 | 0.38 |

Paired score difference, routed - dense:

| routed vs dense | diff [95% CI] |
|---|---|
| 300m_corrected_alpaca_dolly vs 300m_dense_alpaca_dolly | +0.24 [+0.06, +0.45] |
| 300m_modular_alpaca_dolly vs 300m_dense_alpaca_dolly | +0.04 [-0.24, +0.25] |
| 1b_corrected_5b_alpaca_dolly vs 1b_dense_5b_alpaca_dolly | +0.07 [-0.24, +0.33] |
| 1b_corrected_5b_alpaca_dolly vs 1b_dense_10b_alpaca_dolly | -0.05 [-0.35, +0.19] |
| 1b_corrected_10b_alpaca_dolly vs 1b_dense_10b_alpaca_dolly | +0.14 [-0.15, +0.41] |
| 300m_corrected_smol_smoltalk vs 300m_dense_smol_smoltalk | +0.40 [+0.23, +0.59] |
| 300m_modular_smol_smoltalk vs 300m_dense_smol_smoltalk | +0.21 [+0.04, +0.40] |
| 1b_corrected_5b_smol_smoltalk vs 1b_dense_5b_smol_smoltalk | +0.29 [+0.11, +0.46] |
| 1b_corrected_5b_smol_smoltalk vs 1b_dense_10b_smol_smoltalk | +0.23 [+0.06, +0.38] |
| 1b_corrected_10b_smol_smoltalk vs 1b_dense_10b_smol_smoltalk | +0.40 [+0.23, +0.59] |
| 300m_corrected_seed2_alpaca_dolly vs 300m_dense_seed2_alpaca_dolly | +0.15 [-0.04, +0.36] |
| 300m_modular_seed2_alpaca_dolly vs 300m_dense_seed2_alpaca_dolly | +0.03 [-0.28, +0.31] |
| 300m_corrected_seed2_smol_smoltalk vs 300m_dense_seed2_smol_smoltalk | +0.29 [+0.14, +0.44] |
| 300m_modular_seed2_smol_smoltalk vs 300m_dense_seed2_smol_smoltalk | +0.24 [+0.09, +0.40] |
