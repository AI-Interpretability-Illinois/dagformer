# Dense reference on the routing evaluation inputs

Dense and DAGFormer checkpoints see identical natural-text windows and synthetic
repetition sequences. Each period has 16 independent 1,024-token sequences, with
tokens sampled from the WikiText training-token marginal. Copy accuracy scores
teacher-forced next tokens over the second half. Intervals are paired normal
intervals across sequences. Models share backbone scale and processed-token
budget; their total parameter counts differ, as documented in the main report.
The exact scored targets have zero-based indices 513–1023: 511 per sequence.
This retains the original routing-dependence convention for every model, including
its omission of target 512. The separate historical-copy-head transfer uses the
first token after the initial block and reports its own matched reference.

| Scale | Period | Dense accuracy | DAGFormer accuracy | Difference (points) | Paired 95% interval |
|---|---:|---:|---:|---:|---|
| 75m | 64 | 68.87% | 93.04% | +24.17 | [+21.54, +26.80] |
| 75m | 128 | 55.98% | 83.87% | +27.89 | [+25.22, +30.56] |
| 75m | 256 | 43.74% | 69.86% | +26.13 | [+23.15, +29.10] |
| 150m | 64 | 92.66% | 98.70% | +6.04 | [+4.71, +7.38] |
| 150m | 128 | 87.00% | 95.50% | +8.50 | [+7.53, +9.47] |
| 150m | 256 | 78.31% | 89.89% | +11.57 | [+10.20, +12.94] |
| 300m | 64 | 97.86% | 99.78% | +1.92 | [+1.07, +2.77] |
| 300m | 128 | 96.39% | 95.84% | -0.55 | [-4.66, +3.56] |
| 300m | 256 | 93.08% | 93.58% | +0.50 | [-5.30, +6.30] |

At 75M and 150M, all three periods favor DAGFormer on this test. At 300M,
period 64 favors DAGFormer, while the period-128 and period-256 differences
have intervals spanning zero. Copying is not uniformly improved at every scale
and period. These synthetic results are separate from free-text generation.

## Natural-text reference

The following means use the same 128 nonoverlapping WikiText test windows
as the routing-substitution experiment. NLL decreases are improvements.

| Scale | Dense NLL | DAGFormer NLL | DAGFormer − dense |
|---|---:|---:|---:|
| 75m | 4.96919 | 4.76458 | -0.20462 |
| 150m | 4.28158 | 4.08631 | -0.19527 |
| 300m | 3.82582 | 3.61780 | -0.20801 |
