# LAMBADA: does the answer already occur in context?

This posthoc diagnostic was prompted by the observed LAMBADA gains after fixed
head editing. The grouping is determined from the saved prompt and gold answer,
before using each document's change in accuracy. ‘Seen’ means the exact answer
token sequence occurs in the retained input context. It does not include semantic
equivalents or changes in capitalization/tokenization. Encoding and left truncation
match the evaluation harness. Intervals are unadjusted paired document bootstraps.

| Variant | Answer occurrence | Documents | Unedited accuracy | Edited accuracy | Difference (points) | Paired 95% interval |
|---|---|---:|---:|---:|---:|---|
| 300m-dagformer__context_pred_gamma1.5 | answer_seen | 3791 | 39.15% | 40.78% | +1.64 | [+1.11, +2.16] |
| 300m-dagformer__context_pred_gamma1.5 | answer_not_seen | 1362 | 5.21% | 5.29% | +0.07 | [-0.29, +0.44] |
| 300m-dagformer__context_corr_gamma1.25 | answer_seen | 3791 | 39.15% | 41.39% | +2.24 | [+1.61, +2.85] |
| 300m-dagformer__context_corr_gamma1.25 | answer_not_seen | 1362 | 5.21% | 5.73% | +0.51 | [+0.07, +1.03] |
| 300m-dagformer__context_both_gamma1.25 | answer_seen | 3791 | 39.15% | 42.15% | +3.01 | [+2.32, +3.67] |
| 300m-dagformer__context_both_gamma1.25 | answer_not_seen | 1362 | 5.21% | 5.65% | +0.44 | [-0.00, +0.95] |

The JSON retains group membership and the first gain/loss document IDs.
An association with repeated answers does not identify which computation
produced a correct prediction or establish a causal mediation mechanism.
