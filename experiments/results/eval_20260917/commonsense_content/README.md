# CommonsenseQA answer-text control

The prompt contains the question followed by `Answer:`. Each candidate answer
string is scored as a continuation, without presenting option letters. This
is a custom scoring control; upstream CommonsenseQA results remain separate.
Both raw continuation likelihood and the harness length-normalized likelihood
are retained. There are 1,221 validation items.

| Model | Raw-likelihood accuracy | Length-normalized accuracy |
|---|---:|---:|
| 75m-baseline | 18.43% | 24.73% |
| 75m-dagformer | 18.35% | 25.23% |
| 150m-baseline | 23.91% | 26.04% |
| 150m-dagformer | 23.42% | 27.44% |
| 300m-baseline | 27.44% | 28.99% |
| 300m-dagformer | 27.60% | 29.65% |
| 600m-baseline | 29.32% | 30.30% |

The 600M baseline is an unpaired reference with a different training budget.

With length normalization, the 150M DAGFormer pair improves by 1.39 percentage
points, with an unadjusted paired 95% interval of [0.08, 2.70]. Using raw
likelihood, its difference is -0.49 points, with interval [-2.38, 1.39].
The 75M and 300M intervals include zero under both conventions. Thus the
small 150M gain depends on the scoring convention.

- [Length-normalized paired statistics](paired_summary.md)
- [Raw-likelihood paired statistics](raw_accuracy_paired.md)
