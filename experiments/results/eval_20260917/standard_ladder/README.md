# Trained 150M routing ladder

All seven checkpoints have 6,000 optimizer updates and 3,145,728,000 processed training tokens. Configurations use the same backbone dimensions, seed, data index, main learning rate, optimizer settings and learning-rate schedule. The paired uncertainty below describes evaluation documents for these individual training runs.

| Model | External routing | Correction | Total parameters | WikiText BPB ↓ | LAMBADA accuracy ↑ | SciQ accuracy ↑ |
|---|---|---|---:|---:|---:|---:|
| Baseline | none | none | 152,593,152 | 1.195896 | 15.62% | 55.00% |
| Static | learned constant | none | 152,594,447 | 1.186757 | 16.90% | 54.80% |
| Position table | learned by position | none | 153,919,232 | 1.189608 | 15.31% | 55.00% |
| Identity + correction | frozen sequential identity | local MLP | 153,448,335 | 1.152518 | 20.69% | 60.90% |
| Static + correction | learned constant | local MLP | 153,448,335 | 1.148903 | 21.48% | 62.60% |
| Position table + correction | learned by position | local MLP | 154,773,120 | 1.147767 | 22.41% | 60.90% |
| Full DAGFormer | causal encoder | local MLP | 182,561,679 | 1.152107 | 21.37% | 60.00% |

Static + correction uses 15.95% fewer total parameters than full DAGFormer. Its WikiText improvement over full DAGFormer is 0.003204 BPB, with paired 95% interval [0.002144, 0.004309]. Its LAMBADA difference is +0.12 percentage points, with interval [-0.74, +1.01]. Identity + correction also retains a similar WikiText score: its BPB increase versus full DAGFormer is 0.000411, with interval [-0.000907, 0.001741].

The position-table + correction variant has the best WikiText score in this set and improves LAMBADA over the full encoder, but its MathQA accuracy is 1.21 percentage points lower (paired interval [-2.31, -0.10]). The simpler variants are not uniformly best on every endpoint. All five variants remain below the constant-yes BoolQ baseline and collapse strongly toward A on the upstream CommonsenseQA format; see [label bias](label_bias.md).

[All 14 tasks versus full DAGFormer](paired_vs_dagformer.md) and [versus baseline](paired_vs_baseline.md) include the complete paired differences. Intervals use 10,000 paired document-bootstrap draws and are unadjusted for multiple comparisons. The [original-checkpoint budget audit](../provenance/training_budgets.json) verifies update counts and tokens per update.

These trained variants support the importance of local corrections at this scale. They do not establish that a separately trained external encoder is never useful at another scale, budget or dataset. Inference-time predictor substitution is analyzed separately in the main campaign report.
