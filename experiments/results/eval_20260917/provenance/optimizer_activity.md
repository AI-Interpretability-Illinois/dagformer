# External-predictor optimizer participation

The original Delta checkpoints contain update history for every external-predictor
parameter tensor and every correction-MLP tensor. All first and second Adam
moments are finite, and each tensor has at least one nonzero coordinate in both
moments. The initially zero output-head matrices also contain nonzero weights.
Thus omission of the predictor from the optimizer does not explain its weak
content dependence in these evaluations.

| Checkpoint | Predictor tensors | Correction tensors | Updates per inspected tensor |
|---|---:|---:|---:|
| 75M, step 3000 | 40 | 10 | 3,000 |
| 150M, step 6000 | 44 | 14 | 6,000 |
| 300M, step 9000 | 52 | 22 | 9,001 |
| 300M, step 10500 | 52 | 22 | 10,501 |

The mapping follows the training optimizer's order: predictor parameters excluding
`layer_biases`, followed by correction parameters, with `layer_biases` in a
separate group. Tensor counts and shapes match every inspected optimizer slot.
The saved bias group has zero weight decay; the other group has decay 0.1.
The source optimizer code is in `scripts/pretrain_dagformer.py`.

For the 300M model, the external predictor has 30,385,853 parameters. Its token
and position embeddings account for 26,738,688 of them (88.0%); the two Transformer
encoder blocks have 1,579,520 parameters, the trunk 132,096, and output heads plus
biases 1,935,549. Local corrections add a separate 1,924,736 parameters.
“External encoder” in the model comparison therefore includes a large independent
embedding table, not only the two Transformer blocks.

Nonzero optimizer moments establish gradient/update history. They do not establish
convergence or prove every historical training detail correct. Some embedding
coordinates have zero moments because they are unused; a tensor-level result is
not a claim that every vocabulary row was updated.

[Per-parameter records](optimizer_activity.json) retain original checkpoint paths,
shapes, counts, update steps, group learning rates and moment checks. The read-only
CPU audit can be repeated on Delta with `scripts/audit_optimizer_activity.py`, the
existing `training_budgets.json` manifest, and the four model labels above. It does
not modify checkpoints or run training.
