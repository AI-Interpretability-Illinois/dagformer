# Paired evaluation — exact_match,strict-match

Positive differences favor DAGFormer. Intervals resample the same documents
for both models; they do not measure variation across training seeds.

| Size | Task | Baseline | DAGFormer | Difference | Paired 95% CI |
|---|---|---:|---:|---:|---|
| 600m | gsm8k | 0.0030 | 0.0076 | +0.0045 | [-0.0000, +0.0099] |
