# Paired ordinary evaluation

Positive differences favor DAGFormer. Intervals resample the same documents
for both models; they do not measure variation across training seeds.

| Size | Task | Baseline | DAGFormer | Difference | Paired 95% CI |
|---|---|---:|---:|---:|---|
| 75m | gsm8k | 0.0182 | 0.0167 | -0.0015 | [-0.0106, +0.0076] |
| 300m | gsm8k | 0.0159 | 0.0152 | -0.0008 | [-0.0091, +0.0076] |
