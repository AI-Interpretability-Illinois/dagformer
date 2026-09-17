# Paired evaluation — acc,none

Positive differences favor DAGFormer. Intervals resample the same documents
for both models; they do not measure variation across training seeds.

| Size | Task | Baseline | DAGFormer | Difference | Paired 95% CI |
|---|---|---:|---:|---:|---|
| 75m | commonsense_qa_content | 0.1843 | 0.1835 | -0.0008 | [-0.0164, +0.0147] |
| 150m | commonsense_qa_content | 0.2391 | 0.2342 | -0.0049 | [-0.0238, +0.0139] |
| 300m | commonsense_qa_content | 0.2744 | 0.2760 | +0.0016 | [-0.0188, +0.0221] |
