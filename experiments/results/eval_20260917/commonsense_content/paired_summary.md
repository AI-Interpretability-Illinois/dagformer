# Paired evaluation — primary metrics

Positive differences favor DAGFormer. Intervals resample the same documents
for both models; they do not measure variation across training seeds.

| Size | Task | Baseline | DAGFormer | Difference | Paired 95% CI |
|---|---|---:|---:|---:|---|
| 75m | commonsense_qa_content | 0.2473 | 0.2523 | +0.0049 | [-0.0082, +0.0188] |
| 150m | commonsense_qa_content | 0.2604 | 0.2744 | +0.0139 | [+0.0008, +0.0270] |
| 300m | commonsense_qa_content | 0.2899 | 0.2965 | +0.0066 | [-0.0098, +0.0221] |
