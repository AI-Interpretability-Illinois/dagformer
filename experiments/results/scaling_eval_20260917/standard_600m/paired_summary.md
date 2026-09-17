# Paired evaluation — primary metrics

Positive differences favor DAGFormer. Intervals resample the same documents
for both models; they do not measure variation across training seeds.

| Size | Task | Baseline | DAGFormer | Difference | Paired 95% CI |
|---|---|---:|---:|---:|---|
| 600m | wikitext | 0.9762 | 0.9431 | +0.0330 | [+0.0310, +0.0352] |
| 600m | lambada_openai | 0.3377 | 0.3802 | +0.0425 | [+0.0310, +0.0545] |
| 600m | hellaswag | 0.3357 | 0.3550 | +0.0193 | [+0.0134, +0.0254] |
| 600m | piqa | 0.6621 | 0.6654 | +0.0033 | [-0.0120, +0.0185] |
| 600m | arc_easy | 0.4407 | 0.4621 | +0.0215 | [+0.0046, +0.0383] |
| 600m | arc_challenge | 0.2363 | 0.2321 | -0.0043 | [-0.0239, +0.0154] |
| 600m | winogrande | 0.5162 | 0.5107 | -0.0055 | [-0.0387, +0.0292] |
| 600m | openbookqa | 0.2960 | 0.2600 | -0.0360 | [-0.0620, -0.0100] |
| 600m | sciq | 0.7910 | 0.8260 | +0.0350 | [+0.0150, +0.0550] |
| 600m | boolq | 0.5661 | 0.6024 | +0.0364 | [+0.0208, +0.0523] |
| 600m | mathqa | 0.2221 | 0.2231 | +0.0010 | [-0.0131, +0.0151] |
| 600m | commonsense_qa | 0.1957 | 0.1949 | -0.0008 | [-0.0041, +0.0016] |
| 600m | social_iqa | 0.3813 | 0.3797 | -0.0015 | [-0.0174, +0.0143] |
| 600m | gsm8k_bpb | 1.2122 | 1.1165 | +0.0957 | [+0.0935, +0.0978] |
