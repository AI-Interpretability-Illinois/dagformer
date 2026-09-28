# Paired evaluation — primary metrics

Positive differences favor DAGFormer. Intervals resample the same documents
for both models; they do not measure variation across training seeds.

| Size | Task | Baseline | DAGFormer | Difference | Paired 95% CI |
|---|---|---:|---:|---:|---|
| 600m | wikitext | 0.9648 | 0.9309 | +0.0339 | [+0.0317, +0.0364] |
| 600m | lambada_openai | 0.3542 | 0.4002 | +0.0460 | [+0.0347, +0.0576] |
| 600m | hellaswag | 0.3443 | 0.3679 | +0.0236 | [+0.0175, +0.0301] |
| 600m | piqa | 0.6502 | 0.6610 | +0.0109 | [-0.0065, +0.0277] |
| 600m | arc_easy | 0.4659 | 0.4731 | +0.0072 | [-0.0109, +0.0257] |
| 600m | arc_challenge | 0.2355 | 0.2517 | +0.0162 | [-0.0060, +0.0384] |
| 600m | winogrande | 0.5241 | 0.5162 | -0.0079 | [-0.0410, +0.0268] |
| 600m | openbookqa | 0.2860 | 0.2680 | -0.0180 | [-0.0460, +0.0100] |
| 600m | sciq | 0.7880 | 0.8230 | +0.0350 | [+0.0160, +0.0540] |
| 600m | boolq | 0.6110 | 0.5682 | -0.0428 | [-0.0618, -0.0239] |
| 600m | mathqa | 0.2255 | 0.2348 | +0.0094 | [-0.0040, +0.0231] |
| 600m | commonsense_qa | 0.2023 | 0.2039 | +0.0016 | [-0.0320, +0.0369] |
| 600m | social_iqa | 0.4120 | 0.4063 | -0.0056 | [-0.0210, +0.0097] |
| 600m | gsm8k_bpb | 1.2104 | 1.1765 | +0.0339 | [+0.0313, +0.0365] |
