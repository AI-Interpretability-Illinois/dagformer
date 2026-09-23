# Paired evaluation — primary metrics

Positive differences favor DAGFormer. Intervals resample the same documents
for both models; they do not measure variation across training seeds.

| Size | Task | Baseline | DAGFormer | Difference | Paired 95% CI |
|---|---|---:|---:|---:|---|
| 1b | wikitext | 0.9164 | 0.8806 | +0.0358 | [+0.0343, +0.0373] |
| 1b | lambada_openai | 0.4198 | 0.4698 | +0.0501 | [+0.0388, +0.0617] |
| 1b | hellaswag | 0.3789 | 0.4167 | +0.0378 | [+0.0319, +0.0440] |
| 1b | piqa | 0.6649 | 0.6877 | +0.0229 | [+0.0082, +0.0375] |
| 1b | arc_easy | 0.4907 | 0.5248 | +0.0341 | [+0.0189, +0.0492] |
| 1b | arc_challenge | 0.2594 | 0.2602 | +0.0009 | [-0.0179, +0.0196] |
| 1b | winogrande | 0.5170 | 0.5288 | +0.0118 | [-0.0205, +0.0442] |
| 1b | openbookqa | 0.2800 | 0.3140 | +0.0340 | [+0.0060, +0.0620] |
| 1b | sciq | 0.8130 | 0.8350 | +0.0220 | [+0.0030, +0.0410] |
| 1b | boolq | 0.5865 | 0.5116 | -0.0749 | [-0.0985, -0.0508] |
| 1b | mathqa | 0.2332 | 0.2385 | +0.0054 | [-0.0067, +0.0181] |
| 1b | commonsense_qa | 0.1925 | 0.2015 | +0.0090 | [-0.0238, +0.0418] |
| 1b | social_iqa | 0.3966 | 0.3884 | -0.0082 | [-0.0235, +0.0067] |
| 1b | gsm8k_bpb | 1.0298 | 0.9512 | +0.0786 | [+0.0768, +0.0805] |
