# Paired comparison against dagformer

Reference results: `/u/yurenh2/dagformer-20260917/results/completed_scaling/standard_1b`.

Positive differences favor the variant. Intervals resample documents, not training seeds. BPB decreases and accuracy increases are positive.

| Variant | Task | Reference | Variant | Difference | Paired 95% interval |
|---|---|---:|---:|---:|---|
| 1b-dagformer__pred_position | wikitext | 0.8806 | 0.8806 | -0.0001 | [-0.0001, -0.0000] |
| 1b-dagformer__pred_position | lambada_openai | 0.4698 | 0.4700 | +0.0002 | [-0.0021, +0.0025] |
| 1b-dagformer__pred_position | hellaswag | 0.4167 | 0.4175 | +0.0008 | [-0.0006, +0.0022] |
| 1b-dagformer__pred_position | piqa | 0.6877 | 0.6877 | +0.0000 | [-0.0038, +0.0038] |
| 1b-dagformer__pred_position | arc_easy | 0.5248 | 0.5231 | -0.0017 | [-0.0046, +0.0013] |
| 1b-dagformer__pred_position | arc_challenge | 0.2602 | 0.2602 | +0.0000 | [-0.0034, +0.0034] |
| 1b-dagformer__pred_position | winogrande | 0.5288 | 0.5241 | -0.0047 | [-0.0150, +0.0055] |
| 1b-dagformer__pred_position | openbookqa | 0.3140 | 0.3160 | +0.0020 | [+0.0000, +0.0060] |
| 1b-dagformer__pred_position | sciq | 0.8350 | 0.8330 | -0.0020 | [-0.0050, +0.0000] |
| 1b-dagformer__pred_position | boolq | 0.5116 | 0.5113 | -0.0003 | [-0.0049, +0.0043] |
| 1b-dagformer__pred_position | mathqa | 0.2385 | 0.2395 | +0.0010 | [-0.0017, +0.0037] |
| 1b-dagformer__pred_position | commonsense_qa | 0.2015 | 0.1990 | -0.0025 | [-0.0098, +0.0049] |
| 1b-dagformer__pred_position | social_iqa | 0.3884 | 0.3889 | +0.0005 | [-0.0010, +0.0026] |
| 1b-dagformer__pred_position | gsm8k_bpb | 0.9512 | 0.9520 | -0.0008 | [-0.0009, -0.0007] |
