# Paired comparison against dagformer

Reference results: `experiments/results/scaling_eval_20260917/standard_600m`.

Positive differences favor the variant. Intervals resample documents, not training seeds. BPB decreases and accuracy increases are positive.

| Variant | Task | Reference | Variant | Difference | Paired 95% interval |
|---|---|---:|---:|---:|---|
| 600m-dagformer__pred_position | wikitext | 0.9431 | 0.9434 | -0.0003 | [-0.0003, -0.0002] |
| 600m-dagformer__pred_position | lambada_openai | 0.3802 | 0.3788 | -0.0014 | [-0.0039, +0.0012] |
| 600m-dagformer__pred_position | hellaswag | 0.3550 | 0.3551 | +0.0001 | [-0.0011, +0.0013] |
| 600m-dagformer__pred_position | piqa | 0.6654 | 0.6687 | +0.0033 | [-0.0005, +0.0071] |
| 600m-dagformer__pred_position | arc_easy | 0.4621 | 0.4617 | -0.0004 | [-0.0038, +0.0029] |
| 600m-dagformer__pred_position | arc_challenge | 0.2321 | 0.2346 | +0.0026 | [-0.0009, +0.0068] |
| 600m-dagformer__pred_position | winogrande | 0.5107 | 0.5130 | +0.0024 | [-0.0103, +0.0150] |
| 600m-dagformer__pred_position | openbookqa | 0.2600 | 0.2620 | +0.0020 | [-0.0060, +0.0100] |
| 600m-dagformer__pred_position | sciq | 0.8260 | 0.8250 | -0.0010 | [-0.0030, +0.0000] |
| 600m-dagformer__pred_position | boolq | 0.6024 | 0.6012 | -0.0012 | [-0.0040, +0.0015] |
| 600m-dagformer__pred_position | mathqa | 0.2231 | 0.2241 | +0.0010 | [-0.0020, +0.0040] |
| 600m-dagformer__pred_position | commonsense_qa | 0.1949 | 0.1949 | +0.0000 | [+0.0000, +0.0000] |
| 600m-dagformer__pred_position | social_iqa | 0.3797 | 0.3808 | +0.0010 | [-0.0015, +0.0036] |
| 600m-dagformer__pred_position | gsm8k_bpb | 1.1165 | 1.1190 | -0.0025 | [-0.0026, -0.0024] |
