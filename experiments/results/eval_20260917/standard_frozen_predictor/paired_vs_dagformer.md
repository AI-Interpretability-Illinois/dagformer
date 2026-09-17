# Paired comparison against dagformer

Positive differences favor the variant. Intervals resample documents, not training seeds. BPB decreases and accuracy increases are positive.

| Variant | Task | Reference | Variant | Difference | Paired 95% interval |
|---|---|---:|---:|---:|---|
| 150m-dagformer__pred_position | wikitext | 1.1521 | 1.1523 | -0.0002 | [-0.0003, -0.0001] |
| 150m-dagformer__pred_position | lambada_openai | 0.2137 | 0.2123 | -0.0014 | [-0.0033, +0.0006] |
| 150m-dagformer__pred_position | hellaswag | 0.2731 | 0.2736 | +0.0005 | [-0.0009, +0.0020] |
| 150m-dagformer__pred_position | piqa | 0.6017 | 0.6007 | -0.0011 | [-0.0054, +0.0033] |
| 150m-dagformer__pred_position | arc_easy | 0.3485 | 0.3493 | +0.0008 | [-0.0021, +0.0038] |
| 150m-dagformer__pred_position | arc_challenge | 0.2244 | 0.2244 | +0.0000 | [-0.0034, +0.0034] |
| 150m-dagformer__pred_position | winogrande | 0.5233 | 0.5257 | +0.0024 | [-0.0110, +0.0158] |
| 150m-dagformer__pred_position | openbookqa | 0.2560 | 0.2580 | +0.0020 | [-0.0040, +0.0100] |
| 150m-dagformer__pred_position | sciq | 0.6000 | 0.6000 | +0.0000 | [-0.0050, +0.0050] |
| 150m-dagformer__pred_position | boolq | 0.5067 | 0.5076 | +0.0009 | [-0.0031, +0.0049] |
| 150m-dagformer__pred_position | mathqa | 0.2228 | 0.2235 | +0.0007 | [-0.0023, +0.0037] |
| 150m-dagformer__pred_position | commonsense_qa | 0.1966 | 0.1966 | +0.0000 | [+0.0000, +0.0000] |
| 150m-dagformer__pred_position | social_iqa | 0.3639 | 0.3628 | -0.0010 | [-0.0046, +0.0026] |
| 150m-dagformer__pred_position | gsm8k_bpb | 1.5187 | 1.5225 | -0.0038 | [-0.0040, -0.0036] |
| 300m-dagformer__pred_position | wikitext | 1.0225 | 1.0226 | -0.0001 | [-0.0002, -0.0001] |
| 300m-dagformer__pred_position | lambada_openai | 0.3018 | 0.2996 | -0.0021 | [-0.0047, +0.0004] |
| 300m-dagformer__pred_position | hellaswag | 0.3145 | 0.3146 | +0.0001 | [-0.0011, +0.0014] |
| 300m-dagformer__pred_position | piqa | 0.6436 | 0.6431 | -0.0005 | [-0.0049, +0.0038] |
| 300m-dagformer__pred_position | arc_easy | 0.3847 | 0.3847 | +0.0000 | [-0.0034, +0.0034] |
| 300m-dagformer__pred_position | arc_challenge | 0.2321 | 0.2329 | +0.0009 | [-0.0034, +0.0051] |
| 300m-dagformer__pred_position | winogrande | 0.5217 | 0.5178 | -0.0039 | [-0.0158, +0.0079] |
| 300m-dagformer__pred_position | openbookqa | 0.2760 | 0.2780 | +0.0020 | [+0.0000, +0.0060] |
| 300m-dagformer__pred_position | sciq | 0.6980 | 0.6980 | +0.0000 | [-0.0040, +0.0040] |
| 300m-dagformer__pred_position | boolq | 0.6080 | 0.6086 | +0.0006 | [-0.0018, +0.0031] |
| 300m-dagformer__pred_position | mathqa | 0.2278 | 0.2285 | +0.0007 | [-0.0017, +0.0030] |
| 300m-dagformer__pred_position | commonsense_qa | 0.1957 | 0.1957 | +0.0000 | [+0.0000, +0.0000] |
| 300m-dagformer__pred_position | social_iqa | 0.3628 | 0.3628 | +0.0000 | [-0.0026, +0.0031] |
| 300m-dagformer__pred_position | gsm8k_bpb | 1.3314 | 1.3344 | -0.0030 | [-0.0031, -0.0029] |
| 75m-dagformer__pred_position | wikitext | 1.3472 | 1.3477 | -0.0005 | [-0.0006, -0.0004] |
| 75m-dagformer__pred_position | lambada_openai | 0.0704 | 0.0728 | +0.0023 | [+0.0006, +0.0041] |
| 75m-dagformer__pred_position | hellaswag | 0.2625 | 0.2635 | +0.0010 | [-0.0005, +0.0025] |
| 75m-dagformer__pred_position | piqa | 0.5577 | 0.5560 | -0.0016 | [-0.0060, +0.0027] |
| 75m-dagformer__pred_position | arc_easy | 0.3018 | 0.3022 | +0.0004 | [-0.0021, +0.0029] |
| 75m-dagformer__pred_position | arc_challenge | 0.2082 | 0.2073 | -0.0009 | [-0.0051, +0.0034] |
| 75m-dagformer__pred_position | winogrande | 0.5138 | 0.5091 | -0.0047 | [-0.0166, +0.0063] |
| 75m-dagformer__pred_position | openbookqa | 0.2500 | 0.2480 | -0.0020 | [-0.0100, +0.0060] |
| 75m-dagformer__pred_position | sciq | 0.4640 | 0.4650 | +0.0010 | [-0.0030, +0.0050] |
| 75m-dagformer__pred_position | boolq | 0.4248 | 0.4226 | -0.0021 | [-0.0070, +0.0024] |
| 75m-dagformer__pred_position | mathqa | 0.2104 | 0.2080 | -0.0023 | [-0.0060, +0.0013] |
| 75m-dagformer__pred_position | commonsense_qa | 0.1957 | 0.1957 | +0.0000 | [+0.0000, +0.0000] |
| 75m-dagformer__pred_position | social_iqa | 0.3414 | 0.3414 | +0.0000 | [-0.0036, +0.0036] |
| 75m-dagformer__pred_position | gsm8k_bpb | 1.7591 | 1.7616 | -0.0025 | [-0.0028, -0.0023] |
