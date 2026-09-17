# Paired comparison against dagformer

Positive differences favor the variant. Intervals resample documents, not training seeds. BPB decreases and accuracy increases are positive.

| Variant | Task | Reference | Variant | Difference | Paired 95% interval |
|---|---|---:|---:|---:|---|
| 300m-dagformer__context_corr_gamma1.25 | wikitext | 1.0225 | 1.0243 | -0.0018 | [-0.0021, -0.0016] |
| 300m-dagformer__context_corr_gamma1.25 | lambada_openai | 0.3018 | 0.3196 | +0.0179 | [+0.0132, +0.0225] |
| 300m-dagformer__context_corr_gamma1.25 | hellaswag | 0.3145 | 0.3146 | +0.0001 | [-0.0017, +0.0019] |
| 300m-dagformer__context_corr_gamma1.25 | piqa | 0.6436 | 0.6409 | -0.0027 | [-0.0076, +0.0016] |
| 300m-dagformer__context_corr_gamma1.25 | arc_easy | 0.3847 | 0.3834 | -0.0013 | [-0.0059, +0.0034] |
| 300m-dagformer__context_corr_gamma1.25 | arc_challenge | 0.2321 | 0.2346 | +0.0026 | [-0.0034, +0.0085] |
| 300m-dagformer__context_corr_gamma1.25 | winogrande | 0.5217 | 0.5201 | -0.0016 | [-0.0150, +0.0118] |
| 300m-dagformer__context_corr_gamma1.25 | openbookqa | 0.2760 | 0.2780 | +0.0020 | [+0.0000, +0.0060] |
| 300m-dagformer__context_corr_gamma1.25 | sciq | 0.6980 | 0.7070 | +0.0090 | [+0.0010, +0.0180] |
| 300m-dagformer__context_corr_gamma1.25 | boolq | 0.6080 | 0.6086 | +0.0006 | [-0.0028, +0.0040] |
| 300m-dagformer__context_corr_gamma1.25 | mathqa | 0.2278 | 0.2325 | +0.0047 | [+0.0003, +0.0090] |
| 300m-dagformer__context_corr_gamma1.25 | commonsense_qa | 0.1957 | 0.1957 | +0.0000 | [+0.0000, +0.0000] |
| 300m-dagformer__context_corr_gamma1.25 | social_iqa | 0.3628 | 0.3613 | -0.0015 | [-0.0061, +0.0031] |
| 300m-dagformer__context_corr_gamma1.25 | gsm8k_bpb | 1.3314 | 1.3359 | -0.0045 | [-0.0048, -0.0042] |
