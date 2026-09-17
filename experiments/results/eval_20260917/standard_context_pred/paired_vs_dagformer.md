# Paired comparison against dagformer

Positive differences favor the variant. Intervals resample documents, not training seeds. BPB decreases and accuracy increases are positive.

| Variant | Task | Reference | Variant | Difference | Paired 95% interval |
|---|---|---:|---:|---:|---|
| 300m-dagformer__context_pred_gamma1.5 | wikitext | 1.0225 | 1.0254 | -0.0029 | [-0.0033, -0.0026] |
| 300m-dagformer__context_pred_gamma1.5 | lambada_openai | 0.3018 | 0.3140 | +0.0122 | [+0.0082, +0.0163] |
| 300m-dagformer__context_pred_gamma1.5 | hellaswag | 0.3145 | 0.3135 | -0.0010 | [-0.0032, +0.0012] |
| 300m-dagformer__context_pred_gamma1.5 | piqa | 0.6436 | 0.6415 | -0.0022 | [-0.0071, +0.0027] |
| 300m-dagformer__context_pred_gamma1.5 | arc_easy | 0.3847 | 0.3805 | -0.0042 | [-0.0088, +0.0000] |
| 300m-dagformer__context_pred_gamma1.5 | arc_challenge | 0.2321 | 0.2355 | +0.0034 | [-0.0034, +0.0102] |
| 300m-dagformer__context_pred_gamma1.5 | winogrande | 0.5217 | 0.5233 | +0.0016 | [-0.0118, +0.0142] |
| 300m-dagformer__context_pred_gamma1.5 | openbookqa | 0.2760 | 0.2760 | +0.0000 | [-0.0060, +0.0060] |
| 300m-dagformer__context_pred_gamma1.5 | sciq | 0.6980 | 0.7010 | +0.0030 | [-0.0040, +0.0110] |
| 300m-dagformer__context_pred_gamma1.5 | boolq | 0.6080 | 0.6089 | +0.0009 | [-0.0028, +0.0049] |
| 300m-dagformer__context_pred_gamma1.5 | mathqa | 0.2278 | 0.2288 | +0.0010 | [-0.0037, +0.0057] |
| 300m-dagformer__context_pred_gamma1.5 | commonsense_qa | 0.1957 | 0.1957 | +0.0000 | [+0.0000, +0.0000] |
| 300m-dagformer__context_pred_gamma1.5 | social_iqa | 0.3628 | 0.3659 | +0.0031 | [-0.0015, +0.0082] |
| 300m-dagformer__context_pred_gamma1.5 | gsm8k_bpb | 1.3314 | 1.3367 | -0.0053 | [-0.0057, -0.0049] |
