# Paired comparison against dagformer

Positive differences favor the variant. Intervals resample documents, not training seeds. BPB decreases and accuracy increases are positive.

| Variant | Task | Reference | Variant | Difference | Paired 95% interval |
|---|---|---:|---:|---:|---|
| 300m-dagformer__context_both_gamma1.25 | wikitext | 1.0225 | 1.0268 | -0.0043 | [-0.0048, -0.0039] |
| 300m-dagformer__context_both_gamma1.25 | lambada_openai | 0.3018 | 0.3251 | +0.0233 | [+0.0180, +0.0285] |
| 300m-dagformer__context_both_gamma1.25 | hellaswag | 0.3145 | 0.3148 | +0.0003 | [-0.0020, +0.0026] |
| 300m-dagformer__context_both_gamma1.25 | piqa | 0.6436 | 0.6415 | -0.0022 | [-0.0076, +0.0033] |
| 300m-dagformer__context_both_gamma1.25 | arc_easy | 0.3847 | 0.3792 | -0.0055 | [-0.0109, -0.0004] |
| 300m-dagformer__context_both_gamma1.25 | arc_challenge | 0.2321 | 0.2338 | +0.0017 | [-0.0051, +0.0085] |
| 300m-dagformer__context_both_gamma1.25 | winogrande | 0.5217 | 0.5154 | -0.0063 | [-0.0205, +0.0079] |
| 300m-dagformer__context_both_gamma1.25 | openbookqa | 0.2760 | 0.2760 | +0.0000 | [-0.0060, +0.0060] |
| 300m-dagformer__context_both_gamma1.25 | sciq | 0.6980 | 0.7060 | +0.0080 | [-0.0010, +0.0180] |
| 300m-dagformer__context_both_gamma1.25 | boolq | 0.6080 | 0.6098 | +0.0018 | [-0.0021, +0.0058] |
| 300m-dagformer__context_both_gamma1.25 | mathqa | 0.2278 | 0.2312 | +0.0034 | [-0.0017, +0.0084] |
| 300m-dagformer__context_both_gamma1.25 | commonsense_qa | 0.1957 | 0.1957 | +0.0000 | [+0.0000, +0.0000] |
| 300m-dagformer__context_both_gamma1.25 | social_iqa | 0.3628 | 0.3634 | +0.0005 | [-0.0046, +0.0056] |
| 300m-dagformer__context_both_gamma1.25 | gsm8k_bpb | 1.3314 | 1.3399 | -0.0085 | [-0.0090, -0.0080] |
