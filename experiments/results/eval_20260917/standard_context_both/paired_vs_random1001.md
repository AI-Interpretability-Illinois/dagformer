# Paired comparison against dagformer

Reference results: `experiments/results/eval_20260917/standard_context_random1001`.

Positive differences favor the variant. Intervals resample documents, not training seeds. BPB decreases and accuracy increases are positive.

| Variant | Task | Reference | Variant | Difference | Paired 95% interval |
|---|---|---:|---:|---:|---|
| 300m-dagformer__context_both_gamma1.25 | wikitext | 1.0227 | 1.0268 | -0.0041 | [-0.0045, -0.0037] |
| 300m-dagformer__context_both_gamma1.25 | lambada_openai | 0.3111 | 0.3251 | +0.0140 | [+0.0087, +0.0192] |
| 300m-dagformer__context_both_gamma1.25 | hellaswag | 0.3147 | 0.3148 | +0.0001 | [-0.0023, +0.0025] |
| 300m-dagformer__context_both_gamma1.25 | piqa | 0.6436 | 0.6415 | -0.0022 | [-0.0082, +0.0038] |
| 300m-dagformer__context_both_gamma1.25 | arc_easy | 0.3859 | 0.3792 | -0.0067 | [-0.0118, -0.0017] |
| 300m-dagformer__context_both_gamma1.25 | arc_challenge | 0.2355 | 0.2338 | -0.0017 | [-0.0094, +0.0060] |
| 300m-dagformer__context_both_gamma1.25 | winogrande | 0.5241 | 0.5154 | -0.0087 | [-0.0229, +0.0055] |
| 300m-dagformer__context_both_gamma1.25 | openbookqa | 0.2760 | 0.2760 | +0.0000 | [-0.0080, +0.0080] |
| 300m-dagformer__context_both_gamma1.25 | sciq | 0.6990 | 0.7060 | +0.0070 | [-0.0020, +0.0160] |
| 300m-dagformer__context_both_gamma1.25 | boolq | 0.6089 | 0.6098 | +0.0009 | [-0.0031, +0.0049] |
| 300m-dagformer__context_both_gamma1.25 | mathqa | 0.2298 | 0.2312 | +0.0013 | [-0.0037, +0.0067] |
| 300m-dagformer__context_both_gamma1.25 | commonsense_qa | 0.1957 | 0.1957 | +0.0000 | [+0.0000, +0.0000] |
| 300m-dagformer__context_both_gamma1.25 | social_iqa | 0.3639 | 0.3634 | -0.0005 | [-0.0056, +0.0046] |
| 300m-dagformer__context_both_gamma1.25 | gsm8k_bpb | 1.3301 | 1.3399 | -0.0097 | [-0.0102, -0.0092] |
