# Paired comparison against dagformer

Reference results: `experiments/results/eval_20260917/standard_context_random1000`.

Positive differences favor the variant. Intervals resample documents, not training seeds. BPB decreases and accuracy increases are positive.

| Variant | Task | Reference | Variant | Difference | Paired 95% interval |
|---|---|---:|---:|---:|---|
| 300m-dagformer__context_both_gamma1.25 | wikitext | 1.0229 | 1.0268 | -0.0039 | [-0.0043, -0.0035] |
| 300m-dagformer__context_both_gamma1.25 | lambada_openai | 0.2952 | 0.3251 | +0.0299 | [+0.0239, +0.0359] |
| 300m-dagformer__context_both_gamma1.25 | hellaswag | 0.3126 | 0.3148 | +0.0022 | [-0.0002, +0.0046] |
| 300m-dagformer__context_both_gamma1.25 | piqa | 0.6425 | 0.6415 | -0.0011 | [-0.0071, +0.0049] |
| 300m-dagformer__context_both_gamma1.25 | arc_easy | 0.3838 | 0.3792 | -0.0046 | [-0.0101, +0.0008] |
| 300m-dagformer__context_both_gamma1.25 | arc_challenge | 0.2321 | 0.2338 | +0.0017 | [-0.0060, +0.0094] |
| 300m-dagformer__context_both_gamma1.25 | winogrande | 0.5162 | 0.5154 | -0.0008 | [-0.0150, +0.0134] |
| 300m-dagformer__context_both_gamma1.25 | openbookqa | 0.2740 | 0.2760 | +0.0020 | [-0.0040, +0.0100] |
| 300m-dagformer__context_both_gamma1.25 | sciq | 0.6930 | 0.7060 | +0.0130 | [+0.0020, +0.0240] |
| 300m-dagformer__context_both_gamma1.25 | boolq | 0.6070 | 0.6098 | +0.0028 | [-0.0015, +0.0070] |
| 300m-dagformer__context_both_gamma1.25 | mathqa | 0.2295 | 0.2312 | +0.0017 | [-0.0037, +0.0074] |
| 300m-dagformer__context_both_gamma1.25 | commonsense_qa | 0.1957 | 0.1957 | +0.0000 | [+0.0000, +0.0000] |
| 300m-dagformer__context_both_gamma1.25 | social_iqa | 0.3634 | 0.3634 | +0.0000 | [-0.0061, +0.0061] |
| 300m-dagformer__context_both_gamma1.25 | gsm8k_bpb | 1.3319 | 1.3399 | -0.0080 | [-0.0085, -0.0075] |
