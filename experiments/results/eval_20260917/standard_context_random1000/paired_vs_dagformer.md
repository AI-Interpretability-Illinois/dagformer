# Paired comparison against dagformer

Reference results: `experiments/results/eval_20260917/standard_matched`.

Positive differences favor the variant. Intervals resample documents, not training seeds. BPB decreases and accuracy increases are positive.

| Variant | Task | Reference | Variant | Difference | Paired 95% interval |
|---|---|---:|---:|---:|---|
| 300m-dagformer__context_both_gamma1.25_random1000_normmatched | wikitext | 1.0225 | 1.0229 | -0.0004 | [-0.0005, -0.0003] |
| 300m-dagformer__context_both_gamma1.25_random1000_normmatched | lambada_openai | 0.3018 | 0.2952 | -0.0066 | [-0.0109, -0.0023] |
| 300m-dagformer__context_both_gamma1.25_random1000_normmatched | hellaswag | 0.3145 | 0.3126 | -0.0019 | [-0.0034, -0.0004] |
| 300m-dagformer__context_both_gamma1.25_random1000_normmatched | piqa | 0.6436 | 0.6425 | -0.0011 | [-0.0049, +0.0027] |
| 300m-dagformer__context_both_gamma1.25_random1000_normmatched | arc_easy | 0.3847 | 0.3838 | -0.0008 | [-0.0046, +0.0029] |
| 300m-dagformer__context_both_gamma1.25_random1000_normmatched | arc_challenge | 0.2321 | 0.2321 | +0.0000 | [-0.0043, +0.0051] |
| 300m-dagformer__context_both_gamma1.25_random1000_normmatched | winogrande | 0.5217 | 0.5162 | -0.0055 | [-0.0182, +0.0063] |
| 300m-dagformer__context_both_gamma1.25_random1000_normmatched | openbookqa | 0.2760 | 0.2740 | -0.0020 | [-0.0100, +0.0040] |
| 300m-dagformer__context_both_gamma1.25_random1000_normmatched | sciq | 0.6980 | 0.6930 | -0.0050 | [-0.0120, +0.0010] |
| 300m-dagformer__context_both_gamma1.25_random1000_normmatched | boolq | 0.6080 | 0.6070 | -0.0009 | [-0.0040, +0.0021] |
| 300m-dagformer__context_both_gamma1.25_random1000_normmatched | mathqa | 0.2278 | 0.2295 | +0.0017 | [-0.0013, +0.0047] |
| 300m-dagformer__context_both_gamma1.25_random1000_normmatched | commonsense_qa | 0.1957 | 0.1957 | +0.0000 | [+0.0000, +0.0000] |
| 300m-dagformer__context_both_gamma1.25_random1000_normmatched | social_iqa | 0.3628 | 0.3634 | +0.0005 | [-0.0031, +0.0041] |
| 300m-dagformer__context_both_gamma1.25_random1000_normmatched | gsm8k_bpb | 1.3314 | 1.3319 | -0.0005 | [-0.0007, -0.0004] |
