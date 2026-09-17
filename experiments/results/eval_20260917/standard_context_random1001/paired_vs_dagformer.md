# Paired comparison against dagformer

Reference results: `experiments/results/eval_20260917/standard_matched`.

Positive differences favor the variant. Intervals resample documents, not training seeds. BPB decreases and accuracy increases are positive.

| Variant | Task | Reference | Variant | Difference | Paired 95% interval |
|---|---|---:|---:|---:|---|
| 300m-dagformer__context_both_gamma1.25_random1001_normmatched | wikitext | 1.0225 | 1.0227 | -0.0002 | [-0.0003, -0.0001] |
| 300m-dagformer__context_both_gamma1.25_random1001_normmatched | lambada_openai | 0.3018 | 0.3111 | +0.0093 | [+0.0056, +0.0130] |
| 300m-dagformer__context_both_gamma1.25_random1001_normmatched | hellaswag | 0.3145 | 0.3147 | +0.0002 | [-0.0012, +0.0017] |
| 300m-dagformer__context_both_gamma1.25_random1001_normmatched | piqa | 0.6436 | 0.6436 | +0.0000 | [-0.0038, +0.0038] |
| 300m-dagformer__context_both_gamma1.25_random1001_normmatched | arc_easy | 0.3847 | 0.3859 | +0.0013 | [-0.0013, +0.0042] |
| 300m-dagformer__context_both_gamma1.25_random1001_normmatched | arc_challenge | 0.2321 | 0.2355 | +0.0034 | [-0.0009, +0.0085] |
| 300m-dagformer__context_both_gamma1.25_random1001_normmatched | winogrande | 0.5217 | 0.5241 | +0.0024 | [-0.0079, +0.0126] |
| 300m-dagformer__context_both_gamma1.25_random1001_normmatched | openbookqa | 0.2760 | 0.2760 | +0.0000 | [-0.0080, +0.0080] |
| 300m-dagformer__context_both_gamma1.25_random1001_normmatched | sciq | 0.6980 | 0.6990 | +0.0010 | [-0.0030, +0.0050] |
| 300m-dagformer__context_both_gamma1.25_random1001_normmatched | boolq | 0.6080 | 0.6089 | +0.0009 | [-0.0018, +0.0037] |
| 300m-dagformer__context_both_gamma1.25_random1001_normmatched | mathqa | 0.2278 | 0.2298 | +0.0020 | [-0.0010, +0.0054] |
| 300m-dagformer__context_both_gamma1.25_random1001_normmatched | commonsense_qa | 0.1957 | 0.1957 | +0.0000 | [+0.0000, +0.0000] |
| 300m-dagformer__context_both_gamma1.25_random1001_normmatched | social_iqa | 0.3628 | 0.3639 | +0.0010 | [-0.0020, +0.0041] |
| 300m-dagformer__context_both_gamma1.25_random1001_normmatched | gsm8k_bpb | 1.3314 | 1.3301 | +0.0013 | [+0.0011, +0.0014] |
