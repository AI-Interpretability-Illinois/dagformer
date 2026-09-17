# Paired comparison against dagformer

Positive differences favor the variant. Intervals resample documents, not training seeds. BPB decreases and accuracy increases are positive.

| Variant | Task | Reference | Variant | Difference | Paired 95% interval |
|---|---|---:|---:|---:|---|
| 300m-dagformer | wikitext | 1.0225 | 1.0183 | +0.0042 | [+0.0036, +0.0048] |
| 300m-dagformer | lambada_openai | 0.3018 | 0.3056 | +0.0039 | [-0.0023, +0.0101] |
| 300m-dagformer | hellaswag | 0.3145 | 0.3132 | -0.0013 | [-0.0044, +0.0018] |
| 300m-dagformer | piqa | 0.6436 | 0.6415 | -0.0022 | [-0.0103, +0.0060] |
| 300m-dagformer | arc_easy | 0.3847 | 0.3902 | +0.0055 | [-0.0013, +0.0122] |
| 300m-dagformer | arc_challenge | 0.2321 | 0.2398 | +0.0077 | [-0.0017, +0.0179] |
| 300m-dagformer | winogrande | 0.5217 | 0.5154 | -0.0063 | [-0.0221, +0.0103] |
| 300m-dagformer | openbookqa | 0.2760 | 0.2780 | +0.0020 | [-0.0100, +0.0140] |
| 300m-dagformer | sciq | 0.6980 | 0.7050 | +0.0070 | [-0.0020, +0.0160] |
| 300m-dagformer | boolq | 0.6080 | 0.6092 | +0.0012 | [-0.0043, +0.0067] |
| 300m-dagformer | mathqa | 0.2278 | 0.2302 | +0.0023 | [-0.0044, +0.0087] |
| 300m-dagformer | commonsense_qa | 0.1957 | 0.1957 | +0.0000 | [+0.0000, +0.0000] |
| 300m-dagformer | social_iqa | 0.3628 | 0.3613 | -0.0015 | [-0.0082, +0.0051] |
| 300m-dagformer | gsm8k_bpb | 1.3314 | 1.3319 | -0.0006 | [-0.0011, -0.0000] |
