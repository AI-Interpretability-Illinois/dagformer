# Paired comparison against dagformer

Reference results: `experiments/results/eval_20260917/gsm8k_full`.

Positive differences favor the variant. Intervals resample documents, not training seeds. BPB decreases and accuracy increases are positive.

| Variant | Task | Reference | Variant | Difference | Paired 95% interval |
|---|---|---:|---:|---:|---|
| 300m-dagformer | gsm8k | 0.0152 | 0.0197 | +0.0045 | [-0.0023, +0.0121] |
