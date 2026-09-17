# Paired evaluation — primary metrics

Positive differences favor DAGFormer. Intervals resample the same documents
for both models; they do not measure variation across training seeds.

| Size | Task | Baseline | DAGFormer | Difference | Paired 95% CI |
|---|---|---:|---:|---:|---|
| 75m | gsm8k_answer_bpb | 2.2130 | 2.1047 | +0.1083 | [+0.1043, +0.1123] |
| 75m | gsm8k_question_bpb | 1.4113 | 1.3641 | +0.0472 | [+0.0452, +0.0493] |
| 150m | gsm8k_answer_bpb | 1.9205 | 1.8223 | +0.0982 | [+0.0945, +0.1021] |
| 150m | gsm8k_question_bpb | 1.2246 | 1.1752 | +0.0494 | [+0.0474, +0.0514] |
| 300m | gsm8k_answer_bpb | 1.6870 | 1.5684 | +0.1186 | [+0.1150, +0.1222] |
| 300m | gsm8k_question_bpb | 1.0999 | 1.0591 | +0.0408 | [+0.0389, +0.0427] |
