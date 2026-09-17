# Conditional GSM8K answer likelihood

All 1,319 test items are evaluated. The answer task conditions on the question
and scores only the gold solution suffix, including its leading space and final
answer marker. It is teacher-forced likelihood, not generated solution accuracy.

| Scale | Question BPB: baseline → DAGFormer | Answer BPB: baseline → DAGFormer | Answer BPB reduction [paired 95% interval] |
|---|---|---|---|
| 75m | 1.4113 → 1.3641 | 2.2130 → 2.1047 | 0.1083 [0.1043, 0.1123] |
| 150m | 1.2246 → 1.1752 | 1.9205 → 1.8223 | 0.0982 [0.0945, 0.1021] |
| 300m | 1.0999 → 1.0591 | 1.6870 → 1.5684 | 0.1186 [0.1150, 0.1222] |

All three pairs improve conditional gold-answer likelihood as well as question
likelihood. Thus the joint-BPB gain is not confined to modeling the question.
The full generation results separately test whether the model produces correct
answers without being given the earlier gold solution tokens.

The 600M baseline is an unpaired reference: question BPB 1.0142 and conditional
answer BPB 1.4591. It differs in both model size and training budget.

The [paired report](paired_summary.md) includes both endpoints. Intervals resample
documents and are unadjusted. BPB levels across question and answer distributions
are not directly comparable as task difficulty or reasoning scores.

These are separate harness tasks: the question prefix uses rolling likelihood
with the harness prefix token, while the answer task uses the question as context
without an added BOS token. The two values are not an exact additive decomposition
of the joint task. All question/answer texts fit within 418 tokenizer tokens,
so these likelihood runs do not truncate them.
