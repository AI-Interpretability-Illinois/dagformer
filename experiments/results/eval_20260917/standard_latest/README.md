# 300M DAGFormer: step 9000 to step 10500

The reference column in the [paired table](paired_vs_dagformer.md) is step
9000; the variant is step 10500. These checkpoints contain 9,001 and 10,501
optimizer updates, or 4.719B and 5.506B processed tokens. This comparison
measures continued training of the same model.

WikiText BPB improves from 1.02248 to 1.01830. LAMBADA rises from 30.18% to
30.56%, with its paired interval including zero. All eleven other accuracy
endpoints also have intervals including zero. GSM8K joint question/answer
BPB changes slightly in the worse direction, from 1.33139 to 1.33195.

The full table retains all 14 endpoints. Intervals reflect document sampling
for these fixed checkpoints and are not adjusted for multiple comparisons.
The later checkpoint's full GSM8K generation is being evaluated separately.
