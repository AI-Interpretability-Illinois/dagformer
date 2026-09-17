# Ordinary tasks with fixed random-head controls

The two controls use the same seeds as the context-generation experiment
(1000 and 1001), the same layer/stream/source slots and number of edited
coordinates, and locally match the named edit's norm on incoming activations.
All runs use the step-9000 300M DAGFormer checkpoint, both channels at gamma
1.25, batch size eight and all 14 ordinary endpoints.

| Edit | LAMBADA accuracy | Change from unedited (points) [95% interval] | WikiText BPB cost | GSM8K joint BPB cost |
|---|---:|---|---:|---:|
| Unedited | 30.18% | reference | 0 | 0 |
| Named ten edges | 32.51% | +2.33 [1.80, 2.85] | +0.004319 | +0.008475 |
| Random 1000 | 29.52% | -0.66 [-1.09, -0.23] | +0.000402 | +0.000510 |
| Random 1001 | 31.11% | +0.93 [0.56, 1.30] | +0.000200 | -0.001254 |

The named edit's direct paired LAMBADA advantage is **2.99 points [2.39, 3.59]**
over random 1000 and **1.40 points [0.87, 1.92]** over random 1001. One measured
random control also improves LAMBADA. Two controls do not estimate the full
distribution of possible random circuits.

The named edit has a substantially larger likelihood cost. Equal local edit
norms do not give equal language-model damage, and these comparisons do not
establish optimality at matched NLL cost. The
[answer-occurrence stratification](lambada_answer_occurrence.md) retains both
controls alongside the named edits: random 1001 improves answer-seen prompts
by 1.13 points, while the named edit improves them by 3.01 points.

All task results, including losses and uncertain differences, are retained in
the [random-1000 comparison](../standard_context_random1000/paired_vs_dagformer.md)
and [random-1001 comparison](../standard_context_random1001/paired_vs_dagformer.md).
The direct [named-minus-1000](../standard_context_both/paired_vs_random1000.md)
and [named-minus-1001](../standard_context_both/paired_vs_random1001.md) tables
use the same saved documents. Intervals are unadjusted document bootstraps
for these fixed checkpoint/intervention settings.
