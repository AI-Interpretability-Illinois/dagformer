# Explicit attribute questions: transfer is endpoint-dependent

This posthoc follow-up asks directly for the color, metal or animal. It uses
1,024 additional item keys (seed 20260920), the same step-9000 checkpoint,
fixed ten-edge selection and doses, two fixed norm-matched controls per
channel, greedy generation and a 16-token cap. It changes both the question
template and item set, so the difference from the earlier cohort is not a
paired estimate of question wording alone.

The primary endpoint remains whether the first mentioned candidate value
matches the fact. Predictor-only and both-channel edits reduce this rate:

| Edit | Gamma | Neutral target-first rate | Deceptive target-first rate | Deceptive change (points) [95% interval] |
|---|---:|---:|---:|---|
| Reference | 1 | 83.11% | 81.25% | reference |
| Predictor | 1.5 | 76.46% | 75.39% | -5.86 [-7.91, -3.91] |
| Correction | 1.25 | 82.32% | 81.54% | +0.29 [-1.37, +1.86] |
| Both | 1.25 | 77.83% | 78.22% | -3.03 [-5.08, -1.07] |

Intervals use the [paired prompt-cluster bootstrap](../prompt_clusters/README.md).
The neutral both-channel difference is -5.27 points [-7.39, -3.24]. The
predictor and both-channel edits also fall below each of the two fixed
controls on the primary endpoint. Correction-only differences from the
reference and controls have intervals containing zero.

The full [literal-mention diagnostics](context_generation_diagnostics.md)
show a tradeoff under the deceptive cue:

| Edit | Target first | Alternative first | No candidate value | Any alternative anywhere | Target mentioned with no alternative |
|---|---:|---:|---:|---:|---:|
| Reference | 81.25% | 7.52% | 11.23% | 23.73% | 65.04% |
| Predictor | 75.39% | 8.30% | 16.31% | 18.65% | 65.04% |
| Correction | 81.54% | 5.08% | 13.38% | 15.14% | 71.48% |
| Both | 78.22% | 5.37% | 16.41% | 13.87% | 69.73% |

The first three columns form a partition. Other columns are secondary
diagnostics added after inspecting the continuations. For correction-only,
target mention without any alternative rises by 6.45 points [4.30, 8.69];
for both channels it rises by 4.69 [1.95, 7.42]. These are literal constraints,
not semantic grading of negation, quoted claims or explanations of another
speaker's words. They do not replace the primary target-first endpoint.

[Examples selected by item index](context_generation_examples.md) show both
types of change: some iron-coin answers switch from copper to iron, while
some color questions lose their color and instead say where the object was.
Some reference continuations first give the correct attribute and then name
a different one. Lowering alternative mentions alone can therefore hide an
increase in omitted answers.

All 1,024 item keys are distinct from the original discovery keys. Keys
include the receiver, which neutral QA omits: neutral prompts have 830 unique
strings, while deceptive prompts have 1,024. Seventy-nine distinct factual
sentences (94 rows) appeared in the original discovery set with another
item key. This is transfer across item combinations and a new template,
not a fact-disjoint benchmark. Grouping repeated neutral prompts preserves
the negative predictor/both-channel result.
