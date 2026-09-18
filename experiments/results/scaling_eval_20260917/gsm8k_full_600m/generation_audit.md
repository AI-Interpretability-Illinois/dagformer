# GSM8K generation audit

Scores below retain the upstream extraction rules. Flexible extraction
uses the last matched number; it does not validate the derivation or answer units.

| Model | Documents | Strict match | Flexible match | Best constant answer | Mean repeated 4-gram fraction |
|---|---:|---:|---:|---|---:|
| 600m-baseline | 1319 | 0.30% | 2.35% | 5: 3.03% | 0.754 |
| 600m-dagformer | 1319 | 0.76% | 1.59% | 5: 3.03% | 0.410 |

The constant-answer diagnostic uses gold frequencies from the same evaluated documents.
Repetition is a mechanical text statistic, not a correctness judgment.
[Deterministically selected examples](generation_examples.md) show the continuations behind these metrics.
