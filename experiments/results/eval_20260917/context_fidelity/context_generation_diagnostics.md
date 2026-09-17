# Context generation: value inclusion and omission

Greedy continuations contain at most 16 new tokens. The endpoint asks whether
the first mentioned candidate value is the value stated in the fact. It
counts omission as a failure to include the target value. This is not a
semantic judgment that the answer is false: for an iron ring, saying
only ‘a ring’ omits the material while remaining compatible with the fact.

Candidate alternatives are scored by literal mention. The diagnostic also
checks whether an alternative appears anywhere after a correct first value;
it does not resolve negation, quotation or conversational intent.

| Arm | Condition | Target first | Alternative first | No candidate | Correct object named but value omitted | Any alternative anywhere |
|---|---|---:|---:|---:|---:|---:|
| reference | original/neutral | 47.95% | 0.00% | 52.05% | 51.66% | 0.00% |
| reference | original/deceptive | 45.90% | 0.00% | 54.10% | 53.71% | 0.00% |
| reference | qa/neutral | 90.23% | 0.00% | 9.77% | 9.77% | 0.00% |
| reference | qa/deceptive | 90.33% | 0.00% | 9.67% | 6.54% | 0.00% |
| reference | dialogue/neutral | 87.99% | 0.00% | 12.01% | 8.11% | 0.00% |
| reference | dialogue/deceptive | 87.01% | 0.00% | 12.99% | 8.01% | 0.00% |
| pred/circuit | original/neutral | 49.41% | 0.00% | 50.59% | 50.00% | 0.00% |
| pred/circuit | original/deceptive | 48.34% | 0.00% | 51.66% | 51.27% | 0.00% |
| pred/circuit | qa/neutral | 92.48% | 0.00% | 7.52% | 7.52% | 0.00% |
| pred/circuit | qa/deceptive | 90.53% | 0.00% | 9.47% | 5.57% | 0.00% |
| pred/circuit | dialogue/neutral | 89.94% | 0.00% | 10.06% | 5.96% | 0.00% |
| pred/circuit | dialogue/deceptive | 89.36% | 0.00% | 10.64% | 5.86% | 0.00% |
| pred/random0 | original/neutral | 47.95% | 0.00% | 52.05% | 51.66% | 0.00% |
| pred/random0 | original/deceptive | 46.09% | 0.00% | 53.91% | 53.42% | 0.00% |
| pred/random0 | qa/neutral | 89.84% | 0.00% | 10.16% | 10.16% | 0.00% |
| pred/random0 | qa/deceptive | 89.94% | 0.00% | 10.06% | 6.84% | 0.00% |
| pred/random0 | dialogue/neutral | 87.89% | 0.00% | 12.11% | 8.30% | 0.00% |
| pred/random0 | dialogue/deceptive | 86.91% | 0.00% | 13.09% | 8.20% | 0.00% |
| pred/random1 | original/neutral | 47.85% | 0.00% | 52.15% | 51.66% | 0.00% |
| pred/random1 | original/deceptive | 46.00% | 0.00% | 54.00% | 53.61% | 0.00% |
| pred/random1 | qa/neutral | 88.77% | 0.00% | 11.23% | 11.23% | 0.00% |
| pred/random1 | qa/deceptive | 88.96% | 0.00% | 11.04% | 8.59% | 0.00% |
| pred/random1 | dialogue/neutral | 86.91% | 0.00% | 13.09% | 9.86% | 0.00% |
| pred/random1 | dialogue/deceptive | 84.86% | 0.00% | 15.14% | 10.45% | 0.00% |
| corr/circuit | original/neutral | 49.51% | 0.00% | 50.49% | 50.00% | 0.00% |
| corr/circuit | original/deceptive | 47.66% | 0.00% | 52.34% | 52.25% | 0.00% |
| corr/circuit | qa/neutral | 91.80% | 0.00% | 8.20% | 8.11% | 0.00% |
| corr/circuit | qa/deceptive | 91.02% | 0.00% | 8.98% | 4.88% | 0.00% |
| corr/circuit | dialogue/neutral | 89.55% | 0.00% | 10.45% | 6.35% | 0.00% |
| corr/circuit | dialogue/deceptive | 89.26% | 0.00% | 10.74% | 5.76% | 0.00% |
| corr/random0 | original/neutral | 48.63% | 0.00% | 51.37% | 50.98% | 0.00% |
| corr/random0 | original/deceptive | 46.68% | 0.00% | 53.32% | 52.93% | 0.00% |
| corr/random0 | qa/neutral | 89.45% | 0.00% | 10.55% | 10.55% | 0.00% |
| corr/random0 | qa/deceptive | 89.65% | 0.00% | 10.35% | 8.20% | 0.00% |
| corr/random0 | dialogue/neutral | 87.50% | 0.00% | 12.50% | 8.98% | 0.00% |
| corr/random0 | dialogue/deceptive | 85.84% | 0.00% | 14.16% | 9.18% | 0.00% |
| corr/random1 | original/neutral | 47.85% | 0.00% | 52.15% | 51.76% | 0.00% |
| corr/random1 | original/deceptive | 46.29% | 0.00% | 53.71% | 53.32% | 0.00% |
| corr/random1 | qa/neutral | 88.09% | 0.00% | 11.91% | 11.91% | 0.00% |
| corr/random1 | qa/deceptive | 88.09% | 0.00% | 11.91% | 9.77% | 0.00% |
| corr/random1 | dialogue/neutral | 85.94% | 0.00% | 14.06% | 10.84% | 0.00% |
| corr/random1 | dialogue/deceptive | 84.38% | 0.00% | 15.62% | 11.13% | 0.00% |
| both/circuit | original/neutral | 50.10% | 0.00% | 49.90% | 49.41% | 0.00% |
| both/circuit | original/deceptive | 49.12% | 0.00% | 50.88% | 50.78% | 0.00% |
| both/circuit | qa/neutral | 92.38% | 0.00% | 7.62% | 7.62% | 0.00% |
| both/circuit | qa/deceptive | 91.11% | 0.00% | 8.89% | 4.30% | 0.00% |
| both/circuit | dialogue/neutral | 90.04% | 0.00% | 9.96% | 5.66% | 0.00% |
| both/circuit | dialogue/deceptive | 89.75% | 0.00% | 10.25% | 5.47% | 0.00% |
| both/random0 | original/neutral | 48.54% | 0.00% | 51.46% | 51.07% | 0.00% |
| both/random0 | original/deceptive | 47.27% | 0.00% | 52.73% | 52.34% | 0.00% |
| both/random0 | qa/neutral | 89.45% | 0.00% | 10.55% | 10.55% | 0.00% |
| both/random0 | qa/deceptive | 89.65% | 0.00% | 10.35% | 8.40% | 0.00% |
| both/random0 | dialogue/neutral | 87.79% | 0.00% | 12.21% | 8.98% | 0.00% |
| both/random0 | dialogue/deceptive | 86.13% | 0.00% | 13.87% | 8.89% | 0.00% |
| both/random1 | original/neutral | 47.95% | 0.00% | 52.05% | 51.76% | 0.00% |
| both/random1 | original/deceptive | 46.09% | 0.00% | 53.91% | 53.52% | 0.00% |
| both/random1 | qa/neutral | 87.50% | 0.00% | 12.50% | 12.50% | 0.00% |
| both/random1 | qa/deceptive | 87.30% | 0.00% | 12.70% | 10.74% | 0.00% |
| both/random1 | dialogue/neutral | 85.35% | 0.00% | 14.65% | 11.62% | 0.00% |
| both/random1 | dialogue/deceptive | 83.69% | 0.00% | 16.31% | 12.01% | 0.00% |

[Deterministically selected examples](context_generation_examples.md) retain the full short continuations.

Counts by category and all edit/control arms are retained in the companion JSON.
