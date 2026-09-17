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
| reference | attribute_qa/neutral | 83.11% | 2.73% | 14.16% | 13.57% | 2.73% |
| reference | attribute_qa/deceptive | 81.25% | 7.52% | 11.23% | 11.13% | 23.73% |
| pred/circuit | attribute_qa/neutral | 76.46% | 3.03% | 20.51% | 20.31% | 3.03% |
| pred/circuit | attribute_qa/deceptive | 75.39% | 8.30% | 16.31% | 16.31% | 18.65% |
| pred/random0 | attribute_qa/neutral | 83.01% | 3.22% | 13.77% | 13.28% | 3.22% |
| pred/random0 | attribute_qa/deceptive | 80.47% | 7.81% | 11.72% | 11.23% | 24.02% |
| pred/random1 | attribute_qa/neutral | 82.71% | 2.93% | 14.36% | 13.96% | 2.93% |
| pred/random1 | attribute_qa/deceptive | 79.98% | 7.71% | 12.30% | 12.01% | 23.54% |
| corr/circuit | attribute_qa/neutral | 82.32% | 1.56% | 16.11% | 15.43% | 1.56% |
| corr/circuit | attribute_qa/deceptive | 81.54% | 5.08% | 13.38% | 13.38% | 15.14% |
| corr/random0 | attribute_qa/neutral | 83.30% | 3.42% | 13.28% | 12.79% | 3.42% |
| corr/random0 | attribute_qa/deceptive | 81.15% | 7.91% | 10.94% | 10.94% | 22.85% |
| corr/random1 | attribute_qa/neutral | 83.50% | 2.34% | 14.16% | 13.28% | 2.34% |
| corr/random1 | attribute_qa/deceptive | 81.45% | 7.13% | 11.43% | 10.94% | 21.97% |
| both/circuit | attribute_qa/neutral | 77.83% | 1.86% | 20.31% | 19.92% | 1.86% |
| both/circuit | attribute_qa/deceptive | 78.22% | 5.37% | 16.41% | 16.21% | 13.87% |
| both/random0 | attribute_qa/neutral | 83.79% | 3.22% | 12.99% | 12.50% | 3.22% |
| both/random0 | attribute_qa/deceptive | 80.66% | 8.30% | 11.04% | 10.94% | 23.63% |
| both/random1 | attribute_qa/neutral | 83.40% | 2.54% | 14.06% | 13.38% | 2.54% |
| both/random1 | attribute_qa/deceptive | 80.76% | 7.32% | 11.91% | 11.52% | 21.68% |

[Deterministically selected examples](context_generation_examples.md) retain the full short continuations.

Counts by category and all edit/control arms are retained in the companion JSON.
