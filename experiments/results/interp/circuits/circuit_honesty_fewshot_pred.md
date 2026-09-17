# Circuit for `honesty_fewshot` — channel `pred`

Source: `/tmp/interp_fs4/contrast_honesty_fewshot.pt`  
Selection: `{"rule": "null", "q": 0.005, "s_cutoff": 1.901371479997673, "expected_false_positives": 16.17, "n_null_draws_est": 6, "n_null_draws_cal": 6}`, hyperconnections only

## Is there a behaviour to explain?

Instruction gap (pos - neg) = **+0.0130** (paired t = +0.83 over 52 items) — WEAK — the instruction barely moves the behaviour; treat any circuit below as a circuit for reading the instruction

## Does the instruction move the routing weights at all?

| quantity | value |
|---|---|
| mean |alpha| over eligible edges | 0.3784 |
| instruction effect (mean |Delta|) | 0.00069 (0.18% of |alpha|) |
| content effect (sd across items) | 0.00201 (0.53% of |alpha|) |
| paraphrase effect (sd within a pool) | 0.00252 (0.67% of |alpha|) |
| instruction / content | 0.344 |
| instruction / paraphrase | 0.274 |

This is the ceiling on everything below. If the instruction moves `alpha` by a fraction of a percent while the content moves it by several percent, then the routing map is content-driven and the behavioural contrast is a small perturbation riding on it — a circuit selected here can be perfectly real and still have nothing to intervene on.

## Sparsity of the difference matrix

| quantity | value |
|---|---|
| edges total | 3773 |
| edges selected | 0 (0.00%) |
| selected by the matched null (mean) | 7.5 |
| signal / null selection ratio | 0.00x |
| edges holding 50% of |Delta| mass | 598 |
| edges holding 90% of |Delta| mass | 2160 |
| Gini of |Delta| | 0.542 |
| max |Delta| | 0.0158 |
| mean |Delta| | 0.00070 |

## Does it replicate on held-out items?

| quantity | value |
|---|---|
| sign agreement on selected edges | nan |
| sign agreement on random edges | nan |
| corr(Delta_train, Delta_test) on selected | nan |

## Circuit shape

| quantity | value |
|---|---|
| edges | 0 |
| of which hyperconnections (src < layer) | 0 |
| distinct target heads | 0 |
| distinct target layers | 0 |
| connected components | 0 |
| largest component (nodes) | 0 |
| by stream | {} |
| by target layer | {} |
| by source | {} |

## Top edges

| edge | Delta | t | kind |
|---|---|---|---|

Positive Delta means the connection is weighted *more* under the `pos` instruction than under `neg`.

## Reading the selection

The rule picked 0 edges from the real contrast and 7.5 on average from balanced-split nulls that contain no behavioural signal — a ratio of **0.00x**. A ratio near 1 means the selection is not distinguishable from paraphrase noise, whatever the p-values say.

Held-out sign agreement is nan on selected edges against nan on random ones. When those two are equal, Delta is essentially the same for every item — the routing difference is a function of the instruction alone, so per-item significance testing has no power to separate edges and `--select null` is the rule to use.
