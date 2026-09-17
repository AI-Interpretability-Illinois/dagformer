# Circuit for `honesty` — channel `pred`

Source: `/tmp/interp_gpu/contrast_honesty.pt`  
Selection: `{"rule": "null", "q": 0.005, "s_cutoff": 1.6616817300514293, "expected_false_positives": 16.17, "n_null_draws_est": 6, "n_null_draws_cal": 6}`, hyperconnections only

## Is there a behaviour to explain?

Instruction gap (pos - neg) = **-0.0238** (paired t = -1.33 over 52 items) — WEAK — the instruction barely moves the behaviour; treat any circuit below as a circuit for reading the instruction

## Does the instruction move the routing weights at all?

| quantity | value |
|---|---|
| mean |alpha| over eligible edges | 0.3766 |
| instruction effect (mean |Delta|) | 0.00168 (0.45% of |alpha|) |
| content effect (sd across items) | 0.00207 (0.55% of |alpha|) |
| paraphrase effect (sd within a pool) | 0.00379 (1.01% of |alpha|) |
| instruction / content | 0.811 |
| instruction / paraphrase | 0.444 |

This is the ceiling on everything below. If the instruction moves `alpha` by a fraction of a percent while the content moves it by several percent, then the routing map is content-driven and the behavioural contrast is a small perturbation riding on it — a circuit selected here can be perfectly real and still have nothing to intervene on.

## Sparsity of the difference matrix

| quantity | value |
|---|---|
| edges total | 3773 |
| edges selected | 2 (0.05%) |
| selected by the matched null (mean) | 5.0 |
| signal / null selection ratio | 0.40x |
| edges holding 50% of |Delta| mass | 610 |
| edges holding 90% of |Delta| mass | 2169 |
| Gini of |Delta| | 0.535 |
| max |Delta| | 0.0359 |
| mean |Delta| | 0.00169 |

## Does it replicate on held-out items?

| quantity | value |
|---|---|
| sign agreement on selected edges | 1.000 |
| sign agreement on random edges | 1.000 |
| corr(Delta_train, Delta_test) on selected | 1.000 |

## Circuit shape

| quantity | value |
|---|---|
| edges | 2 |
| of which hyperconnections (src < layer) | 2 |
| distinct target heads | 2 |
| distinct target layers | 2 |
| connected components | 2 |
| largest component (nodes) | 2 |
| by stream | {"k": 1, "q": 1} |
| by target layer | {"7": 1, "8": 1} |
| by source | {"0": 1, "5": 1} |

## Top edges

| edge | Delta | t | kind |
|---|---|---|---|
| k:emb->L7/h2 | -0.0017 | -43.18 | hyper |
| q:L4->L8/h1 | +0.0014 | +85.82 | hyper |

Positive Delta means the connection is weighted *more* under the `pos` instruction than under `neg`.

## Reading the selection

The rule picked 2 edges from the real contrast and 5.0 on average from balanced-split nulls that contain no behavioural signal — a ratio of **0.40x**. A ratio near 1 means the selection is not distinguishable from paraphrase noise, whatever the p-values say.

Held-out sign agreement is 1.000 on selected edges against 1.000 on random ones. When those two are equal, Delta is essentially the same for every item — the routing difference is a function of the instruction alone, so per-item significance testing has no power to separate edges and `--select null` is the rule to use.
