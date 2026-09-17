# Circuit for `honesty` — channel `corr`

Source: `/tmp/interp_gpu/contrast_honesty.pt`  
Selection: `{"rule": "null", "q": 0.005, "s_cutoff": 2.253045778329588, "expected_false_positives": 16.17, "n_null_draws_est": 6, "n_null_draws_cal": 6}`, hyperconnections only

## Is there a behaviour to explain?

Instruction gap (pos - neg) = **-0.0238** (paired t = -1.33 over 52 items) — WEAK — the instruction barely moves the behaviour; treat any circuit below as a circuit for reading the instruction

## Does the instruction move the routing weights at all?

| quantity | value |
|---|---|
| mean |alpha| over eligible edges | 0.3895 |
| instruction effect (mean |Delta|) | 0.00891 (2.29% of |alpha|) |
| content effect (sd across items) | 0.04653 (11.94% of |alpha|) |
| paraphrase effect (sd within a pool) | 0.02111 (5.42% of |alpha|) |
| instruction / content | 0.191 |
| instruction / paraphrase | 0.422 |

This is the ceiling on everything below. If the instruction moves `alpha` by a fraction of a percent while the content moves it by several percent, then the routing map is content-driven and the behavioural contrast is a small perturbation riding on it — a circuit selected here can be perfectly real and still have nothing to intervene on.

## Sparsity of the difference matrix

| quantity | value |
|---|---|
| edges total | 3773 |
| edges selected | 13 (0.34%) |
| selected by the matched null (mean) | 80.0 |
| signal / null selection ratio | 0.16x |
| edges holding 50% of |Delta| mass | 494 |
| edges holding 90% of |Delta| mass | 2091 |
| Gini of |Delta| | 0.578 |
| max |Delta| | 0.2839 |
| mean |Delta| | 0.00813 |

## Does it replicate on held-out items?

| quantity | value |
|---|---|
| sign agreement on selected edges | 1.000 |
| sign agreement on random edges | 1.000 |
| corr(Delta_train, Delta_test) on selected | 0.999 |

## Circuit shape

| quantity | value |
|---|---|
| edges | 13 |
| of which hyperconnections (src < layer) | 13 |
| distinct target heads | 12 |
| distinct target layers | 6 |
| connected components | 5 |
| largest component (nodes) | 7 |
| by stream | {"k": 2, "r": 1, "q": 8, "v": 2} |
| by target layer | {"4": 1, "6": 3, "7": 3, "8": 3, "10": 2, "11": 1} |
| by source | {"0": 2, "1": 4, "2": 2, "5": 3, "6": 1, "8": 1} |

## Top edges

| edge | Delta | t | kind |
|---|---|---|---|
| k:L0->L10/h2 | +0.0248 | +33.75 | hyper |
| r:emb->L6/r | -0.0206 | -19.30 | hyper |
| q:L7->L11/h8 | +0.0183 | +19.41 | hyper |
| q:L0->L7/h5 | +0.0128 | +17.78 | hyper |
| q:L4->L6/h1 | +0.0127 | +42.40 | hyper |
| q:L1->L7/h14 | -0.0122 | -34.38 | hyper |
| q:emb->L4/h12 | -0.0115 | -15.37 | hyper |
| v:L0->L10/h5 | -0.0112 | -28.26 | hyper |
| q:L5->L8/h7 | -0.0100 | -24.83 | hyper |
| q:L4->L6/h8 | +0.0077 | +29.53 | hyper |
| q:L1->L8/h13 | +0.0075 | +30.72 | hyper |
| v:L4->L8/h6 | -0.0071 | -55.50 | hyper |
| k:L0->L7/h14 | -0.0067 | -10.98 | hyper |

Positive Delta means the connection is weighted *more* under the `pos` instruction than under `neg`.

## Reading the selection

The rule picked 13 edges from the real contrast and 80.0 on average from balanced-split nulls that contain no behavioural signal — a ratio of **0.16x**. A ratio near 1 means the selection is not distinguishable from paraphrase noise, whatever the p-values say.

Held-out sign agreement is 1.000 on selected edges against 1.000 on random ones. When those two are equal, Delta is essentially the same for every item — the routing difference is a function of the instruction alone, so per-item significance testing has no power to separate edges and `--select null` is the rule to use.
