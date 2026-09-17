# Circuit for `domain_code` — channel `pred`

Source: `/tmp/interp_final/contrast_domain_code.pt`  
Selection: `{"rule": "null", "q": 0.005, "s_cutoff": 1.868829515971337, "expected_false_positives": 16.17, "n_null_draws_est": 8, "n_null_draws_cal": 8}`, hyperconnections only

## Is there a behaviour to explain?

Instruction gap (pos - neg) = **+3.3322** (paired t = +15.99 over 16 items) — USABLE

## Does the instruction move the routing weights at all?

| quantity | value |
|---|---|
| mean |alpha| over eligible edges | 0.3790 |
| instruction effect (mean |Delta|) | 0.00311 (0.82% of |alpha|) |
| content effect (sd across items) | 0.00285 (0.75% of |alpha|) |
| paraphrase effect (sd within a pool) | 0.00450 (1.19% of |alpha|) |
| instruction / content | 1.092 |
| instruction / paraphrase | 0.691 |

This is the ceiling on everything below. If the instruction moves `alpha` by a fraction of a percent while the content moves it by several percent, then the routing map is content-driven and the behavioural contrast is a small perturbation riding on it — a circuit selected here can be perfectly real and still have nothing to intervene on.

## Sparsity of the difference matrix

| quantity | value |
|---|---|
| edges total | 3773 |
| edges selected | 513 (13.60%) |
| selected by the matched null (mean) | 42.6 |
| signal / null selection ratio | 12.04x |
| edges holding 50% of |Delta| mass | 577 |
| edges holding 90% of |Delta| mass | 2162 |
| Gini of |Delta| | 0.546 |
| max |Delta| | 0.0738 |
| mean |Delta| | 0.00304 |

## Does it replicate on held-out items?

| quantity | value |
|---|---|
| sign agreement on selected edges | 1.000 |
| sign agreement on random edges | 0.973 |
| corr(Delta_train, Delta_test) on selected | 1.000 |

## Circuit shape

| quantity | value |
|---|---|
| edges | 513 |
| of which hyperconnections (src < layer) | 513 |
| distinct target heads | 160 |
| distinct target layers | 11 |
| connected components | 1 |
| largest component (nodes) | 171 |
| by stream | {"v": 208, "k": 159, "q": 144, "r": 2} |
| by target layer | {"1": 9, "2": 19, "3": 25, "4": 27, "5": 39, "6": 49, "7": 70, "8": 70, "9": 60, "10": 64, "11": 81} |
| by source | {"0": 93, "1": 77, "2": 63, "3": 64, "4": 62, "5": 51, "6": 36, "7": 29, "8": 24, "9": 8, "10": 6} |

## Top edges

| edge | Delta | t | kind |
|---|---|---|---|
| v:emb->L9/h6 | +0.0659 | +30.67 | hyper |
| v:emb->L8/h5 | +0.0510 | +30.95 | hyper |
| v:emb->L7/h8 | +0.0400 | +36.83 | hyper |
| v:emb->L8/h11 | +0.0350 | +29.95 | hyper |
| v:emb->L1/h5 | -0.0349 | -23.66 | hyper |
| v:emb->L2/h2 | -0.0299 | -29.72 | hyper |
| v:emb->L10/h6 | +0.0270 | +31.93 | hyper |
| v:emb->L5/h6 | -0.0265 | -29.60 | hyper |
| v:emb->L3/h9 | -0.0263 | -26.34 | hyper |
| v:emb->L7/h5 | -0.0262 | -32.85 | hyper |
| v:emb->L10/h11 | +0.0259 | +30.15 | hyper |
| v:emb->L9/h8 | +0.0254 | +35.47 | hyper |
| v:emb->L5/h3 | +0.0234 | +35.54 | hyper |
| k:emb->L2/h11 | -0.0226 | -28.05 | hyper |
| v:emb->L5/h13 | -0.0223 | -29.91 | hyper |
| q:emb->L2/h11 | -0.0219 | -26.50 | hyper |
| k:L0->L8/h2 | -0.0212 | -31.64 | hyper |
| k:emb->L2/h0 | -0.0193 | -26.75 | hyper |
| v:emb->L1/h4 | -0.0183 | -15.88 | hyper |
| q:emb->L2/h9 | +0.0182 | +30.55 | hyper |
| v:emb->L2/h13 | -0.0176 | -30.30 | hyper |
| v:emb->L8/h10 | +0.0174 | +40.99 | hyper |
| v:emb->L9/h11 | +0.0166 | +33.13 | hyper |
| v:emb->L1/h10 | -0.0166 | -18.47 | hyper |
| v:emb->L2/h10 | -0.0159 | -24.25 | hyper |

Positive Delta means the connection is weighted *more* under the `pos` instruction than under `neg`.

## Reading the selection

The rule picked 513 edges from the real contrast and 42.6 on average from balanced-split nulls that contain no behavioural signal — a ratio of **12.04x**. The circuit is selected well above what paraphrase noise produces.

Held-out sign agreement is 1.000 on selected edges against 0.973 on random ones. When those two are equal, Delta is essentially the same for every item — the routing difference is a function of the instruction alone, so per-item significance testing has no power to separate edges and `--select null` is the rule to use.
