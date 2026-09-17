# Circuit for `domain_code` — channel `corr`

Source: `/tmp/interp_gpu2/contrast_domain_code.pt`  
Selection: `{"rule": "null", "q": 0.005, "s_cutoff": 5.1315412039172745, "expected_false_positives": 16.17, "n_null_draws_est": 6, "n_null_draws_cal": 6}`, hyperconnections only

## Is there a behaviour to explain?

Instruction gap (pos - neg) = **+3.3328** (paired t = +16.02 over 16 items) — USABLE

## Does the instruction move the routing weights at all?

| quantity | value |
|---|---|
| mean |alpha| over eligible edges | 0.5060 |
| instruction effect (mean |Delta|) | 0.12718 (25.14% of |alpha|) |
| content effect (sd across items) | 0.10378 (20.51% of |alpha|) |
| paraphrase effect (sd within a pool) | 0.04999 (9.88% of |alpha|) |
| instruction / content | 1.225 |
| instruction / paraphrase | 2.544 |

This is the ceiling on everything below. If the instruction moves `alpha` by a fraction of a percent while the content moves it by several percent, then the routing map is content-driven and the behavioural contrast is a small perturbation riding on it — a circuit selected here can be perfectly real and still have nothing to intervene on.

## Sparsity of the difference matrix

| quantity | value |
|---|---|
| edges total | 3773 |
| edges selected | 829 (21.97%) |
| selected by the matched null (mean) | 6.3 |
| signal / null selection ratio | 130.89x |
| edges holding 50% of |Delta| mass | 421 |
| edges holding 90% of |Delta| mass | 2023 |
| Gini of |Delta| | 0.607 |
| max |Delta| | 5.9363 |
| mean |Delta| | 0.11056 |

## Does it replicate on held-out items?

| quantity | value |
|---|---|
| sign agreement on selected edges | 1.000 |
| sign agreement on random edges | 0.953 |
| corr(Delta_train, Delta_test) on selected | 0.995 |

## Circuit shape

| quantity | value |
|---|---|
| edges | 829 |
| of which hyperconnections (src < layer) | 829 |
| distinct target heads | 176 |
| distinct target layers | 11 |
| connected components | 1 |
| largest component (nodes) | 187 |
| by stream | {"v": 245, "k": 299, "q": 269, "r": 16} |
| by target layer | {"1": 19, "2": 40, "3": 34, "4": 62, "5": 55, "6": 111, "7": 137, "8": 54, "9": 72, "10": 92, "11": 153} |
| by source | {"0": 178, "1": 129, "2": 108, "3": 106, "4": 65, "5": 77, "6": 62, "7": 44, "8": 32, "9": 20, "10": 8} |

## Top edges

| edge | Delta | t | kind |
|---|---|---|---|
| v:emb->L9/h2 | +4.2615 | +7.74 | hyper |
| v:emb->L9/h6 | +3.9174 | +8.78 | hyper |
| v:emb->L6/h11 | +2.5767 | +11.59 | hyper |
| k:emb->L7/h1 | -1.9835 | -9.75 | hyper |
| v:emb->L8/h3 | +1.5684 | +8.46 | hyper |
| v:emb->L9/h14 | -1.3215 | -15.53 | hyper |
| k:emb->L9/h5 | +1.2996 | +14.14 | hyper |
| v:emb->L8/h8 | +1.2704 | +10.81 | hyper |
| k:emb->L7/h10 | +1.2588 | +5.59 | hyper |
| v:emb->L9/h0 | -1.2167 | -15.98 | hyper |
| v:emb->L9/h13 | +1.1752 | +9.20 | hyper |
| k:emb->L7/h15 | +1.1343 | +14.94 | hyper |
| v:emb->L9/h7 | -1.1005 | -19.99 | hyper |
| v:emb->L6/h13 | -1.0183 | -10.44 | hyper |
| k:emb->L9/h6 | -0.9995 | -19.96 | hyper |
| v:emb->L9/h4 | +0.9849 | +4.26 | hyper |
| k:emb->L9/h7 | +0.9475 | +13.47 | hyper |
| v:emb->L9/h1 | -0.9359 | -13.87 | hyper |
| k:emb->L9/h2 | -0.9082 | -22.57 | hyper |
| q:emb->L7/h1 | -0.8919 | -9.09 | hyper |
| v:emb->L11/h9 | +0.8273 | +7.48 | hyper |
| v:emb->L9/h9 | -0.8271 | -14.76 | hyper |
| v:emb->L6/h15 | +0.8239 | +7.84 | hyper |
| v:emb->L11/h5 | +0.8062 | +5.34 | hyper |
| v:emb->L11/h12 | +0.8024 | +4.56 | hyper |

Positive Delta means the connection is weighted *more* under the `pos` instruction than under `neg`.

## Reading the selection

The rule picked 829 edges from the real contrast and 6.3 on average from balanced-split nulls that contain no behavioural signal — a ratio of **130.89x**. The circuit is selected well above what paraphrase noise produces.

Held-out sign agreement is 1.000 on selected edges against 0.953 on random ones. When those two are equal, Delta is essentially the same for every item — the routing difference is a function of the instruction alone, so per-item significance testing has no power to separate edges and `--select null` is the rule to use.
