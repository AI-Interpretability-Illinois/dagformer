# Circuit for `domain_code` — channel `corr`

Source: `experiments/results/eval_20260917/domain_code/contrast_domain_code.pt`  
Selection: `{"rule": "null", "q": 0.005, "s_cutoff": 2.671120999332342, "expected_false_positives": 16.17, "n_null_draws_est": 8, "n_null_draws_cal": 8}`, hyperconnections only

## Is there a behaviour to explain?

Instruction gap (pos - neg) = **+3.3342** (paired t = +15.98 over 16 items) — USABLE

## Does the instruction move the routing weights at all?

| quantity | value |
|---|---|
| mean |alpha| over eligible edges | 0.5060 |
| instruction effect (mean |Delta|) | 0.12712 (25.12% of |alpha|) |
| content effect (sd across items) | 0.10379 (20.51% of |alpha|) |
| paraphrase effect (sd within a pool) | 0.05003 (9.89% of |alpha|) |
| instruction / content | 1.225 |
| instruction / paraphrase | 2.541 |

This is the ceiling on everything below. If the instruction moves `alpha` by a fraction of a percent while the content moves it by several percent, then the routing map is content-driven and the behavioural contrast is a small perturbation riding on it — a circuit selected here can be perfectly real and still have nothing to intervene on.

## Sparsity of the difference matrix

| quantity | value |
|---|---|
| edges total | 3773 |
| edges selected | 1598 (42.35%) |
| selected by the matched null (mean) | 20.8 |
| signal / null selection ratio | 77.01x |
| edges holding 50% of |Delta| mass | 422 |
| edges holding 90% of |Delta| mass | 2024 |
| Gini of |Delta| | 0.607 |
| max |Delta| | 5.9202 |
| mean |Delta| | 0.11049 |

## Does it replicate on held-out items?

| quantity | value |
|---|---|
| sign agreement on selected edges | 1.000 |
| sign agreement on random edges | 0.954 |
| corr(Delta_train, Delta_test) on selected | 0.982 |

## Circuit shape

| quantity | value |
|---|---|
| edges | 1598 |
| of which hyperconnections (src < layer) | 1598 |
| distinct target heads | 170 |
| distinct target layers | 11 |
| connected components | 1 |
| largest component (nodes) | 181 |
| by stream | {"v": 500, "k": 588, "q": 481, "r": 29} |
| by target layer | {"1": 1, "2": 58, "3": 66, "4": 106, "5": 122, "6": 176, "7": 247, "8": 160, "9": 210, "10": 205, "11": 247} |
| by source | {"0": 283, "1": 238, "2": 219, "3": 209, "4": 162, "5": 152, "6": 124, "7": 85, "8": 76, "9": 33, "10": 17} |

## Top edges

| edge | Delta | t | kind |
|---|---|---|---|
| v:emb->L7/h1 | +7.0796 | +4.43 | hyper |
| v:emb->L7/h7 | +6.6115 | +4.21 | hyper |
| v:emb->L7/h9 | -5.4832 | -4.00 | hyper |
| v:emb->L7/h14 | -5.4304 | -4.04 | hyper |
| v:emb->L7/h11 | +4.5682 | +3.57 | hyper |
| v:emb->L9/h2 | +4.2565 | +7.72 | hyper |
| v:emb->L9/h6 | +3.9130 | +8.76 | hyper |
| v:emb->L6/h11 | +2.5766 | +11.52 | hyper |
| v:emb->L7/h8 | +2.2361 | +3.77 | hyper |
| v:emb->L7/h3 | -2.2240 | -3.49 | hyper |
| k:emb->L7/h12 | +2.2217 | +5.72 | hyper |
| k:emb->L7/h1 | -1.9837 | -9.77 | hyper |
| k:emb->L7/h6 | +1.9718 | +4.45 | hyper |
| k:emb->L7/h0 | +1.9207 | +4.80 | hyper |
| v:emb->L8/h5 | +1.8702 | +8.71 | hyper |
| v:emb->L8/h11 | +1.6442 | +9.56 | hyper |
| v:emb->L7/h2 | -1.5683 | -3.20 | hyper |
| v:emb->L8/h3 | +1.5654 | +8.45 | hyper |
| v:emb->L7/h5 | -1.4173 | -2.90 | hyper |
| k:emb->L7/h2 | +1.4121 | +4.44 | hyper |
| v:emb->L9/h14 | -1.3221 | -15.50 | hyper |
| k:emb->L9/h5 | +1.2990 | +14.10 | hyper |
| v:emb->L8/h8 | +1.2698 | +10.84 | hyper |
| k:emb->L7/h10 | +1.2591 | +5.60 | hyper |
| k:emb->L7/h4 | +1.2562 | +4.84 | hyper |

Positive Delta means the connection is weighted *more* under the `pos` instruction than under `neg`.

## Reading the selection

The rule picked 1598 edges from the real contrast and 20.8 on average from balanced-split nulls that contain no behavioural signal — a ratio of **77.01x**. The circuit is selected well above what paraphrase noise produces.

Held-out sign agreement is 1.000 on selected edges against 0.954 on random ones. When those two are equal, Delta is essentially the same for every item — the routing difference is a function of the instruction alone, so per-item significance testing has no power to separate edges and `--select null` is the rule to use.
