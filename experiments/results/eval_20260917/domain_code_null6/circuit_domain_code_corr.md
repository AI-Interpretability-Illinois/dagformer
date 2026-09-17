# Circuit for `domain_code` — channel `corr`

Source: `experiments/results/eval_20260917/domain_code/contrast_domain_code.pt`  
Selection: `{"rule": "null", "q": 0.005, "s_cutoff": 5.204363870998586, "expected_false_positives": 16.17, "n_null_draws_est": 6, "n_null_draws_cal": 6}`, hyperconnections only

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

These ratios describe the size of the observed routing response. They do not bound the output effect: a small routing change could still matter on sensitive coordinates. Causal verification tests whether the selected differences change the scored behavior.

## Sparsity of the difference matrix

| quantity | value |
|---|---|
| edges total | 3773 |
| edges selected | 813 (21.55%) |
| selected by the matched null (mean) | 6.0 |
| signal / null selection ratio | 135.50x |
| edges holding 50% of |Delta| mass | 422 |
| edges holding 90% of |Delta| mass | 2024 |
| Gini of |Delta| | 0.607 |
| max |Delta| | 5.9202 |
| mean |Delta| | 0.11049 |

## Does it replicate on held-out items?

| quantity | value |
|---|---|
| sign agreement on selected edges | 1.000 |
| sign agreement on random edges | 0.947 |
| corr(Delta_train, Delta_test) on selected | 0.995 |

## Circuit shape

| quantity | value |
|---|---|
| edges | 813 |
| of which hyperconnections (src < layer) | 813 |
| distinct target heads | 176 |
| distinct target layers | 11 |
| connected components | 1 |
| largest component (nodes) | 187 |
| by stream | {"v": 242, "k": 292, "q": 263, "r": 16} |
| by target layer | {"1": 19, "2": 40, "3": 34, "4": 59, "5": 53, "6": 110, "7": 134, "8": 52, "9": 70, "10": 92, "11": 150} |
| by source | {"0": 175, "1": 125, "2": 106, "3": 102, "4": 64, "5": 78, "6": 61, "7": 43, "8": 31, "9": 20, "10": 8} |

## Top edges

| edge | Delta | t | kind |
|---|---|---|---|
| v:emb->L9/h2 | +4.2565 | +7.72 | hyper |
| v:emb->L9/h6 | +3.9130 | +8.76 | hyper |
| v:emb->L6/h11 | +2.5766 | +11.52 | hyper |
| k:emb->L7/h1 | -1.9837 | -9.77 | hyper |
| v:emb->L8/h3 | +1.5654 | +8.45 | hyper |
| v:emb->L9/h14 | -1.3221 | -15.50 | hyper |
| k:emb->L9/h5 | +1.2990 | +14.10 | hyper |
| v:emb->L8/h8 | +1.2698 | +10.84 | hyper |
| k:emb->L7/h10 | +1.2591 | +5.60 | hyper |
| v:emb->L9/h0 | -1.2176 | -15.98 | hyper |
| v:emb->L9/h13 | +1.1745 | +9.17 | hyper |
| k:emb->L7/h15 | +1.1352 | +15.02 | hyper |
| v:emb->L9/h7 | -1.1019 | -19.97 | hyper |
| v:emb->L6/h13 | -1.0193 | -10.41 | hyper |
| k:emb->L9/h6 | -0.9982 | -19.84 | hyper |
| v:emb->L9/h4 | +0.9809 | +4.23 | hyper |
| k:emb->L9/h7 | +0.9469 | +13.43 | hyper |
| v:emb->L9/h1 | -0.9366 | -13.88 | hyper |
| k:emb->L9/h2 | -0.9069 | -22.50 | hyper |
| q:emb->L7/h1 | -0.8912 | -9.11 | hyper |
| v:emb->L9/h9 | -0.8276 | -14.76 | hyper |
| v:emb->L11/h9 | +0.8267 | +7.49 | hyper |
| v:emb->L6/h15 | +0.8230 | +7.80 | hyper |
| v:emb->L11/h5 | +0.8031 | +5.32 | hyper |
| v:emb->L11/h12 | +0.7993 | +4.54 | hyper |

Positive Delta means the connection is weighted *more* under the `pos` instruction than under `neg`.

## Reading the selection

The rule picked 813 edges from the real contrast and 6.0 on average from balanced-split nulls that contain no behavioural signal — a ratio of **135.50x**. The circuit is selected well above what paraphrase noise produces.

Held-out sign agreement is 1.000 on selected edges against 0.947 on random ones. When those two are equal, Delta is essentially the same for every item — the routing difference is a function of the instruction alone, so per-item significance testing has no power to separate edges and `--select null` is the rule to use.
