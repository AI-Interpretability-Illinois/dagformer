# Scaling analysis: DAGFormer (routed) vs dense OLMo-2, along model size, data and compute

Status (2026-10-01): every run except the 1B fourway_corrected 10B-token continuation is in the
tables (1B dense 10B added by job 22590844). That last run (on the gpua046 chain, moving to gpua038)
is due ~2026-10-03 and its queue re-runs the common eval, collect and fit; the timan runs are refreshed by hand. `results.md`, `runs_table.md`, `scaling_fits.json`
and the four `fig_*.png` are outputs, do not edit them. This file holds the method, the run inventory
and the interpretation, which is updated when the final numbers land.

## What is measured

Every pretraining run we have (26 across three machines, `runs.yaml`) is characterised by

* **N**: backbone parameters (OLMo-2, tied embeddings; total and non-embedding are both recorded;
  routing parameters, i.e. the FourWay predictor + correction MLPs, are listed separately and are
  NOT part of N, because the question is what the routing buys a given backbone);
* **D**: tokens seen (`train/tokens_seen_B` from metrics.csv, or steps x tokens/step);
* **C**: training FLOPs = 6 x forward MACs/token x D, with the routing cost included: FourWay
  routing projects every prior layer's output through q/k/v (an extra `sum_l l * 3D^2` MACs), the
  modular variant twice that, plus the predictor encoder/trunk/heads and correction MLPs. This is
  what makes the iso-compute comparison honest: the routed models cost 1.2x (75M) to 2.1x (1B)
  dense FLOPs per token (`runs_table.md`, column "FLOPs/token");
* **L**: held-out NLL. Two kinds:
  * *training-time curve*: the trainer's periodic eval on the run's own corpus cache
    (`eval/nll_soft` for routed models, `eval/nll` for dense). Comparable only within a corpus;
    used for the L(D) curves and the dense-routed gap along training;
  * *common re-evaluation* (`scripts/scaling_common_eval.py`): the final checkpoint of every run on
    the same four caches (Dolma-21B tail, wikitext-2, MathInstruct, GSM8K). Comparable across
    machines and corpora; used for every cross-size fit.

## Runs (three corpora, all Dolma v1.7)

| corpus | machine | sizes x families | tokens/param | note |
|---|---|---|---|---|
| delta21b (21B-token tokenisation) | gpua046/047 | 300M x {dense, corrected, modular}, 1B x {dense, corrected} at 5B and 10B tokens | 300M: 21 (Chinchilla); 1B: 4 and 8 | 1B at 5B tokens by decision (routing is ~2x dense cost at 1B), both continued to 10B (LR re-warm) |
| shared12b (original 12B slice) | Delta shared models | {75M,150M,300M,600M} dense, {75M,150M,300M} corrected | 21 (300M corrected stopped at 9000/12000 steps, 15.5) | the paper's original pairs; no metrics.csv, common eval only |
| timan12b (local 12B rebuild) | timan1 / timan108 | {75M,150M} x {dense, corrected, modular}, 300M modular | 21 | same configs as shared12b |
| delta1p7b (1.7B slice) | gpua046/047 | 75M x {dense, corrected, modular, global, local, per_layer, modular_sparse} | 21 | the router-locality arms (experiment 1) |

## Fits (`scripts/scaling_fit.py`)

1. **L(N) at Chinchilla-scale budgets** (12-30 tokens/param), per family, on the common Dolma-21B
   NLL: `L = E + A N^-alpha`. Dense has 5 points (75M...600M shared + 300M delta21b), corrected 4
   (75M, 150M, 300M x2), modular 3 (75M, 150M, 300M x2). The 1B runs (4 tokens/param) are shown but
   excluded from this fit.
2. **Effective parameters**: from the dense fit, the dense size `N_eff` whose loss equals each routed
   run's loss; `N_eff / N` is the "parameter multiplier" of the routing at that size. If it grows
   with N the routing scales better than dense, if it shrinks the advantage is a small-model effect.
3. **Gap dense - routed** at matched corpus, size and tokens: final (common eval) vs N, and along
   training (own-corpus curves at the same step) vs D.
4. **Joint L(N, D) = E + A/N^alpha + B/D^beta** per family and corpus on the second half of every
   training curve. Indicative only: one cosine-schedule run per size, so intermediate points are
   pessimistic (beta is an upper bound) and 2-3 sizes per corpus do not pin alpha.
5. **L(C)** on the common-eval finals (all budgets) and the **compute multiplier**: the dense compute
   that reaches each routed run's loss, divided by the routed run's compute. Below 1 means the extra
   routing FLOPs are not repaid at that size; this is the number to quote against "just train a
   bigger dense model for the same FLOPs".

## How to regenerate

```
# Delta (queued automatically on gpua046 after the 1B run; one-off: sbatch scripts/slurm/scaling_common_eval.slurm)
bash scripts/scaling_common_eval.sh delta            # -> common_eval_delta.json (resumable)
python scripts/scaling_collect.py --machine delta    # -> collected_delta.json
# timan1 / timan108 (checkout copy in /srv/local/xy51/scaling/dagformer, caches in /srv/local/xy51/scaling/data)
CUDA_VISIBLE_DEVICES=1 PYTHON=/home/xy51/anaconda3/envs/modularity/bin/python EVAL_ROOT=/srv/local/xy51/scaling/data \
    REPO_DIR=$PWD bash scripts/scaling_common_eval.sh timan1
PYTHONPATH=$PWD python scripts/scaling_collect.py --machine timan1
# then copy collected_timan*.json / common_eval_timan*.json here and
python scripts/scaling_fit.py                        # -> results.md, runs_table.md, scaling_fits.json, fig_*.png
```

## Results so far (2026-10-01, all runs except the 1B DAGFormer at 10B tokens; common wikitext-2 NLL unless stated)

Full tables: `results.md`; per-run inventory: `runs_table.md`; figures `fig_*.png`.

**Gap at matched corpus, size and tokens** (dense minus routed, final checkpoints):

| size | corpus | dense | fourway_corrected | gap | fourway_modular | gap |
|---|---|---|---|---|---|---|
| 75M | shared12b | 4.980 | 4.775 | +0.21 | | |
| 75M | timan12b | 4.997 | 4.711 | +0.29 | 4.738 | +0.26 |
| 150M | shared12b | 4.297 | 4.100 | +0.20 | | |
| 150M | timan12b | 4.268 | 4.085 | +0.18 | 4.116 | +0.15 |
| 300M | delta21b | 3.808 | 3.544 | +0.26 | 3.628 | +0.18 |
| 1B, 5B tok (4 tok/param) | delta21b | 3.569 | 3.307 | +0.26 | | |
| 1B, 10B tok (8 tok/param) | delta21b | 3.446 | due ~Oct 3 | | | |

The routed advantage does not shrink from 75M to 1B (0.18-0.29 nats for fourway_corrected; 0.26 at
300M and at 1B). The two
300M dense runs agree to 0.004 nats across corpora (shared 3.812 at 12B slice, Delta 3.808 at 21B corpus),
so wikitext-2 is a fair cross-corpus yardstick.

**L(N) = E + A N^-alpha at 12-30 tokens/param** (few points, E and alpha are correlated; read the
effective-parameter column, not the exponents): dense E 2.80, alpha 0.56 (7 points, 77M-683M);
fourway_corrected E 1.89, alpha 0.38 (6 points); fourway_modular E 2.15, alpha 0.40 (4 points).

**Effective parameters** (dense size with the same loss on the dense fit): fourway_corrected 1.20-1.29x at
75M and 150M, 1.39x (shared, 16 tok/param) to 1.72x (Delta, 21 tok/param) at 300M; fourway_modular
1.24x at 75M-150M, 1.33-1.42x at 300M. The multiplier grows with N over this range.

**Iso-compute** (C = 6 x MACs/token x tokens, routing cost included; C_eq = dense compute reaching the
same loss on the dense L(C) fit, 9 dense points with 1B dense 5B and 10B): fourway_corrected C_eq / C =
1.20-1.36 at 75M-150M, 1.65-1.93 at 300M and 2.13 at 1B (5B tokens), i.e. it beats a dense model given the
same FLOPs despite costing 1.18-2.13x per token, and the margin grows with size; fourway_modular 1.15 at
75M and 0.81-0.93 at 150M-300M, i.e. its 1.33-2.30x routing cost is not repaid. This is the compute-side
argument for fourway_corrected as the main method.

**Direct iso-compute check at 1B, no fit involved:** the routed 1B at 5B tokens (86.1 EFLOP) against the
dense 1B at 10B tokens (80.8 EFLOP, 6% less compute): wikitext-2 3.307 vs 3.446, MathInstruct 2.403 vs
2.483, GSM8K 2.755 vs 2.900. Spending the routing FLOPs beats spending the same FLOPs on twice the data.
The routed 10B continuation is already below the dense 10B final on the own-corpus cache at step 14000
(2.028 vs 2.116; matched-step gap 0.12-0.15 nats from step 10000 to 14000, vs 0.13 at the 5B end).

**Along training** (own-corpus curves, same cache within a corpus): the gap appears within the first
quarter of every run, peaks, and then narrows slowly (`results.md`, "gap along training"); the joint
L(N, D) fits are reported per corpus but rest on 2-3 sizes each and one cosine run per size.

Pending: the 1B fourway_corrected 10B continuation (ETA ~2026-10-03) completes the 10B pair; the 1B runs
(4 and 8 tokens/param) stay out of the Chinchilla L(N) fit, so the 1B effective-parameter multiplier is
not defined, but they enter the gap table, the L(C) fit and the compute multiplier. The gpua046/gpua038
queue re-runs the common eval, collect and fit automatically after it.

## The Dolma-21B held-out tail is not a neutral cache (found 2026-09-29)

`dolma_v1_7_21b/eval_cache.pt` is the last 200 windows of the corpus's last shard, and the corpus was
tokenized by 8 workers over 8 contiguous blocks of Dolma's source-sorted file list (w0 = books, w1-w5 =
web, w6 = code, w7 = tulu_flan + wiki). Training samples are block-shuffled over the whole concatenation,
so the 21B-corpus runs train on a mix of all sources (~12% flan/wiki), but the eval tail is 100% flan/wiki
templated text. That is why the 21B-corpus 300M runs score ~2.22 on it while the timan 300M modular scores
3.70 (both ~3.6 on wikitext-2), and why the 1B dense reads 2.21 at 5B tokens. Consequences: dolma21b
NLL is comparable only within the 21B corpus; cross-corpus fits use `wikitext2` (the default `--eval-key`
now), with MathInstruct and GSM8K as secondary columns; the 1.7B-slice cache is similarly a single
Gutenberg gazetteer, so the locality arms' training-time evals are one-document numbers. A source-balanced
Dolma held-out set from files none of the corpora consumed would fix this for every model and is the
recommended follow-up before the paper's tables are final.

## Caveats to keep in the paper

* One seed per run; differences below ~0.02 nats are noise at these eval-set sizes.
* Three corpora: the cross-corpus fits rely on the common re-evaluation; training corpora are all
  Dolma v1.7 but different slices, so a small offset between corpora is possible (visible as
  the corpus labels on `fig_loss_vs_params.png`).
* The 1B runs are at 4 tokens/param, not Chinchilla; they enter the L(C) fit and the gap-vs-N plot
  but not the Chinchilla L(N) fit.
* Routing FLOPs are analytic (MAC counts), not measured; the observed wall-clock ratios (1.2x at
  75M, ~2x at 300M, ~3x at 1B for modular) are consistent with them.
