# Scaling analysis: DAGFormer (routed) vs dense OLMo-2, along model size, data and compute

Status (2026-09-28): pipeline in place and validated on the finished runs; the final numbers are
regenerated automatically by the gpua046 queue after the last pretraining item (1B fourway_corrected,
ETA 2026-09-30) and by hand for the timan runs. `results.md`, `runs_table.md`, `scaling_fits.json`
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
| delta21b (21B-token tokenisation) | gpua046/047 | 300M x {dense, corrected, modular}, 1B x {dense, corrected} | 300M: 21 (Chinchilla); 1B: 4 | 1B at 5B tokens by decision (routing is ~2x dense cost at 1B) |
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

## Preliminary reading (training-time evals, before the common re-evaluation)

* Own-corpus final NLL, dense / corrected / modular: 75M (timan12b) 4.352 / 4.212 / 4.195;
  150M (timan12b) 3.776 / 3.673 / 3.689; 75M (delta1p7b) 4.213 / 4.014 / 3.926. The routed
  advantage is 0.10-0.20 nats at 75M-150M and, on the 75M curves, appears after the first quarter of
  training and then slowly shrinks (`results.md`, "gap along training"), i.e. the routing helps most
  in the data-limited regime of a run.
* The 300M and 1B comparisons (same corpus, same eval cache) land with the chains: 300M corrected
  (running) vs modular 2.2675 (done) vs dense (queued); 1B dense 2.2106 at 5B tokens vs 1B corrected
  (running).

## Caveats to keep in the paper

* One seed per run; differences below ~0.02 nats are noise at these eval-set sizes.
* Three corpora: the cross-corpus fits rely on the common re-evaluation; training corpora are all
  Dolma v1.7 but different slices, so a small offset between corpora is possible (visible as
  the corpus labels on `fig_loss_vs_params.png`).
* The 1B runs are at 4 tokens/param, not Chinchilla; they enter the L(C) fit and the gap-vs-N plot
  but not the Chinchilla L(N) fit.
* Routing FLOPs are analytic (MAC counts), not measured; the observed wall-clock ratios (1.2x at
  75M, ~2x at 300M, ~3x at 1B for modular) are consistent with them.
