# Global predictor vs layerwise routers: three experiments

DAGFormer's distinguishing design choice against mHC / MUDDFormer /
Hyper-Connections is *where the routing comes from*: one external predictor
reads the tokens and emits every layer's routing before the backbone runs,
instead of each layer computing its own mixing from its own hidden state. The
claim that follows is that a single module can plan a *joint* topology and
co-evolve with the backbone in a way independent per-layer routers cannot.
Three experiments test that claim from different sides.

| | question | where |
|---|---|---|
| 1 | At matched backbone, DoF and tokens, does a global router beat local routers? | `configs/locality/`, `scripts/slurm/locality_*.slurm` |
| 2 | On trained checkpoints, is the global router's per-layer output actually *coordinated* across layers, and is that coordination used? | `cross_layer_patch.py`, `results/` |
| 3 | When units are pruned during finetuning, does a global router recover better than layerwise routers? | `configs/prune/150m_{muddformer,dagformerst,baselinest}_math.yaml`, `experiments/pruning/runlists/exp3_*.txt` |

## Experiment 2: cross-layer coherence (done, all sizes)

The shipped checkpoints have both kinds of router in one model, addressed in
the same coordinates: `pred` (the global `FourWayPredictor`) and `corr` (the
per-layer correction MLPs, an mHC-style local router reading the residual).
For each channel, every layer's routing distribution is kept but the
cross-layer pairing is manipulated:

- **coherent**: all layers' routing of channel c taken from one other window
  (a consistent foreign topology);
- **incoherent**: each layer from an independently chosen window (same
  per-layer marginals, cross-layer pairing destroyed);
- **posmean**: channel c replaced by its per-position mean.

`specificity = coherent - intact` is how much the channel's per-token output
matters at all; `gap = incoherent - coherent` is the cross-layer coordination
the model relies on. 64 windows x 256 tokens, 3 permutations, paired
bootstrap CIs; numbers are mean NLL in nats.

| model | windows | intact | pred specificity | pred gap | corr specificity | corr gap | dev-norm corr across layers (pred / corr) |
|---|---|---|---|---|---|---|---|
| 75M | math | 4.319 | +0.007 | +0.001 | +0.560 | -0.056 | 0.92 / 0.39 |
| 75M | wikitext | 5.121 | +0.005 | -0.000 | +0.672 | -0.066 | 0.93 / 0.48 |
| 150M | math | 3.76 | +0.00 | +0.00 | +0.797 | -0.084 | 0.93 / 0.44 |
| 150M | wikitext | 4.499 | +0.002 | +0.000 | +0.923 | -0.075 | 0.88 / 0.48 |
| 300M | math | 3.383 | +0.001 | -0.001 | +1.062 | -0.153 | 0.94 / 0.28 |
| 300M | wikitext | 4.080 | +0.002 | -0.000 | +1.294 | -0.128 | 0.95 / 0.27 |


What it says:

1. **The global predictor's per-token output is causally inert on every
   checkpoint** (specificity <= 0.007 nats at all sizes, both domains). This
   replicates the interp study's finding on 300M and extends it to 75M and
   150M: whatever the global predictor contributes, it contributes through
   its average, not through per-token planning.
2. **Its output *is* highly coordinated across layers** (the per-token
   deviation norms correlate 0.88-0.95 between layers) but nothing downstream
   depends on that coordination (gap = 0). The local routers carry all the
   token-specific routing (0.56-1.29 nats) and are only weakly coordinated
   (0.27-0.48), decreasing with model size.
3. **No channel shows a positive coherence gap.** `corr` is negative because
   independently sourced donors partially average out (posmean is the best of
   the three conditions), which is exactly the behaviour of independent
   per-layer routers.

So on the existing checkpoints the "one module plans a joint topology" claim
cannot be supported: training used the global predictor as a shared static
topology and let the local routers do the per-token work. This is a fact about
how these models were trained (the local channel is the cheaper path to
per-token variation), not a proof that a global router cannot do it, which is
what experiment 1 tests by removing the local channel.

## Experiment 1: router-locality ablation (queued)

Same 75M FourWay backbone, same routing granularity, 1.57B Dolma tokens,
4 x A100 each; only the source of alpha changes:

| arm | routing source | config |
|---|---|---|
| `global` | one shared predictor, no local routers | `75m_global.yaml` (`routing_mode: fourway`) |
| `local` | per-layer correction MLPs, predictor frozen at identity | `75m_local.yaml` (`fourway_corrected` + `freeze_predictor`) |
| `both` | the shipped configuration | `75m_both.yaml` |
| `per_layer` | an independent encoder per layer on the same tokens (sees the input, no shared computation) | `75m_per_layer.yaml` (`fourway_predictor_variant: per_layer`) |
| `dense` | OLMo-2 baseline | `75m_dense.yaml` |

`global` vs `local` is the headline; `per_layer` vs `global` separates "sees
the input before the backbone runs" from "one shared module". The older
`fourway_pure` / `fourway_local_only` archive metrics are from the pre-bugfix
overfit era and are not usable. Jobs: `locality_prep` (splits the eval cache
off the new Dolma slice) then five `locality_<arm>` jobs; after they finish,
run `cross_layer_patch.py` on the `global` arm: if a global-only model still
shows zero specificity, the predictor is vestigial by construction and not
by competition with local routers.

## Experiment 3: recovery under pruning vs a layerwise-router model (done)

The prune-during-finetune pipeline (`experiments/pruning/`) runs unchanged on
MUDDFormer through an architecture adapter (`src/pruning/masks.py`,
`MuddAdapter`; tested in `tests/test_pruning_muddformer.py`). Three
streamed-Dolma-era 150M checkpoints trained on the same data (MUDDFormer with
per-layer dynamic-dense routers, DAGFormer, dense) were pruned to 30 / 50 /
70% of heads and MLP channels while finetuning on MathInstruct, plus
sparsity-0 controls and frozen-router variants. Domain eval NLL; table and
figures in `results/exp3/`. Compare within this trio only (different data
pipeline from the shared models).

| block sparsity | dense | MUDDFormer | DAGFormer | DAG - MUDD | max post-prune damage (MUDD / DAG) |
|---|---|---|---|---|---|
| 0% (finetune only) | 2.075 | 1.935 | 1.924 | 0.011 | |
| 30% | 2.157 | 2.037 | 1.990 | 0.047 | |
| 50% | 2.273 | 2.153 | 2.087 | 0.066 | 2.433 / 2.410 |
| 70% | 2.507 | 2.367 | 2.268 | 0.099 | 2.569 / 2.464 |
| 50%, routers frozen | | 2.150 | 2.078 | 0.072 | |

- Unpruned, the two routed models are equivalent on this domain (0.011
  nats apart); under pruning the DAGFormer edge over MUDDFormer grows
  monotonically to 0.10 nats at 70%, and its immediate damage after each
  pruning step is smaller. So at matched routing granularity and matched
  unpruned quality, the model routed by an external predictor over all prior
  layers degrades more gracefully than the model with per-layer routers.
- Freezing the routers changes nothing for either model (MUDD 2.150 vs
  2.153, DAG 2.078 vs 2.087). The recovery is done by the backbone weights
  under a fixed routing pattern, not by routers re-planning. Read together
  with experiment 2, the advantage is structural (which sources each head
  can draw on) rather than a property of on-line replanning. Note that for
  DAGFormer `freeze_predictor` freezes the global predictor while the
  correction MLPs remain trainable; the MUDDFormer arm freezes all of its
  routers.

## Experiment 2 on the locality arms (2026-10-03, job 22636494; `results/locality/`)

Same patching test on the 75M router-locality arms (64 windows x 256 tokens, own-corpus cache / WikiText-2).

| arm | channel | specificity (own / wikitext2) | coherence gap | posmean − intact | dev-norm corr across layers |
|---|---|---|---|---|---|
| global (predictor only) | pred | +0.077 / +0.034 | −0.011 / −0.005 | +0.022 / +0.008 | 0.98 |
| per_layer (unshared predictors) | pred | +0.093 / +0.038 | −0.018 / −0.007 | +0.026 / +0.006 | 0.69 |
| local (correction MLPs only) | corr | +0.817 / +0.585 | −0.060 / −0.027 | +0.478 / +0.399 | 0.43 |
| both (shipped DAGFormer) | pred | +0.005 / −0.001 | −0.001 / −0.000 | +0.003 / −0.002 | 0.92 |
| both | corr | +0.718 / +0.519 | −0.071 / −0.038 | +0.394 / +0.320 | 0.35 |

What it adds to the shipped-checkpoint result:
- The predictor's per-token output is inert only when a local channel is present. Alone (`global`), it carries
  0.03-0.08 nats of token-specific routing, and its output is near-perfectly coordinated across layers
  (0.98), though nothing relies on the coordination (gap <= 0). So the inertness in the corrected models is
  competition (the local channel is the cheaper path to per-token variation), not a property of the
  predictor by construction.
- Even in the predictor-only model, replacing its output by the per-position mean costs just 0.008-0.022
  nats, a small fraction of the arm's advantage over dense (about 0.18-0.4 nats on these caches): most of
  what the predictor contributes is a static topology.
- Local routers are far more token-specific (0.5-0.8 nats) and only weakly coordinated across layers
  (0.35-0.44), with a negative coherence gap, i.e. they behave as independent per-layer routers.

## Experiment 1 addendum: static-table arm (2026-10-03, job 22638614; eval 22638947)

`configs/locality/75m_static.yaml` (`fourway_predictor_variant: static`): the FourWay routing weights are one
learned table per layer (500 parameters in total), no token input, same backbone, data and recipe as the arms.

| arm | own cache | wikitext2 | mathinstruct | gsm8k |
|---|---|---|---|---|
| dense | 4.213 | 6.593 | 5.373 | 5.404 |
| static table | 4.171 | 6.529 | 5.243 | 5.280 |
| global (predictor only) | 4.028 | 6.207 | 4.981 | 5.116 |

The static table recovers only about a quarter of the predictor-only arm's gain over dense (0.04 of 0.19 nats on
the own cache; 0.06 of 0.39 on wikitext2). Read with the patching results (the trained predictor's output can be
replaced by its mean at <= 0.02 nats): the token-conditioned predictor is needed to *learn* the topology, not to
apply it. Single run; the table has 500 parameters against the predictor's 29M.
