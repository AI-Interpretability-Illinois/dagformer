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

## Experiment 3: recovery under pruning vs a layerwise-router model (queued)

The prune-during-finetune pipeline (`experiments/pruning/`) runs unchanged on
MUDDFormer through an architecture adapter (`src/pruning/masks.py`,
`MuddAdapter`; tested in `tests/test_pruning_muddformer.py`). Three
streamed-Dolma-era 150M checkpoints trained on the same data (MUDDFormer,
DAGFormer, dense) are pruned to 30 / 50 / 70% of heads and MLP channels while
finetuning on MathInstruct, plus sparsity-0 controls and frozen-router
variants (`freeze_predictor=true` freezes `dense_bs` / `dynamic_dense` on
MUDDFormer and the predictor on DAGFormer). The prediction: at matched
sparsity the model whose router can re-plan all downstream edges recovers
more than one whose routers each see only their own layer. Compare within
this trio only (different data pipeline from the shared models); plot with
`scripts/plot_prune_pareto.py --families muddformer,dagformerst,baselinest`.
