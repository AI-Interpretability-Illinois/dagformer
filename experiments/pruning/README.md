# Pruning during domain finetuning: does routing recover pruned capacity?

Hypothesis under test: DAGFormer has far more inter-module connections than a
dense transformer (every head reads every earlier layer), so when a unit is
pruned during finetuning on a domain, the routing predictor can re-route around
it and the model keeps more of its domain ability at the same parameter budget.
If true, the *same* pruning recipe yields a more compact model from DAGFormer
than from the matched baseline.

Everything here runs identically on a dense OLMo-2 checkpoint and on a
DAGFormer checkpoint; the comparison is between the matched pairs in
`/work/hdd/bfqt/shared/dagformer-models` (75M / 150M / 300M, same data, same
steps).

```
src/pruning/masks.py          StructuredMasker: head / neuron / attn-block / mlp-block
                              masks via forward hooks, Taylor importance from mask
                              gradients, structure-aware parameter accounting
src/pruning/schedule.py       cubic gradual-sparsity schedule (Zhu & Gupta 2017)
scripts/pretokenize_domain.py cached HF dataset -> mmap shards + eval cache
scripts/prune_finetune.py     the trainer: finetune + prune on schedule + recover
scripts/plot_prune_pareto.py  sweep -> Pareto figure + trajectory figure + table
configs/prune/                one config per (size, family); sweeps via --override
scripts/slurm/prune_sweep.sh  submits the matched sweep
src/model/modular_routing.py  the answer to "can the predictor see intra-layer
                              edges": module-granular routing (section 3)
```

## 1. The pipeline

Follows the gradual-pruning-during-finetuning recipe: the network is finetuned
on the domain corpus and, on a schedule, the least important structured units
are removed permanently while training continues, so the remaining weights (and
here also the routing predictor) adapt to each removal.

| piece | choice | source |
|---|---|---|
| schedule | cubic ramp of target sparsity from `prune_start_step` to `prune_end_step`, update every `prune_every` steps, then fixed masks for a recovery phase | Zhu & Gupta 2017 |
| units | attention heads, MLP intermediate channels, whole attention blocks, whole MLP blocks (any subset) | CoFi (Xia et al. 2022) unit set |
| importance | first-order Taylor on a unit's output gate, `|dL/dg|` accumulated over the steps since the last pruning update (also `fisher`, `magnitude`, `random` control) | Michel et al. 2019; Molchanov et al. 2019 |
| selection | global ranking across layers within each unit type; `min_alive_per_layer` guard | |
| accounting | backbone parameters removed, structure-aware (a block that lost all its heads counts once) | |

The masks are multiplicative gates on module outputs, attached with forward
hooks to `o_proj` (heads), `down_proj` (neurons) and the two post-norms
(blocks). Both forward passes call those modules by module, so the same object
prunes the dense HF model and the OLMo backbone inside `FourWayDAGFormer`.
`tests/test_pruning.py` checks that a masked head equals zeroing its `o_proj`
columns, in both forwards, and that the final "baked" checkpoint (pruned
weights zeroed in place, no hooks) reproduces the masked model.

What gets logged (`metrics.csv`, `trajectory.json`, `summary.json` per run):
domain eval NLL and out-of-domain control NLL (WikiText-2) at every eval,
**immediately after every pruning step** (the damage) and at the next periodic
eval (the recovery), backbone parameter sparsity, per-type unit counts and the
per-layer pruning pattern.

### Domain

MathInstruct (262K instruction/solution pairs, cached locally) as the
finetuning corpus with a held-out 2K-document slice as the in-domain eval;
WikiText-2 test as the general-domain control; GSM8K test is tokenized too for
a post-hoc `gsm8k_bpb`-style eval of the final checkpoints. Math was chosen
because the lm-eval study found DAGFormer's largest edge on math-reasoning
text (`experiments/results/lmeval/reasoning/FINDINGS.md`), so it is the domain
where re-routing has the most to recover.

Budget per run: 32K tokens/step x 2000 steps = 65M tokens (about one epoch),
pruning from step 200 to 1400 in 13 updates, 600 recovery steps, single GPU.

### Running

```bash
# 1. data (once; offline from the HF cache)
sbatch scripts/slurm/pretokenize_domain.slurm

# 2. the matched sweep: 75m + 150m, both families, unit sparsity 0.3 / 0.5 / 0.7
bash scripts/slurm/prune_sweep.sh
SIZES=300m SPARSITIES="0.5" bash scripts/slurm/prune_sweep.sh
UNITS=modules SPARSITIES="0.33 0.5" bash scripts/slurm/prune_sweep.sh   # whole blocks
IMPORTANCE=random SPARSITIES=0.5 bash scripts/slurm/prune_sweep.sh     # control

# 3. figures + table
python scripts/plot_prune_pareto.py --out experiments/pruning/results
```

A single run: `CONFIG=configs/prune/150m_dagformer_math.yaml OVERRIDES="target_sparsity=0.7" sbatch scripts/slurm/prune_finetune.slurm`.
Useful overrides for the DAGFormer side: `freeze_predictor=true` (no
re-routing allowed: isolates how much of the recovery is routing) and
`freeze_base=true` (recovery through routing alone).

### Reading the result

The headline figure is domain NLL against remaining backbone parameters, one
curve per family. The hypothesis predicts the DAGFormer curve lies below the
baseline curve by *more* than the unpruned gap, i.e. the pair's delta grows
with sparsity. Two things to keep honest:

- The predictor is ~30M parameters at 300M scale and is not pruned. Report
  backbone params (what the recipe removes) and total params (what you
  deploy); at high sparsity the predictor is a real share of the total.
- The post-prune ticks in the trajectory figure separate *robustness to
  removal* (smaller immediate damage) from *recovery* (faster return). Both
  are consistent with re-routing, but only the second needs the predictor to
  move; the `freeze_predictor` ablation tells them apart.

## 2. Why the current predictor cannot prune modules

`FourWayDAGFormer` routes head inputs over *layer outputs* X_0..X_l. Inside a
layer the wiring is fixed:

```
X_{l+1} = R_l + a_l + m_l        a_l = attention block output, m_l = MLP block output
mlp_in  = R_l + a_l
final   = X_L
```

The attention->MLP edge, the "block -> everything downstream" edges and the
read-out are hard-wired at weight 1 and invisible to the predictor. It can add
hyperconnections; it cannot express "this block's output is not needed here",
so no routing matrix column corresponds to a module and the routing cannot be
used as a pruning signal.

## 3. Module-granular routing (`routing_mode: fourway_modular`)

`src/model/modular_routing.py` changes the *source set* to the module outputs:

```
S_l = [emb, a_0, m_0, a_1, m_1, ..., a_{l-1}, m_{l-1}]           (2l+1 sources)

q, k, v   head h of layer l reads S_l              alpha [B, T, H, 2l+1]
r         residual carrier of layer l reads S_l    alpha [B, T, 2l+1]
m         MLP input of layer l reads S_l + [a_l]   alpha [B, T, 2l+2]   <- attn->MLP edge
o         final norm reads S_L                     alpha [B, T, 2L+1]
```

A pre-norm-residual transformer is exactly the all-ones routing
(X_l = emb + sum(a_i + m_i)), so the identity init is alpha = 1 everywhere
(`tests/test_modular_routing.py` checks step 0 equals vanilla OLMo-2). Every
edge of the standard model, including the same-layer attention->MLP edge, is
now one routing entry on the same footing as a skip over ten layers.

What this buys:

- **A module is a column.** Its whole downstream influence is
  `alpha[.., s]` over all readers. `source_column_mass(rw)` reports it per
  module; zero column = module deletable with no change to the function
  (tested). `prune_finetune.py --override importance=routing_column` prunes
  whole blocks by this routing-native score instead of Taylor.
- **Sparsity on the connection matrix itself.** `routing_sparsity_lambda`
  adds a regulariser on the routing entries, ramped in linearly between
  `routing_sparsity_start_frac` and `routing_sparsity_warmup_frac`:
  `l1` (per-token lasso), `sqrt` (a group over tokens per edge, so an edge
  is driven to zero for *all* tokens and can be dropped from the graph),
  or `column` (the module group-lasso below). `routing_sparsity_hyper_only`
  leaves the vanilla transformer's sequential edges free so only skips are
  penalised; `routing_sparsity_streams` restricts it to chosen reads. Logged:
  mean |alpha|, fraction of entries and of whole edges below eps, and with
  `routing_sparsity_eval_eps > 0` an `eval/nll_sparsified` where entries
  below eps are hard-zeroed, so the sparsity claimed is the sparsity that
  works. The same option exists in `prune_finetune.py` for finetuning.
- **The predictor can learn to switch modules off.** `routing_column_group_lambda`
  adds a group-lasso `sum_s sqrt(mass_s)` over columns, zero-inducing on whole
  columns rather than thinning edges uniformly. Column statistics are logged
  as `routing/column_mass_*` and `routing/dead_sources` during pretraining.
- **More informative predictor.** Layer-level sources bundle attention and MLP
  contributions; module-level sources let a head read layer 3's attention
  but not its MLP, and let the MLP of layer l ignore its own attention block.

Cost: the fused QKV projection runs over 2l+1 instead of l+1 sources (about
1.8x FourWay's routing overhead at L=12), and the predictor emits
(3H+1)(2l+1) + (2l+2) values per routed layer plus 2L+1 for the read-out.

To train one: `configs/pretrain_75m_dagformer_modular.yaml` (same recipe as
the shared 75M DAGFormer; the Dolma shards it points to were deleted from
`/work/hdd/bfqt/data/pretok` and need rebuilding with `scripts/pretokenize.py`
first). `fourway_modular_corrected` adds the per-layer correction MLPs. The
verified loader (`scripts/eval_lm_harness.load_fourway`) and the prune trainer
dispatch on the routing mode, so a modular checkpoint goes through the same
pruning sweep and lm-eval harness as the others.

Not done: physically slicing pruned weights into a smaller HF config (masks
are baked as zeros instead; parameter counts are analytic), and a Flax port.

## Results (75M / 150M / 300M matched pairs, math domain, 2026-09-25)

Domain eval NLL on held-out MathInstruct after 2000 finetuning steps.
"unpruned" is the same finetuning with no pruning. Full tables:
`results/prune_summary.md`; figures: `results/prune_pareto.png`,
`results/prune_pareto_total.png`, `results/prune_trajectory.png`.

| units | block sparsity | 75M base / DAG (gap) | 150M base / DAG (gap) | 300M base / DAG (gap) |
|---|---|---|---|---|
| unpruned finetune | 0% | 2.388 / 2.223 (+0.165) | 1.883 / 1.743 (+0.139) | 1.608 / 1.488 (+0.120) |
| heads + neurons | 30% | 2.478 / 2.298 (+0.180) | 1.957 / 1.807 (+0.150) | 1.665 / 1.541 (+0.124) |
| heads + neurons | 50% | 2.590 / 2.391 (+0.199) | 2.071 / 1.899 (+0.172) | 1.762 / 1.620 (+0.142) |
| heads + neurons | 70% | 2.829 / 2.614 (+0.215) | 2.278 / 2.091 (+0.187) | 1.967 / 1.790 (+0.177) |
| whole blocks | 33% | 2.610 / 2.547 (+0.063) | 2.200 / 2.170 (+0.030) | |
| whole blocks | 50% | 2.832 / 3.452 (**-0.620**) | 2.380 / 2.245 (+0.135) | |
| heads + neurons, random importance | 50% | 3.029 / 2.935 (+0.094) | 2.280 / 2.129 (+0.151) | |
| heads + neurons, predictor frozen | 50% | DAG 2.388 | DAG 1.899 | |

What the data says:

- **The DAGFormer advantage grows with sparsity at every size.** The
  unpruned finetuning gap is 0.165 / 0.139 nats (75M / 150M); at 70% of
  block parameters removed it is 0.215 / 0.187, and at 300M it rises from
  0.120 (unpruned) to 0.177 (70%). In compactness terms, DAGFormer with half its
  heads and MLP channels removed matches or beats the unpruned finetuned
  baseline at 75M (2.391 vs 2.388) and 150M (1.899 vs 1.883). Post-prune
  ticks in the trajectory figure show smaller immediate damage for DAGFormer
  at high sparsity.
- **Relative degradation** (minus the increase of held-out MathInstruct NLL
  over the same model's unpruned finetune; the 300M unpruned controls ran
  2026-10-01, outputs in /work/hdd/biro/xiaocong/dagformer_pruning, linked
  into the checkpoint root):

  | removed | 75M dense / DAG (adv.) | 150M dense / DAG (adv.) | 300M dense / DAG (adv.) |
  |---|---|---|---|
  | 30% | -3.7% / -3.4% (+0.4 pts) | -3.9% / -3.6% (+0.3 pts) | -3.6% / -3.5% (+0.0 pts) |
  | 50% | -8.4% / -7.5% (+0.9 pts) | -10.0% / -8.9% (+1.1 pts) | -9.6% / -8.9% (+0.7 pts) |
  | 70% | -18.5% / -17.6% (+0.9 pts) | -21.0% / -20.0% (+1.1 pts) | -22.3% / -20.3% (+2.1 pts) |

  In relative terms the advantage is small at 30% (none at 300M) and grows
  with sparsity; at 70% it is largest at 300M. DAGFormer starts 7.4% lower
  at 300M (6.9% / 7.4% at 75M / 150M), so at 50% removed it is within 0.9%
  of the unpruned dense finetune at every size (75M +0.1%, 150M +0.9%, 300M +0.8%).
- **It is not the predictor re-routing.** Freezing the global predictor
  during prune-finetune changes nothing (75M: 2.388 vs 2.391; 150M: 1.899
  vs 1.899). Consistent with `experiments/coherence` (the predictor's
  per-token output is causally inert), the robustness comes from the routed
  weights themselves and, in the `fourway_corrected` checkpoints, from the
  local correction MLPs, which stay trainable under `freeze_predictor`.
- **Importance selection matters, for both models.** Random selection at 50%
  costs the baseline 0.44 / 0.21 nats and DAGFormer 0.54 / 0.23 nats
  (75M / 150M) relative to Taylor, and the gap between the two families
  shrinks at 75M, so part of the DAGFormer edge is that Taylor gates rank
  its units better.
- **Whole-block pruning is the exception at 75M** (DAGFormer 3.45 vs 2.83 at
  half the blocks removed) but not at 150M (+0.135 for DAGFormer). With 6
  layers, removing 3 of 6 attention blocks and 3 MLPs leaves too little for
  either model; Taylor on a block gate ranked layers 3 to 5's attention
  lowest for DAGFormer and the removal cost 8 nats before recovery. The
  routing-column importance of the modular variant (section 3) targets
  exactly this case.
- Parameter caveat: the untouched predictor is 29M parameters at 75M, 30M
  at 150M, so at equal *total* parameters a heavily pruned DAGFormer can be
  larger than the unpruned baseline. `results/prune_pareto_total.png` shows
  the total-parameter view; the claim supported here is about the backbone.

## References

- Zhu & Gupta, 2017. To prune, or not to prune: exploring the efficacy of pruning for model compression.
- Michel, Levy & Neubig, 2019. Are sixteen heads really better than one?
- Molchanov et al., 2019. Importance estimation for neural network pruning.
- Xia, Zhong & Chen, 2022. Structured pruning learns compact and accurate models (CoFi).
- Xia et al., 2023. Sheared LLaMA: accelerating language model pre-training via structured pruning.
