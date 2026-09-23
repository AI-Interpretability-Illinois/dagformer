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

## Results so far (75M pair, math domain, 2026-09-23)

Numbers are domain eval NLL on held-out MathInstruct after 2000 finetuning
steps; "unpruned" is the same finetuning with no pruning. Full table:
`results/prune_summary.md`; figures: `results/prune_pareto.png`,
`results/prune_trajectory.png`. 150M / 300M runs and the random-importance
and frozen-predictor controls are queued.

| units | block sparsity | baseline | dagformer | baseline - dagformer |
|---|---|---|---|---|
| heads + neurons | 0.30 | 2.478 | 2.298 | +0.180 |
| heads + neurons | 0.50 | 2.590 | 2.391 | +0.199 |
| heads + neurons | 0.70 | 2.829 | 2.614 | +0.215 |
| whole blocks | 0.33 | 2.610 | 2.547 | +0.063 |
| whole blocks | 0.50 | 2.832 | 3.452 | **-0.620** |

What the 75M data says:

- **Head/neuron pruning: the DAGFormer advantage grows with sparsity**
  (0.18 -> 0.20 -> 0.22 nats from 30% to 70% of block parameters removed),
  and the post-prune ticks in the trajectory figure show much smaller
  immediate damage for DAGFormer at 70% (peak 2.9 vs 3.2). Read against the
  unpruned-finetune gap once the sparsity-0 controls finish; the
  frozen-predictor run will say how much of this is re-routing versus the
  routed model simply being more robust to removal.
- **Whole-block pruning is where DAGFormer loses.** At one third of blocks
  removed the gap shrinks to 0.06; at half it flips hard (DAGFormer 3.45 vs
  2.83). Taylor importance on a block gate ranked layers 3-5's attention and
  layers 1/3/4's MLP lowest for DAGFormer, and removing them cost 8 nats
  before recovery versus 6 for the baseline. In the FourWay model every later
  head has trained routing weights onto those blocks' layer outputs, so a
  block removal perturbs every downstream reader at once, and a scalar gate
  gradient is a poor estimate of that. This is exactly the case the
  module-granular routing (section 3) is built for: the block's routing
  column is the right importance score, and the predictor can be trained
  to turn columns off gradually instead of having them cut.
- Parameter accounting caveat: at 75M the untouched predictor is 29M
  parameters, so at equal *total* parameters the 70%-pruned DAGFormer (88M)
  is larger than the unpruned baseline (77M). The claim this experiment can
  support at 75M is about the backbone; `results/prune_pareto_total.png`
  shows the total-parameter view.

## References

- Zhu & Gupta, 2017. To prune, or not to prune: exploring the efficacy of pruning for model compression.
- Michel, Levy & Neubig, 2019. Are sixteen heads really better than one?
- Molchanov et al., 2019. Importance estimation for neural network pruning.
- Xia, Zhong & Chen, 2022. Structured pruning learns compact and accurate models (CoFi).
- Xia et al., 2023. Sheared LLaMA: accelerating language model pre-training via structured pruning.
