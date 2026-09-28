# Inverse input optimization and sparse routing mediation — 2026-09-28

Two exploratory experiments on the frozen **300M FourWay + local correction,
step 9000** checkpoint. The procedure runs, but has **not identified a semantic
entity–color binding circuit**. A sparse set can reproduce an increased tendency
to output color words. The more specific binding improvement from a second soft
prefix does not transfer through routing coefficients alone.

## First experiment: full-vocabulary target likelihood

Optimize an eight-position soft prefix in both backbone and predictor embedding
spaces, shared by all examples, for 100 updates. The model and all routing-network
parameters remain frozen. Select the prefix at step 80 using validation true-token
log probability. Train/validation/test contain 64/32/96 examples; another 96 use a
QA template. Object names are disjoint between training, validation, and test.
Every two examples swap the two context colors and form one bootstrap unit.

Capture the learned-prefix donor's predictor and correction outputs. Transplant
these into the ordinary-prefix recipient at aligned token positions. Unselected
correction coordinates are recomputed from the recipient's current hidden states.
Validation selects both channels, excluding prefix positions, for sparse search.

Binary straight-through gates train for 100 updates at a budget of 256 out of
7,546 channel/route coordinates. Each coordinate is shared across all content
positions. Gates minimize full-vocabulary KL from the full-transplant teacher.
The same ranking is evaluated at 16, 64, 256 and 1,024 coordinates; the other
budgets are not independently optimized. Three random masks per budget preserve
channel/layer/stream counts.

| Intervention | Test true-token Δlogp | Test 7-color accuracy | QA Δlogp | QA 7-color accuracy |
|---|---:|---:|---:|---:|
| Baseline | — | 53.13% | — | 32.29% |
| Soft prefix | +0.522 | 53.13% | +1.263 | 39.58% |
| Full predictor, content | +0.159 | 50.00% | −0.038 | 31.25% |
| Full correction, content | +0.818 | 50.00% | +0.482 | 32.29% |
| Both channels, content | +0.847 | 48.96% | +0.676 | 31.25% |
| Learned 256 (3.39%) | +0.483 | 52.08% | +0.303 | 32.29% |
| Matched random 256, mean of 3 | +0.134 | 53.82% | −0.021 | 31.94% |
| Learned 1,024 (13.57%) | +0.932 | 52.08% | +0.332 | 32.29% |
| Matched random 1,024, mean of 3 | +0.297 | 51.04% | +0.367 | 31.25% |

The 256-coordinate mask retains 57.1% of the full transplantation's test Δlogp,
but this is not 57.1% recovery of semantic accuracy. All QA arms have zero
full-vocabulary top-1 accuracy.

An independent diagnostic exactly reproduced the baseline and selected prefix
log probabilities. Of learned-256's QA improvement of 0.30266 nats, 0.29199 nats
(96.5%) comes from increased total probability on the seven color words.
Its change in conditional probability of the correct one of the two context
colors is −0.00052, pair-bootstrap 95% CI [−0.00281, 0.00172]. Swapping donor
routes between the paired color-swapped contexts also produces no reliable
movement toward the donor answer: +0.00038, CI [−0.00072, 0.00154]. This tests
value changes, not a query-identity swap; absence of this effect alone would not
rule out a content-invariant retrieval mechanism.

On a small ordinary-text check (8 WikiText windows × 128 tokens), baseline NLL
is 4.52778, soft-prefix NLL 4.66691, and learned-256 NLL 4.52411. This check is
too small to establish general language-model quality preservation.

## Second experiment: explicitly distinguish the two context colors

This follow-up was designed after inspecting the first experiment. It is not an
independent replication of that protocol. Use a new seed, 16 training objects,
8 validation objects, and 8 previously unused test objects. Train/validation/test
sizes are 256/96/192; a further 192 examples use a new painted-object template.
Optimize `softplus(logit_distractor − logit_true) + 0.1 × full-vocabulary CE`
for 200 updates. Select using validation log probability conditional on the two
context colors. The final step is selected.

| Intervention | Test two-color accuracy | Test binding Δlogp | New-template two-color accuracy |
|---|---:|---:|---:|
| Baseline | 46.88% | — | 46.35% |
| Soft prefix | 76.56% | +0.2358 | 44.27% |
| Full predictor, content | 47.40% | −0.0277 | 45.31% |
| Full correction, content | 50.00% | −0.0314 | 46.88% |
| Both channels, content | 50.00% | −0.0926 | 46.88% |

The soft prefix improves entity-disjoint, same-template binding accuracy by
29.69 points, 95% CI [22.92, 35.95], but fails to generalize to the new template.
Most importantly, none of the three full content-routing transplants passes the
prespecified validation binding-logp gain threshold of +0.02 nats. Sparse search
is therefore **not run** in this experiment. Soft-prefix ordinary-text NLL is
4.62111, versus the same 4.52778 baseline.

The input-optimization step can elicit behavior. The tested routing coefficients
are insufficient to transplant the more specific behavior. This does not rule
out a circuit involving attention/value activations or other inputs, and it does
not establish that direct routing optimization would behave the same way.

## Files and reproduction

- [First run](300m_seed20260928/results.json): per-example scores, examples,
  gate rankings, selected edges, validation histories, and random controls.
- [Independent diagnosis](300m_seed20260928/diagnostics.json): two-color metrics
  and swapped-donor intervention, without further fitting.
- [Binding follow-up](300m_binding_seed20260929/results.json).
- `soft_prefix.pt` in each run directory: saved prefix; ignored by Git, retained
  locally. The training commands regenerate it from the same checkpoint.

```bash
export CUDA_VISIBLE_DEVICES=2
PY=/scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python
$PY scripts/semantic_routing_pilot.py \
  --out experiments/results/semantic_routing_pilot_20260928/300m_seed20260928
$PY scripts/semantic_routing_diagnostics.py \
  --run experiments/results/semantic_routing_pilot_20260928/300m_seed20260928
$PY scripts/semantic_routing_pilot.py \
  --out experiments/results/semantic_routing_pilot_20260928/300m_binding_seed20260929 \
  --seed 20260929 --objective binding --data-version 2 \
  --prompt-steps 200 --gate-steps 200
```

Run from the repository root. Default checkpoint:
`checkpoints/pr_sync_20260917/300m-dagformer`; tokenizer and WikiText caches are
in the adjacent `tokenizer` and `eval_corpora` directories. The original model
code revision was `8a8cbb21a724d6d97c8f98759059554785e39bf9`; experiment scripts
were new working-tree files during execution. No PR #3 architecture was used.
GPU: local RTX A6000. Results record computation timestamps excluding model
loading. This is one prefix seed per exploratory protocol, not a multi-seed
benchmark. Reported CIs resample color-swap pairs, not training seeds.
