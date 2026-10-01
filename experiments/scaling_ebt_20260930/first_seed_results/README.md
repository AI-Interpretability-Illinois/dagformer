# First-seed results

Four runs completed by 2026-10-01 14:21 UTC. All use width 512, seed 42,
3,000 optimizer updates and 1,572,864,000 training tokens from the new Dolma
split described in [EXECUTION.md](../EXECUTION.md). Values below are final
token-weighted NLL in nats per token; lower is better.

| Layers | Model | Total parameters | Dolma heldout | WikiText2 | MathInstruct | GSM8K text |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 4 | Dense | 68,166,144 | 4.815648 | 5.349737 | 4.154727 | 4.716941 |
| 4 | DAGFormer | 96,957,281 | 4.608517 | 5.018963 | 3.796538 | 4.370693 |
| 6 | Dense | 76,558,848 | 4.876487 | 5.458102 | 4.259972 | 4.837791 |
| 6 | DAGFormer | 105,657,332 | 4.487872 | 4.850497 | 3.650345 | 4.177075 |

DAGFormer has lower loss at both tested depths. Increasing depth from four
to six layers reduces DAGFormer's loss on all four caches; Dense loss rises
slightly. These are two depth points at one seed, so they do not establish a
scaling exponent. Predictor parameters are included in the table. This sweep
holds backbone width and token budget fixed; total parameters change with
depth and differ between the two architectures.

The [depth diagnostic](depth_diagnostic.json) records an unresolved training
precision issue found on 2026-10-01. The Dense backbone parameters and Adam
moments are BF16. All 8,704 normalization weights in the four-layer checkpoint
and all 12,800 in the six-layer checkpoint still equal their initial value,
exactly 1.0, after 3,000 updates. The inspected six-layer DAGFormer backbone
has the same property; its predictor's Adam moments are FP32. Both Dense
training loss and monitoring loss are higher at six layers throughout the
recorded training. The depth reversal cannot yet be attributed to architecture:
a comparison preserving FP32 weights and optimizer states is needed to separate
optimization and precision effects. The stored results remain measurements of
the original BF16 training recipe.

Training and evaluation used revision `f036b92`. DAGFormer uses the complete
causal encoder predictor and correction MLPs, with V normalization disabled.
The 512-window Dolma split is held out from these new runs. WikiText2,
MathInstruct and GSM8K use the merged PR's common caches. The mathematical
datasets measure text likelihood. These new runs use a different corpus
setup from the earlier small-model ladder and remain a separate plot series.

Each run directory contains its final `evaluation.json`, per-window loss sums
and token counts in `evaluation.npz`, and `training_settings.json`. The latter
records the actual trainer configuration with machine-specific storage paths
omitted. Generate runnable paths with `scripts/prepare_six_axis_runs.py`.
Checkpoint filenames and cache basenames are retained for provenance; model
weights and evaluation caches are stored outside Git.
