# DAGFormer

DAGFormer studies learned, per-token, per-head routing between transformer
layers. The current implementation trains an OLMo-2 backbone from scratch with
separate Q/K/V input mixtures and a shared residual mixture.

## Current architecture

For target layer `l`, the available sources are the token embedding and the
outputs of the preceding layers. Each attention head has its own Q/K/V mixing
coefficients; R coefficients are shared across heads. An external causal
encoder predicts `alpha_pred` from token IDs. In `fourway_corrected` runs,
local MLPs read the backbone hidden states and add `alpha_corr`:

```
input_ids -> causal predictor ----------------> alpha_pred
prior backbone hidden state -> correction MLP -> alpha_corr
                                                |
                         alpha_eff = alpha_pred + alpha_corr
                                                |
                         per-head Q/K/V and shared R mixing
```

The commonly analyzed 300M model has 12 layers, 16 heads and 3,773 routing
coefficients per token. Its coefficients are signed real values, not binary
gates or attention probabilities. The old Qwen/Gumbel adjacency design is
preserved in [the historical specification](docs/legacy_oracle_design.md).

## Code and experiments

- Model: [FourWayDAGFormer](src/model/olmo_graph.py) and
  [FourWayPredictor](src/model/predictor.py).
- Training: [pretrain_dagformer.py](scripts/pretrain_dagformer.py),
  [pretrain_baseline.py](scripts/pretrain_baseline.py), and `configs/*mmap.yaml`.
- Experiment history: [results.md](experiments/results.md), including head and
  connection interventions, context-fidelity cloze, generation and routing SAE.
- Interpretability: `scripts/interp_*.py` contains the original head/connection
  analyses; [experiments/interp](experiments/interp/README.md) contains the
  instruction-contrast circuit suite from PR #2.
- Benchmark evaluation: [lm-eval instructions](experiments/results/lmeval/README.md)
  and [reasoning findings](experiments/results/lmeval/reasoning/FINDINGS.md).
- September 17: [matched ordinary and interpretability evaluations](experiments/results/eval_20260917/README.md)
  and [merged-PR evidence audit](experiments/PROJECT_SYNC_2026-09-17.md);
  [中文验收说明](experiments/results/eval_20260917/验收说明.md) and
  [reproduction commands](experiments/results/eval_20260917/REPRODUCE.md).
- Larger scales: [600M / 1B checkpoint and continuation audit](experiments/SCALING_COMPLETION_2026-09-17.md),
  including confirmation of the 1B per-head FourWay architecture and remaining training budgets.

## Data and checkpoints on Delta

Use `ssh delta` to access the project. The current evaluation workspace is
`/work/hdd/bfqt/yurenh2/dagformer-eval-20260917`, containing the consolidated
code, exported evaluation weights and new raw results.

The recovered source worktree remains at
`/projects/bfqt/users/yurenh2/ml-projects/DAGFormer`. Its original trained
checkpoints, optimizer states and cached interpretation data are preserved;
they are not stored in Git. The September 17 campaign uses the new workspace
because the original `/projects/bfqt` quota is full.

The collaborator model bundle is `/work/hdd/bfqt/shared/dagformer-models`:
75M/150M/300M baseline–DAGFormer pairs and a 600M baseline, with configs and
tokenizer. The 300M shared pair is **not step-matched**: DAGFormer step 9000
and baseline step 12000. The project checkpoint directory also retains a
step-9000 baseline for a matched comparison and a step-10500 DAGFormer.

The mmap training configs point to `/work/hdd/bfqt/data/pretok/dolma_v1_7_12b`,
a historical 12B-token slice. That path was absent when checked on September
17; the checkpoints and evaluation caches used in the current campaign are
available separately. The slice size is not the number of tokens every
checkpoint consumed.
Older streamed-data runs, including the exports under
`/work/hdd/bfqt/dagformer_checkpoints/hf`, are a separate comparison.

## Local checks

```bash
python -m pytest tests -q
```

The original unit suite covers binary-gate components. Passing it does not
reproduce the trained FourWay checkpoint results; use the evaluation scripts
with the recorded checkpoint, configuration and corpus for that purpose.
Additional tests cover FourWay checkpoint variants, simultaneous head edits,
and paired evaluation statistics.
