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
  and [merged-PR evidence audit](experiments/PROJECT_SYNC_2026-09-17.md).

## Data and checkpoints on Delta

The working project is `/projects/bfqt/users/yurenh2/ml-projects/DAGFormer`.
Use `ssh delta` to access it. Trained checkpoints and cached interpretation data
remain there; they are not stored in Git.

The collaborator model bundle is `/work/hdd/bfqt/shared/dagformer-models`:
75M/150M/300M baseline–DAGFormer pairs and a 600M baseline, with configs and
tokenizer. The 300M shared pair is **not step-matched**: DAGFormer step 9000
and baseline step 12000. The project checkpoint directory also retains a
step-9000 baseline for a matched comparison and a step-10500 DAGFormer.

The mmap runs use `/work/hdd/bfqt/data/pretok/dolma_v1_7_12b`. This is the
available training slice, not the number of tokens every checkpoint consumed.
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
