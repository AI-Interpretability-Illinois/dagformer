# Pretraining settings and loss curves — 2026-09-28

Delta shared directory:
`/work/hdd/bfqt/yurenh2/pretraining_handoff_20260928/`

This package covers the **completed 1B and 600M pairs**. September work resumed
the existing runs with their optimizer states and original LR schedules; it was
not a new from-scratch 75M/300M campaign.

Start with:

- `configs/*_actual_resume.yaml`: exact generated settings used for continuation.
- `configs/*_final_checkpoint.yaml`: configs attached to the completed checkpoints.
- `curves/loss_curves.png`: full training curves, dense and DAGFormer side by side.
- `curves/loss_curves_after_warmup.png`: same data after 0.5B tokens for a closer view.
- `curves/*_train.csv`: stitched, machine-readable curves with raw NLL, LR, tokens,
  phase, and source-file columns. `curves/summary.csv` records endpoints.
- `raw/*_metrics.csv`: unchanged original and continuation metrics.
- `manifest.json`: original file paths, actual optimizer-update completion proofs,
  Slurm completion times, and curve-stitching rules.

## Settings

| Setting | 1B dense | 1B FourWay | 600M dense | 600M FourWay |
|---|---:|---:|---:|---:|
| Layers | 16 | 16 | 14 | 14 |
| Hidden dimension | 2,048 | 2,048 | 1,536 | 1,536 |
| Attention heads | 16 | 16 | 16 | 16 |
| MLP intermediate dimension | 8,192 | 8,192 | 6,144 | 6,144 |
| Training data | OLMo-mix-1124 | OLMo-mix-1124 | Dolma v1.7 | Dolma v1.7 |
| Sequence length | 1,024 | 1,024 | 1,024 | 1,024 |
| GPUs | 8 | 8 | 8 | 8 |
| Microbatch per GPU | 16 | 4 | 8 | 4 |
| Gradient accumulation | 4 | 16 | 8 | 16 |
| Global tokens / update | 524,288 | 524,288 | 524,288 | 524,288 |
| Backbone peak LR | 4e-4 | 4e-4 | 4.5e-4 | 4.5e-4 |
| Predictor peak LR | — | 2e-4 | — | 2.5e-4 |
| Warmup updates | 1,000 | 1,000 | 500 | 500 |
| Final optimizer updates | 38,160 | 38,160 | 22,900 | 22,900 |
| Final processed tokens | 20.006830080B | 20.006830080B | 12.006195200B | 12.006195200B |

Common settings: seed 42, AdamW betas (0.9, 0.95), weight decay 0.1, gradient
clipping 1.0, linear LR decay, tied embeddings, vocabulary 100,352,
`allenai/OLMo-2-0425-1B` tokenizer. September continuations used 8×H200.

The architecture is **`fourway_corrected`**, with an external causal encoder
predictor (dimension 256, 2 layers, 4 encoder attention heads, trunk 512), plus
per-layer correction MLPs (hidden 128). Q/K/V coefficients are per head and token;
R coefficients are shared across attention heads and vary per token. Sources
are the embedding and earlier complete layer outputs. Both 1B and 600M use V
post-mix normalization (`use_v_norm: true`).

This is the existing layer-output routing implementation, not PR #3's newer
module-granularity graph. Its MLP input and final readout do not have the
independent `m` / `o` routing channels introduced in that variant. Match this
distinction when comparing pretraining or pruning variants.

## Reading the curves

Loss is **training minibatch NLL in nats/token**, not validation or WikiText NLL.
Dense logs name it `train/loss`; FourWay logs name it `train/nll`. Faint lines are
raw batches; solid lines are a trailing mean of 21 logged batches, computed
separately within each historical/continuation phase. Raw CSVs are unchanged.

Historical tails after the recovered checkpoints are excluded from the stitched
curves: keep 1B DAG steps through 31,000, and both 600M steps through 14,000, then
append continuation logs. Those checkpoints had 31,001 / 14,001 actual optimizer
updates. Vertical dashed lines mark continuation at 16.253452288B / 7.340556288B
tokens. The old April `1b_dagformer_stitched_train.csv` elsewhere in the repository
belongs to a different ~5B Dolma run and is not used here.

The 1B cache was reconstructed after the original cache disappeared. The reader's
actual legacy ordering and document-selection behavior were reproduced, but
historical decompression failures prevent claiming exact identity of every old
and reconstructed training token. A visible loss jump at the continuation
boundary should be interpreted with that data transition in mind. The nominal
OLMo mix was processed in source order by the legacy reader; these curves should
not be interpreted as stationary loss on a probability-interleaved mixture.

The 600M runs resumed on the same frozen nominal Dolma suffix. Their earlier
streaming histories had different HTTP failures, so the full histories are not
known to have seen identical examples. The matched budget and shared continuation
data are verified. Do not compare the absolute 1B and 600M training-loss scales as
an architecture effect: the datasets differ.

Final checkpoint update counts are verified separately from logging: the last
logged steps are 38,150 / 22,890, at 20.002111488B / 12.001476608B tokens. There is
no invented loss point at the final checkpoint. These are single minibatch
endpoints; use the full curve and common held-out eval for quality comparison.

## Reproduction and checkpoint locations

From a Python environment with NumPy, pandas and Matplotlib:

```bash
python plot_curves.py
```

Continuation code revisions: `cb24428` for 1B and `6cfdc7b` for 600M, with 600M
data preparation at `2586e43`, in `AI-Interpretability-Illinois/dagformer`.
The manifest records exact source paths for configs and CSVs.

Completed evaluation checkpoints remain under:

```text
/work/hdd/biro/yurenh2/dagformer-20260917/checkpoints/eval/1b-baseline/
/work/hdd/biro/yurenh2/dagformer-20260917/checkpoints/eval/1b-dagformer/
/work/hdd/biro/yurenh2/dagformer-20260917/checkpoints/eval/600m-baseline/
/work/hdd/biro/yurenh2/dagformer-20260917/checkpoints/eval/600m-dagformer/
```

Large checkpoint files are not duplicated in this sharing directory. Access to
those separate checkpoint directories depends on their own project permissions;
the settings, original loss CSVs, and plots here are readable by `delta_bfqt`.
