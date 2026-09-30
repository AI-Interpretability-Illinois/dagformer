# Six-axis execution

PR #3 is merged into `main` at `1919dbd`. The first seed contains 14 distinct
settings for each of Dense and FourWay DAGFormer, with the complete encoder
predictor and correction MLPs. The shared center is L6 / d512 / 1.572864B tokens /
512 sequences per update. The architecture and budgets are in `six_axis_plan.json`.

The parameter and FLOP axes use the existing parameter ladder, with common-cache
evaluation added for the completed 600M and 1B checkpoints. Runs from different
training corpora remain separate series. The new depth, width, data and batch
axes use one fixed 21B Dolma corpus and seed 42. The depth sweep fixes width;
the width sweep fixes depth. Neither is a fixed-total-parameter DepthBench sweep.

## Data and evaluation

`scripts/prepare_six_axis_data.py` excludes a complete-document suffix from the
last shard of each of the eight tokenization workers. Source `.bin` files remain
unchanged. The new training index contains 20,983,768,178 tokens. The held-out
cache has 64 windows per worker, 512 windows total, and a fixed 128-window subset
is used for training monitoring. Worker strata approximate the mixture; they
are not verified source-domain labels.

The new split is held out for new training only. Original-index checkpoints
can have consumed its documents. Existing ladder comparisons use the same
WikiText2, MathInstruct and GSM8K caches as the merged PR. The two mathematical
datasets are likelihood measurements, not generated-answer accuracy.

Final evaluation uses token-weighted FP32 cross entropy, saves per-window
losses, and counts the complete model including the predictor. Training FLOPs
use the PR's analytic projection/attention/MLP estimate and include routing
overhead. Allocated GPU time is recorded separately by the runner.

## Execution

`scripts/prepare_six_axis_runs.py` creates all 28 YAML configurations and a
manifest. `scripts/run_six_axis.py --manifest ... --task N` trains, resumes,
evaluates and records completion of one setting. `--tasks 0,1` runs a local
sequence. Slurm uses `scripts/slurm/six_axis.slurm` with `SIX_CODE`, `SIX_HOME`,
`SIX_PYTHON` and `SIX_MANIFEST` set at submission.

Small files (code, configs, caches, CSV logs and evaluation outputs) go under
the user's home. Delta checkpoints go on biro storage. The trainer keeps one
checkpoint group per run, saves atomically, and checks for a stop request after
an update. Four-hour jobs checkpoint before their deadline and requeue. New
checkpoints explicitly record completed updates; final evaluation checks both
that count and Adam's step count against the planned token budget.

The token stream uses zero DataLoader workers, seed 42 and fixed block order.
Global batch is implemented by accumulation and warmup is fixed at 104,857,600
tokens. Every data-budget run uses its own linear decay endpoint. A new seed
is a separate run; a restarted job continues its previous optimizer and stream.

`scripts/dispatch_six_axis_local.py` runs two local workers while the other
settings remain queued on Delta. When a local worker becomes available, it can
hold and cancel one never-started pending task from this study's array, then run
that setting locally. Running tasks and checkpointed requeues remain on Delta.
The dispatcher synchronizes
small result files and regenerates the six-panel progress figure every five
minutes. Completed local checkpoint groups are copied to the same biro study
directory as the Delta runs, with the local copies retained. A `STOP_DISPATCH`
file stops dispatch and checkpoints active local work.

The initial local center runs use GPUs 3 and 0. GPU 0 receives only the smaller
routed shapes because another resident process reserves part of its memory.
The DAGFormer run migrated after update 27; both checkpoint metadata and Adam
state recorded 27, and resumption skipped exactly 13,824 consumed samples.

Related validation: 45 merged-PR tests passed; three focused execution tests
cover held-out separation, sample resume offsets and checkpoint update counts.
All 28 generated configurations were accepted by their trainer dataclasses,
with exact parameter counts, token budgets and warmup budgets checked.
