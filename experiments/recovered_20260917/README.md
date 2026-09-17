# Pre-existing timan1 stash

`local_stash_9901429.patch` preserves the complete patch from the April 26,
2026 stash `9901429f2d03bcf7327a51977d14bdd6485764dc`. This stash predated the
September synchronization and was separate from the recovered working-tree
changes. The original stash remains available locally.

It is archived rather than applied as a whole:

| Patch component | Current disposition |
|---|---|
| Resume token counter and explicit iterator skipping | The current loader already resumes using per-rank packed-sample counts and global mmap offsets. Applying the old iterator loop as well would skip data twice; it also treats batched iterator steps as individual sequences. |
| Streaming retry position | The current dataset tracks consumed documents and resumes after them, with sharding handled before the retry counter. |
| Dense warmup branch through the DDP wrapper | The patch addresses a legacy non-FourWay branch that still calls the inner base model. The active FourWay training path already calls its DDP wrapper. No legacy training was launched in this evaluation campaign. |
| Eval-cache batch-count check | The old patch deletes a mismatched cache before rebuilding. This campaign reads explicitly selected caches and verifies their sequence counts/shapes; it does not modify the training cache loader. |
| Disable predictor MHA fastpath | Kept as an explicit numerical control in `scripts/eval_predictor_fastpath.py`, without changing model defaults during the benchmark campaign. Results are written under `results/eval_20260917/routing_dependence/fastpath_*.json`. |

The archived patch is historical source, not an additional required patch for
reproducing the current results. In particular, applying it on top of the
current resume implementation would change training data consumption.
