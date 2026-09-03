# DAGFormer — Code/Design Review & Improvement Plan (2026-07-08)

Context: returning after 21 days offline. All recent SLURM jobs had died; this
doc distills a full review of the model + trainer code (two deep review passes)
into prioritized, actionable fixes, plus the forward experiment plan. The prior
code was written by a weaker model; this is the cleanup + upgrade pass.

---

## 0. What killed every recent run (root-cause post-mortem)

| Job | Outcome | Progress | Root cause |
|---|---|---|---|
| 600M baseline (19272481) | FAILED @7h55m | step 14000/22900 (65%) — resumable | NCCL 600s ALLREDUCE timeout (data stall) |
| 600M FourWay (19272482) | FAILED @20h22m | step 14000/22900 (65%) — resumable | same |
| 1B smoke baseline (19272709) | FAILED @32m | none | NCCL watchdog hang (480s) |
| 1B smoke 4way (19272710) | FAILED @24s | none | config typo `lr_decay_steps` |

**Two root causes, both "weak-AI" artifacts:**

1. **Train-time HF streaming + per-step tokenization in the main process.** A
   silent network stall (no exception — just no bytes) blocks `for doc in
   dataset` forever on one rank; other ranks wait at the DDP all-reduce; after
   the default ~10-min NCCL timeout everyone SIGABRTs. `MAX_RETRIES` only catches
   exceptions, never a hang. Compounded by: `init_process_group` with no
   `timeout`; `manual_shard` making all 8 ranks download+tokenize everything and
   discard 7/8 (8× redundant, 8× HF pressure → rate-limit → stall); `num_workers=0`
   (no prefetch, data blocks the training thread).
2. **Config dataclass drift** — baseline's `TrainConfig` has `lr_decay_steps`,
   dagformer's `DAGFormerPretrainConfig` did not → instant crash.

---

## 1. Robustness fixes — status

| Fix | Status | Files |
|---|---|---|
| `lr_decay_steps` added to dagformer config + wired into `get_lr`/`get_predictor_lr` | ✅ DONE | pretrain_dagformer.py |
| Atomic checkpoint save (`.tmp`→fsync→`os.replace`) | ✅ DONE (both trainers) | pretrain_{dagformer,baseline}.py |
| NCCL PG timeout 30 min on `init_process_group` | ✅ DONE (both) | " |
| Trap SIGTERM (not just SIGUSR1) so SLURM preemption always saves | ✅ DONE (both) | " |
| **Pre-tokenized mmap data pipeline** (the real stall fix) | 🔧 IN PROGRESS | new: scripts/pretokenize.py, src/data/mmap_dataset.py |
| Resume-guard metadata (world_size/micro/accum/seed) + RNG state in ckpt | ⏭ folds into mmap wiring | " |
| Explicit `data_samples_consumed` counter in ckpt (O(1) resume offset) | ⏭ folds into mmap wiring | " |

The mmap pipeline supersedes the interim "stall watchdog" idea: with local
memmap there is no network in the step path, so stalls become impossible and
resume becomes a single integer offset (fixes the streaming-resume correctness
bugs below for free).

### Resume correctness bugs the mmap pipeline fixes (from the trainer review)
- **olmo_mix/dolmino resume is outright wrong**: manual modulo shard + a
  non-deterministic `interleave_datasets` (no seed passed) means a resumed run
  sees a *different* data mixture and re-counts; and all 8 ranks must
  re-download+re-tokenize to the resume point → hours of stall → can NCCL-timeout
  the resume itself. **⇒ streaming 1B-20B cannot be safely resumed at all.**
- **v1_7 resume is arithmetically correct only if world_size/micro/accum are
  byte-identical** to the original run, and still requires re-streaming 65% of
  data before the first new step (impractical). ⇒ resume the 600M runs by
  restoring model/opt/step from the step-14000 checkpoint but reading **fresh
  mmap data** (`data_samples_consumed=0`), not by re-skipping the stream.

---

## 2. Model / algorithm findings (FourWay routing)

Live path: `FourWayPredictor` → `predict_fourway_routing` → routing regularizers
→ `FourWayDAGFormer.forward`. ~60% of predictor.py / olmo_graph.py is dead
legacy (adjacency-matrix + Qwen + Gumbel design) — candidate to move to
`src/model/legacy/`.

### HIGH — identity-init is silently destroyed by several knobs
- `softmax_rv` / `softmax` / `temperature` / `sinkhorn` / `row_col`:
  `softmax([0,…,0,1]) ≠ [0,…,0,1]`, so enabling any of them makes step-0 ≠
  vanilla OLMo-2, violating the project's core guarantee. **This concretely
  vindicates the owner's dislike of softmax_rv** — it's not just "hard to
  analyze", it breaks init. If a normalized variant is ever wanted, re-derive
  identity in *logit* space (large + logit on the identity slot).
- `routing_l2_lambda` / `routing_l1_lambda` penalize the identity slot's `1.0`,
  i.e. they pull α toward all-zeros ("read nothing"), the **opposite** of the
  documented "stay near vanilla". Must penalize `(α − identity)`, not `α`.

### HIGH — V is the runaway stream; R is even more load-bearing
- V mixing = per-head per-token weighted sum of projected V over source layers
  (`olmo_graph.py` ~879). `use_v_norm` applies a global 2048-dim RMSNorm after
  the mix — it controls *global* RMS but not per-head / per-source imbalance, and
  it is **not identity-preserving** (vanilla OLMo has no V-norm), so it slightly
  shifts init.
- R is the **entire residual carrier** (no separate `residual = x`); it doubles
  as router + mandatory skip. If α_r ever down-weights the most-recent layer, the
  residual is attenuated → destabilizes the base model more than V.

**Recommended stabilizers (analyzable, non-softmax — matches owner's preference):**
1. **Identity-anchored residual reparam** for V (and R): `V = V_last + Δ`,
   `Δ = Σ α_v[l']·(V_{l'} − V_last)` with the identity slot excluded. Runaway
   becomes literally `‖Δ‖` (measurable), enables a *correct* `Δ.pow(2)` penalty,
   and preserves identity init exactly.
2. **Per-head soft RMS clamp** instead of global RMSNorm: scale V by
   `min(1, c·rms_ref/rms(v))` per head — identity-preserving in-range, only bites
   outliers, catches head-level imbalance the global norm misses.
3. Cheap extras: `relu(‖α_v‖₁ − 1)` net-amplification penalty (zero at identity);
   per-stream (V) lower LR / higher weight-decay; per-stream delayed unfreeze.

### Prune the routing-reg surface
Keep (identity-safe, principled): `clamp`, `noise`, `dropout`, `top_k`.
Fix or drop: `l2`/`l1` (identity bug), `entropy` (regularizes a softmax the
forward never applies), `softmax*`/`sinkhorn`/`row_col`/`temperature` (break
identity). This is a lot of surface that "looks like science" but perturbs the
one invariant the project relies on.

### Efficiency / correctness misc
- **Triton kernel has no backward** → enabling `use_triton_kernel` silently cuts
  routing gradients. Keep OFF (or wrap in an autograd.Function). The einsum
  "project-then-mix" path is already the good design (fused QKV + SDPA).
- Dead compute: a `-inf` causal mask is built every layer but unused (SDPA uses
  `is_causal=True`); growing Python `layer_outputs` list is torch.compile-hostile.
- Missing shape asserts on `FourWayDAGFormer.forward` hot path (CLAUDE.md §13
  mandates them); the `sparsity` wandb metric actually logs routing-reg loss.
- Q/K per-head-per-source routing is a large DoF sink (q_norm/k_norm already
  normalize, softmax already bounds attention) — candidate to share Q/K across
  heads and reserve per-head routing for V/R (the `alpha_share_heads` diagnostic
  exists for exactly this suspicion).

---

## 3. Forward experiment plan (all runs use the new mmap pipeline)

1. **1B 20B-token fair head-to-head** (headline for resource apps): 1B baseline +
   1B FourWay on olmo-mix-1124 stage-1 mixture, vs AI2 `OLMo-2-0425-1B`
   @ `stage1-step10000-tokens21B`. 8×H200. Requires 8-rank smoke first.
2. **Finish 600M** baseline + FourWay (Dolma v1.7) from step 14000 → 22900 on
   fresh mmap data. Completes the 75M/150M/300M/600M/1B scaling-law figure.
3. **V-stabilizer ablation** (small, 75–150M, A40): current vnorm vs
   identity-anchored residual-Δ V vs per-head soft clamp. Cheap; strengthens the
   method before the next scale-up.

Autonomy: a GPU idle-watcher auto-submits the next queued run when a partition
frees up and auto-resubmits preempted jobs from the latest checkpoint.

### Invariants to keep (from CLAUDE.md + owner feedback)
- Never run load_dataset / HF downloads / model loads on the login node — SLURM only.
- Always run an 8-rank smoke before any >4h job.
- Checkpoints to /work/hdd (not /projects, which is near-full).
- Benchmark tables compare **baseline vs DAGFormer only** (exclude MUDDFormer).
- Use `/usr/bin/python3.9` for HF/wandb/lm_eval.
