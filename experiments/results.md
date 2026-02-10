# DAGFormer Experiment Results

## Sanity Checks

### S0 — Dense Baseline (no predictor)

| Item | Value |
|------|-------|
| Status | **DONE** (from sanity training eval) |
| Date | 2025-02-09 |
| Job ID | 15785016 |
| Hardware | A40×1 |
| Eval set | skip=10000, size=50, seq_len=1024 |
| **NLL_base** | **2.4569** |
| Notes | All experiments must beat this. Consider re-running with eval_size=1000 for more robust estimate. |

---

### S1 — Predictor identity init (constant tau=5, ~10M tokens)

| Item | Value |
|------|-------|
| Status | **DONE** |
| Date | 2026-02-09 |
| Job ID | 15788145 |
| Config | r=32, tau=5→5 (constant), k=5, lambda=0 |
| Tokens | ~10M (2500 steps @ batch=4, seq=1024) |
| Hardware | A40×1 (gpub073) |
| Wall time | ~2 hrs |
| Target | NLL ≈ NLL_base (within 1%) |
| Purpose | Verify init reproduces dense topology |
| **Result** | **PASS** — NLL within 0.3% of baseline |

| Metric | Value (final) |
|--------|---------------|
| eval/nll_soft | **2.4500** (baseline: 2.4569, diff: -0.3%) |
| eval/nll_hard | **2.4506** (diff: -0.3%) |
| eval/nll_baseline | 2.4569 |
| topology/mean_A | 0.975 |
| topology/seq_gate_frac | 0.986 |
| topology/hyp_gate_frac | 0.988 |

**Per-eval-step data:**

| Step | nll_soft | nll_hard | nll_base | mean_A |
|------|----------|----------|----------|--------|
| 100 | 2.4531 | 2.4512 | 2.4569 | 0.970 |
| 500 | 2.4588 | 2.4609 | 2.4569 | 0.974 |
| 1000 | 2.4506 | 2.4506 | 2.4569 | 0.978 |
| 1500 | 2.4562 | 2.4578 | 2.4569 | 0.972 |
| 2000 | 2.4500 | 2.4506 | 2.4569 | 0.978 |
| 2500 | 2.4500 | 2.4506 | 2.4569 | 0.975 |

**Observations:**
- Init NLL matches baseline from step 0 — identity init working correctly
- Step 700 had transient dip (mean_A=0.916, nll_soft=2.496) but recovered — Gumbel noise exploration at high tau
- nll_hard ≈ nll_soft throughout — at tau=5, soft gates ≈ 0.95, so hard threshold (>0) gives similar A

---

### S2 — Gradient flow check (constant tau=2, ~50M tokens)

| Item | Value |
|------|-------|
| Status | **RUNNING** (attempt 2) |
| Config | r=32, tau=2→2 (constant), k=5, lambda=0 |
| Tokens | ~50M (12,500 steps @ batch=4, seq=1024) |
| Hardware | A40×1 |
| Est. Time | ~15 hrs (within 48h limit) |
| Target | NLL < NLL_base (2.4569) |
| Purpose | Lower tau gives sharper gates — does predictor learn useful topology? |

**Attempt 1** — Job 15789537, crashed at step ~1860 (Dolma HTTP range request error)

| Step | nll_soft | nll_hard | nll_baseline | mean_A |
|------|----------|----------|--------------|--------|
| 500 | 2.4581 | 2.4581 | 2.4569 | 0.993 |
| 1000 | 2.4575 | 2.4569 | 2.4569 | 0.999 |
| 1500 | 2.4547 | 2.4559 | 2.4569 | 0.993 |

Observations (attempt 1):
- Eval NLL ≈ baseline throughout — predictor still near init (mean_A ≈ 0.99)
- Train NLL high variance (0.27–2.96) is normal batch-to-batch variation at batch_size=4
- No checkpoint saved (save_every=2500, crashed at 1860)
- Crashed due to Dolma streaming HTTP error, not code bug

**Attempt 2** — Job 15798568 (fresh start, no checkpoint from attempt 1)

| Metric | Value |
|--------|-------|
| eval/nll_soft | |
| eval/nll_hard | |
| topology/mean_A | |

---

## Phase 1 Core

### P1 — Phase 1 default config (5B tokens)

| Item | Value |
|------|-------|
| Status | NOT STARTED |
| Config | r=32, tau=5→0.2 cosine, k=5, lambda=0→0.01 ramp |
| Tokens | 5B |
| Hardware | A40×4 |
| Est. Time | ~4 days |

| Metric | Value |
|--------|-------|
| eval/nll_soft | |
| eval/nll_hard | |
| topology/mean_A | |
| topology/seq_gate_frac | |
| topology/hyp_gate_frac | |

---

### P2 — Phase 1 extended (10B tokens)

| Item | Value |
|------|-------|
| Status | NOT STARTED |
| Config | Continue P1 if still improving at 5B |
| Tokens | 10B |
| Hardware | A40×4 |
| Est. Time | ~7 days |

---

## Ablations

### A1–A4: Rank r

| ID | Rank | NLL_soft | NLL_hard | Sparsity | Notes |
|----|------|----------|----------|----------|-------|
| A1 | 8 | | | | |
| A2 | 16 | | | | |
| P1 | 32 | | | | (reference) |
| A3 | 64 | | | | |
| A4 | 256 | | | | full rank |

### A5–A7: Temperature schedule

| ID | tau_init | tau_final | NLL_soft | NLL_hard | A entropy | Notes |
|----|----------|-----------|----------|----------|-----------|-------|
| A5 | 1 | 1 | | | | constant, perpetually soft |
| P1 | 5 | 0.2 | | | | (reference) |
| A6 | 5 | 0.05 | | | | aggressive anneal |
| A7 | 10 | 1.0 | | | | slow anneal |

### A8–A9: Sparsity lambda

| ID | lambda | NLL_soft | NLL_hard | Density | Notes |
|----|--------|----------|----------|---------|-------|
| A8 | 0 | | | | no sparsity |
| P1 | 0→0.01 | | | | (reference) |
| A9 | 0→0.05 | | | | high sparsity |

### A10–A11: Cascading gate

| ID | Gate | NLL_soft | NLL_hard | Dead heads | Notes |
|----|------|----------|----------|------------|-------|
| A10 | OFF | | | | |
| P1 | k=5 fixed | | | | (reference) |
| A11 | k=5 learnable | | | | |

---

## Analysis Experiments

### X1 — Topology variance analysis
| Item | Value |
|------|-------|
| Status | NOT STARTED |
| Result | |

### X2 — Domain-specific topology
| Item | Value |
|------|-------|
| Status | NOT STARTED |
| Result | |

### X3 — Topology-NLL sensitivity
| Item | Value |
|------|-------|
| Status | NOT STARTED |
| Result | |

---

## Speed Estimates (A40×1, batch=4, micro_batch=2, seq=1024)

| Component | Time | Notes |
|-----------|------|-------|
| Training step | ~3s | Forward + backward + optimizer |
| Eval round (50 samples) | ~2 min | 25 batches × 3 modes (soft/hard/baseline) |
| Model loading | ~10 min | OLMo + Qwen + eval set build |
| 1K steps (no eval) | ~50 min | |
| 1K steps (eval every 100) | ~70 min | 10 eval rounds add ~20 min |
| 10K steps | ~12 hrs | |
| 100K steps | ~5 days | Exceeds 48h SLURM limit, needs auto-resume |

**Previous 14s/step estimate was wrong** — it included model loading and eval overhead in wall-clock average.

---

## Preliminary Data (from sanity training job 15785016)

Run with cascading gate bug (layer 0 not exempted). 500/1000 steps completed before timeout.

| Step | train/nll | eval/nll_soft | eval/nll_hard | eval/nll_baseline | mean_A | tau |
|------|-----------|---------------|---------------|-------------------|--------|-----|
| 0 | 3.539 | — | — | — | 0.417 | 5.00 |
| 100 | 2.750 | 2.635 | 4.744 | 2.457 | 0.416 | 4.88 |
| 200 | 3.102 | 2.630 | 4.570 | 2.457 | 0.416 | 4.54 |
| 300 | 2.844 | 2.621 | 4.680 | 2.457 | 0.418 | 4.01 |
| 400 | 2.492 | 2.641 | 4.893 | 2.457 | 0.419 | 3.34 |
| 500 | 1.805 | 2.639 | 4.503 | 2.457 | 0.419 | 2.60 |

**Key observations:**
- train/nll decreasing (3.54 → 1.80) but eval/nll_soft flat (~2.63) — overfitting or predictor not generalizing
- eval/nll_hard very high (4.5-4.9) due to cascading gate layer 0 bug (now fixed in `80579d6`)
- mean_A stable ~0.42 (= ~0.89 over valid entries), no collapse
- Baseline NLL = 2.4569 confirmed correct after double-shift fix
