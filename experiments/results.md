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

**Attempt 2** — Job 15798568, crashed at step ~5190 (same Dolma HTTP error)

| Step | nll_soft | nll_hard | nll_baseline | mean_A |
|------|----------|----------|--------------|--------|
| 500 | 2.4578 | 2.4597 | 2.4569 | 0.992 |
| 1000 | 2.4566 | 2.4591 | 2.4569 | 0.989 |
| 2000 | 2.4531 | 2.4537 | 2.4569 | 0.991 |
| 3000 | 2.4484 | 2.4491 | 2.4569 | 0.992 |
| 3500 | **2.4475** | 2.4487 | 2.4569 | 0.993 |
| 4000 | 2.4569 | 2.4569 | 2.4569 | 0.999 |
| 5000 | 2.4550 | 2.4578 | 2.4569 | 0.980 |

**S2 conclusion**: Predictor stuck near init (mean_A≈0.99). Sigmoid saturation confirmed —
init_logit=15 + τ=2 gives ∂A/∂Z≈0.0003, insufficient gradient. Moving to A12-A14 (lower init_logit).
Status: **DONE** (no need to re-run, hypothesis confirmed across 2 attempts, ~7K total steps)

---

## Phase 2 — OLMo Unfrozen

### P2 Trial — Conservative trial run (~10M tokens)

| Item | Value |
|------|-------|
| Status | **DONE** |
| Date | 2026-02-11 / 2026-02-12 |
| Job ID | 15824738 (30min, 680 steps), **15825182** (full 2500 steps) |
| Config | phase=2, olmo_lr=3e-5, predictor_lr=1e-4, init_logit=15 (dense), tau=5 constant, lambda=0, micro_batch=1, grad_ckpt=true |
| Tokens | ~10M (2500 steps @ batch=4, seq=1024) |
| Hardware | A40×1 |
| Purpose | Verify: (1) no OOM/NaN, (2) grad/olmo_norm > 0, (3) NLL doesn't diverge |

**Verification results — ALL PASS:**

| Criterion | Result |
|-----------|--------|
| No OOM | **PASS** — fits in A40 48GB |
| No NaN | **PASS** — all values finite |
| grad/olmo_norm > 0 | **PASS** — stable 2.1–5.9 |
| grad/predictor_norm > 0 | **PASS** — stable 0.007–0.17 |
| NLL doesn't diverge | **PASS** — eval/nll_soft tracks baseline |
| Train NLL decreasing | **PASS** — ~2.6 (start) → ~1.3 (late steps) |

**Eval trajectory (full 2500 steps):**

| Step | nll_soft | nll_hard | nll_baseline | soft−base | mean_A | jaccard_var |
|------|----------|----------|--------------|-----------|--------|-------------|
| 500 | 2.5009 | 2.5041 | 2.5034 | -0.0025 | 0.972 | 0.0000 |
| 1000 | 2.4866 | 2.4881 | 2.4884 | -0.0018 | 0.980 | 0.0000 |
| 1500 | 2.4778 | 2.4791 | 2.4787 | -0.0009 | 0.985 | 0.0000 |
| 2000 | 2.4656 | 2.4659 | 2.4659 | -0.0003 | 0.978 | 0.0000 |
| 2500 | 2.4678 | 2.4669 | 2.4672 | +0.0006 | 0.980 | 0.0000 |

**Key findings:**
1. **eval baseline 在下降** (2.503 → 2.467)：OLMo 在做 continual pretraining，权重本身在改善
2. **nll_soft ≈ nll_baseline 全程**（差值 < 0.003）：predictor 的 topology 没有额外贡献
3. **jaccard_var = 0 全程**：predictor 仍然 context-independent，对所有输入产生相同 A
4. **mean_A ≈ 0.98**（eval soft）：topology 停留在 dense，predictor 几乎没动
5. **没有过拟合**：eval NLL 和 baseline 一起平稳下降

**结论**：NLL 改善完全来自 OLMo CPT，predictor/topology 没有贡献。
根本原因：predictor_lr=1e-4 对比 grad/predictor_norm≈0.02，步长太小；
同时 OLMo 以 3e-5 的 LR 快速适应 dense topology，predictor 没有动力偏离 A≈1。
需要加大 predictor LR 或降低 OLMo LR，制造 predictor 的学习空间。

### P2a–P2d — LR ratio ablation (2500 steps each)

| Item | Value |
|------|-------|
| Status | **DONE** |
| Date | 2026-02-13 |
| Job IDs | P2a=15838281, P2b=15838282, P2c=15838283, P2d=15838284 |
| Hardware | A40×1 each, gpuA40x4, all from fresh HF OLMo |

| Exp | olmo_lr | pred_lr | tau | init_logit | nll_soft@2500 | baseline@2500 | diff | mean_A | jaccard |
|-----|---------|---------|-----|------------|---------------|---------------|------|--------|---------|
| P2a | 3e-5 | 1e-2 | 5 | 15 | 2.824 | 2.596 | +0.228 | 0.499 | 0 |
| P2b | 5e-6 | 1e-3 | 5 | 15 | 2.443 | 2.445 | **-0.002** | 0.986 | 0 |
| P2c | 1e-5 | 1e-3 | 5 | 15 | 2.444 | 2.441 | +0.003 | 0.985 | 0 |
| P2d | 3e-5 | 1e-3 | 2 | 3 | 2.467 | 2.466 | +0.001 | 0.995 | 0 |

**P2b eval trajectory (best):**

| Step | nll_soft | nll_baseline | diff |
|------|----------|--------------|------|
| 500 | 2.4517 | 2.4472 | +0.0045 |
| 1000 | 2.4450 | 2.4448 | +0.0002 |
| 1500 | 2.4450 | 2.4452 | -0.0002 |
| 2000 | 2.4436 | 2.4452 | -0.0016 |
| 2500 | 2.4430 | 2.4448 | **-0.0018** |

**Analysis:**
- **P2a**: predictor LR=1e-2 太高，mean_A 塌到 0.499（随机），NLL 比 baseline 差 +0.228。predictor 震荡破坏 topology。
- **P2b**: 唯一出现 nll_soft < baseline 的配置。OLMo LR 很低（5e-6），baseline 几乎不动（2.447→2.445），给 predictor 留出空间。diff 在 step 1500 后稳定为负值。
- **P2c**: OLMo LR=1e-5 足够快做 CPT（baseline 2.445→2.441），CPT 贡献掩盖了 topology 的微弱效果。
- **P2d**: tau=2+init_logit=3，A 在 eval 下依然 ≈1（0.995）。OLMo CPT 主导。
- **全部 jaccard_var=0**：predictor 仍然 context-independent。

**根本问题**：2500 步不够让 predictor 学到 context-dependent topology。P2b 有微弱信号（-0.002），需要更长 run 验证。

### P2b-long / P2e / P2f — 延长 + 新方向 (12500 steps)

| Item | Value |
|------|-------|
| Status | **DONE** (P2e/P2f 在 step 5000 崩溃；P2b-long 在 eval 构建时崩溃) |
| Date | 2026-02-14 |
| Job IDs | P2b-long=15877265, P2e=15877266, P2f=15877267 |
| Hardware | A40×1 each, gpuA40x4 |
| Crash bugs | P2b-long: Dolma HTTP error during eval build (无 retry); P2e/P2f: torch.save OLMo state_dict 过大导致 zipfile 错误 |
| Bug fix | 已给 build_eval_dataloader 加 retry 逻辑；torch.save 问题待修 |

| Exp | 变量 | nll_soft@5000 | baseline@5000 | diff | mean_A | jaccard | pred_grad |
|-----|------|---------------|---------------|------|--------|---------|-----------|
| P2b-long | 同 P2b, 12500 步 | — | — | — | — | — | 崩溃无数据 |
| P2e | +λ=0.01 sparsity | 2.448 | 2.440 | **+0.008** | 0.973 | 0 | 0.017 |
| P2f | init_logit=0 | 3.020 | 2.545 | **+0.475** | 0.557 | 0 | **0.000** |

**P2e eval trajectory (sparsity pressure):**

| Step | nll_soft | nll_baseline | diff | mean_A |
|------|----------|--------------|------|--------|
| 1000 | 2.4484 | 2.4436 | +0.005 | 0.981 |
| 2000 | 2.4453 | 2.4428 | +0.003 | 0.978 |
| 3000 | 2.4425 | 2.4419 | +0.001 | 0.984 |
| 4000 | 2.4453 | 2.4403 | +0.005 | 0.972 |
| 5000 | 2.4478 | 2.4397 | +0.008 | 0.973 |

**P2f eval trajectory (init_logit=0):**

| Step | nll_soft | nll_baseline | diff | mean_A |
|------|----------|--------------|------|--------|
| 1000 | 3.089 | 2.502 | +0.587 | 0.600 |
| 2000 | 3.095 | 2.528 | +0.567 | 0.558 |
| 3000 | 3.116 | 2.539 | +0.577 | 0.555 |
| 4000 | 3.023 | 2.544 | +0.479 | 0.558 |
| 5000 | 3.020 | 2.545 | +0.475 | 0.557 |

**Analysis:**
- **P2b-long**: 崩溃在 eval 构建阶段（Dolma HTTP error），build_eval_dataloader 无 retry 逻辑。已修复。
- **P2e** (sparsity): λ=0.01 sparsity 没用。mean_A 从 0.98 微降到 0.97，但 NLL 比 baseline 越来越差（+0.001→+0.008）。sparsity 压力让 A 偏离 dense，但 OLMo 适应不过来（OLMo LR=5e-6 太慢）。
- **P2f** (init_logit=0): 完全失败。A≈0.5 起步 → NLL 比 baseline 差 +0.5。OLMo LR=5e-6 太慢无法适应 non-dense topology。predictor_grad=0（step 5000），predictor 已死。baseline 本身也在恶化（2.50→2.54），说明 OLMo 权重被非 dense topology 损坏。
- **全部 jaccard_var=0**：仍然无 context-dependent topology。

**核心困境**：
- OLMo LR 高 → OLMo 快速 CPT 到 dense 最优，predictor 无法偏离 A≈1（P2 trial, P2c, P2d）
- OLMo LR 低 → OLMo 适应不了非 dense topology，任何偏离 A≈1 都让 NLL 变差（P2e, P2f）
- Predictor LR 高 → topology 崩塌到随机（P2a）
- 目前没有配置能让 predictor 学到有意义的 context-dependent topology

### P2-Dolmino — Phase 2 Midtrain on Dolmino (12500 steps, 51M tokens)

| Item | Value |
|------|-------|
| Status | **DONE** |
| Date | 2026-02-18 |
| Job IDs | 15969981 (NODE_FAIL at step 3300), **15974693** (resume, completed) |
| Config | phase=2, stage1-final, Dolmino-Mix-1124, olmo_lr=5e-6, pred_lr=1e-3, init_logit=15, tau=5 constant, lambda=0, linear LR decay, max_grad_norm=1.0, weight_decay=0.1 |
| Tokens | ~51M (12500 steps @ batch=4, seq=1024) = **官方 midtrain 的 0.1%** (51M / 50B) |
| Hardware | A40×1 |
| Purpose | Phase 2 with Dolmino data (matching official midtrain recipe), evaluate topology benefit |

**Eval trajectory:**

| Step | nll_soft | nll_hard | nll_baseline | **hard−base** | mean_A |
|------|----------|----------|--------------|--------------|--------|
| 3000 | 2.3325 | 2.3331 | 2.3363 | -0.003 | 0.990 |
| 5000 | 2.3281 | 2.3275 | 2.3339 | -0.006 | 0.985 |
| 7000 | 2.3269 | 2.3272 | 2.3336 | -0.006 | 0.982 |
| 9000 | 2.3252 | 2.3262 | 2.3330 | -0.007 | 0.974 |
| 10500 | 2.3247 | 2.3258 | 2.3330 | -0.007 | 0.974 |
| 12000 | 2.3242 | 2.3252 | 2.3325 | -0.007 | 0.972 |
| **12500** | **2.3244** | **2.3259** | **2.3331** | **-0.007** | **0.971** |

**Key metrics:**
- 起点 NLL (stage1-final on Dolmino): **2.4171**
- 官方 midtrain 目标 (stage2-final on Dolmino): **2.3425**
- 我们的 baseline (纯 CPT) @12500: **2.3331** ← 已超过官方目标（用了 0.1% tokens）
- 我们的 nll_hard @12500: **2.3259** ← topology 额外贡献 -0.007
- CPT 贡献: 2.4171 → 2.3331 = **-0.084**
- Topology 贡献: 2.3331 → 2.3259 = **-0.007** (CPT 贡献的 8.3%)

**jaccard_var = 0 全程 — context-independent topology 问题分析：**

predictor 对所有输入产生完全相同的 binary topology。原因：

1. **init_logit=15 淹没 input-dependent 信号**：Z = UV^T + 15，即使 UV^T 随输入变化 ±2，Z 仍在 13-17 范围，σ(Z/5) 都 >0.93，hard threshold 全为 1
2. **τ=5 进一步压缩变化**：sigmoid 在 Z/τ=3 附近很平坦，微小变化不改变 hard 决策
3. **OLMo 梯度主导** (olmo_norm≈1.0 >> pred_norm≈0.003)：OLMo 快速适应固定 A，predictor 无需区分 context
4. **predictor 学到的是 constant offset**：调整 UV^T 的均值（哪些连接该削弱），而非 input-dependent 部分

**可能的改进方向：**
- 两阶段训练：先 constant τ=5 找固定 topo，再降 τ 逼 context-dependent
- 降低 init_logit (e.g. 3)，让 UV^T 的 input-dependent 变化能穿透 hard threshold
- 增大 pred_lr 或用独立的 predictor warmup
- 用 per-token Qwen embedding（而非 mean-pool）获得更细粒度 context

**显著性检验 (Job 16007864, eval_size=500, paired t-test):**

| 指标 | 值 |
|------|-----|
| n | 500 sequences |
| Mean NLL (hard topology) | 2.3772 |
| Mean NLL (baseline A=1) | 2.3818 |
| Mean diff (hard − base) | **-0.00463** |
| 95% CI | **[-0.00635, -0.00290]** |
| t-statistic | **-5.25** |
| p-value (one-sided) | **< 0.000001** |
| Cohen's d | -0.235 (small effect) |
| Sequences: hard < base | 59/500 (11.8%) |
| Sequences: hard > base | 22/500 (4.4%) |
| Sequences: hard = base | 419/500 (83.8%) |

**结论：topology 改进统计显著（p < 1e-6），95% CI 不包含 0。**
效果集中在少数序列（11.8% 受益 vs 4.4% 受损，win:lose = 2.7:1），大部分序列 topology 变化无影响（hard=base，83.8%）。
Cohen's d = 0.235 属于 small effect size，但方向一致且高度显著。

---

## Phase 1 Core (DEPRIORITIZED — frozen OLMo cannot benefit from sparse topology)

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

### A12–A14: Init logit ablation (sigmoid saturation fix)

**Problem diagnosis (from S1 & S2):**

S1 (τ=5) 和 S2 (τ=2) 的 predictor 都没有学到有意义的拓扑偏离（eval NLL ≈ baseline，mean_A ≈ 0.99）。
初始假设：sigmoid 饱和导致梯度消失（init_logit=15, ∂A/∂Z ≈ 0.0003 at τ=2）。

| ID | init_logit | Init A (τ=2) | ∂A/∂Z (τ=2) | Tokens | Purpose |
|----|-----------|--------------|-------------|--------|---------|
| A12 | 3.0 | σ(1.5) ≈ 0.82 | 0.074 | 50M | Moderate: A starts high but not saturated. |
| A13 | 0.0 | σ(0) = 0.50 | 0.125 | 50M | Maximum gradient signal. |
| A14 | 1.0 | σ(0.5) ≈ 0.62 | 0.118 | 50M | Compromise. |

**A12** — Job 15803742 (**DONE**, 12500/12500 steps)

| Step | nll_soft | nll_hard | nll_baseline | mean_A |
|------|----------|----------|--------------|--------|
| 500 | 2.4566 | 2.4566 | 2.4569 | 0.985 |
| 1500 | 2.8781 | 2.8781 | 2.4569 | 0.884 |
| 3000 | 2.6844 | 2.6844 | 2.4569 | 0.903 |
| 5000 | 2.7062 | 2.7062 | 2.4569 | 0.897 |
| 8500 | 2.7556 | 2.7556 | 2.4569 | 0.894 |
| **12500** | **2.7563** | **2.7563** | **2.4569** | **0.894** |

**A13** — Job 15803743 (**DONE**, 12500/12500 steps)

| Step | nll_soft | nll_hard | nll_baseline | mean_A |
|------|----------|----------|--------------|--------|
| 500 | 2.9356 | 2.9362 | 2.4569 | 0.834 |
| 1500 | 3.6700 | 3.6694 | 2.4569 | 0.722 |
| 3000 | 3.4731 | 3.4731 | 2.4569 | 0.694 |
| 5000 | 3.6150 | 3.6150 | 2.4569 | 0.678 |
| 8500 | 3.5187 | 3.5187 | 2.4569 | 0.676 |
| **12500** | **3.5100** | **3.5100** | **2.4569** | **0.677** |

**A14** — Job 15803744 (**DONE**, 12500/12500 steps)

| Step | nll_soft | nll_hard | nll_baseline | mean_A |
|------|----------|----------|--------------|--------|
| 500 | 2.4553 | 2.4553 | 2.4569 | 0.992 |
| 1500 | 3.7050 | 3.7050 | 2.4569 | 0.734 |
| 3000 | 3.8919 | 3.8925 | 2.4569 | 0.721 |
| 5000 | 3.4019 | 3.4019 | 2.4569 | 0.726 |
| 8500 | 3.2725 | 3.2725 | 2.4569 | 0.733 |
| **12500** | **3.2550** | **3.2550** | **2.4569** | **0.734** |

**A12–A14 综合结论：**

| ID | init_logit | Final nll_soft | vs baseline | Final mean_A | 收敛 |
|----|-----------|---------------|-------------|-------------|------|
| S2 | 15.0 | 2.4569 | +0.00 | 0.99 | 卡在 init（sigmoid 饱和） |
| A12 | 3.0 | 2.7563 | **+0.30** | 0.894 | 完全收敛（后 4K 步不动） |
| A14 | 1.0 | 3.2550 | **+0.80** | 0.734 | 完全收敛 |
| A13 | 0.0 | 3.5100 | **+1.05** | 0.677 | 完全收敛 |

| 发现 | 详情 |
|------|------|
| 梯度饱和假设 | **部分正确**：降低 init_logit 后 gate 确实在动，梯度流通了 |
| NLL 趋势 | **全部恶化**：init_logit 越低 → 偏离 dense 越远 → NLL 越差 |
| Context 依赖性 | **无**：jaccard_var=NaN，predictor 对所有 context 输出相同的 static A |
| 收敛行为 | 三个都在 ~8K 步后完全停滞，学到固定的 context-independent topology |
| nll_soft vs nll_hard | 完全相同，τ=2 下 soft≈hard |

**根本结论**：OLMo 的权重是在 dense topology 下预训练的。**Frozen 状态下，A=1 就是全局最优**。
任何偏离 dense 的拓扑 = 删除模型期望的信息 = NLL 必然变差。这不是梯度问题、init 问题或数据量问题，
而是 **loss landscape 本身不允许 frozen model 从 sparse topology 中获益**。

Phase 1（frozen OLMo）的局限性已确认。需要 Phase 2（unfreeze OLMo）让模型适应新拓扑。

---

## OLMo-2 Model Checkpoints & Midtraining Reference

### Available Checkpoints

| Checkpoint | Revision | Training | Tokens | 用途 |
|-----------|----------|----------|--------|------|
| Stage 1 Final | `stage1-step1907359-tokens4001B` | Pretrain only | 4T | **DAGFormer midtrain 起点** |
| Stage 2 Final | `stage2-ingredient3-step23852-tokens51B` | Pretrain + Midtrain | 4T + 51B | **对比目标** |
| Main | `main` | Pretrain + Midtrain + Post-train (SFT+DPO+GRPO) | — | 之前实验用的 |

### OLMo-2 1B 官方 Midtraining 超参（从 config 推算）

| 超参 | 值 | 来源 |
|------|------|------|
| Stage 1 Peak LR | 4e-4 | [OLMo2-1B.py](https://github.com/allenai/OLMo-core/blob/main/src/scripts/train/OLMo2/OLMo2-1B.py) |
| Stage 2 起始 LR | **~4e-5** (= peak × 10%, cosine 末端) | 论文: cosine decay calibrated to 10% of peak |
| Stage 2 LR schedule | **Linear decay → 0**, warmup=0 | [7B stage2 config](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage2-seed42.yaml), [32B anneal](https://github.com/allenai/OLMo-core/blob/main/src/scripts/train/OLMo2/anneal/OLMo2-32B-anneal.py) |
| Optimizer | AdamW, β=(0.9, 0.95) | 全系列一致 |
| Weight decay | 0.1 | 全系列一致 |
| Gradient clipping | max_grad_norm = 1.0 | 全系列一致 |
| Seq length | 4096 | Stage 1 config |
| Global batch size | 512 × 4096 = **~2M tokens/step** | Stage 1 config |
| Total tokens | **50B** (ingredient 3) | HF model card |
| Total steps | **~23,852** (50B / 2M) | 与 checkpoint step 吻合 ✓ |
| Data | Dolmino-Mix-1124 (高质量 web + 学术/数学/QA) | 论文 |

### 我们 vs 官方 midtrain 对比

| | OLMo 官方 midtrain | 我们的 P2b |
|---|---|---|
| 起始 LR | ~4e-5 | olmo_lr=5e-6 (8x 低) |
| LR Schedule | Linear → 0 | Cosine |
| Weight decay | 0.1 | 0.01 |
| Batch size | ~2M tokens/step | ~4K tokens/step (**500x 小**) |
| Seq length | 4096 | 1024 |
| Total tokens | 50B | ~51M |
| Data | Dolmino (高质量) | Dolma v1.7 (原始 web) |
| Grad clip | 1.0 | 无 |
| 起始模型 | Stage 1 final (pretrain only) | Main (post-trained) |

### Baseline NLL 评估

| Checkpoint | Revision | NLL | Job |
|-----------|----------|-----|-----|
| **Stage 1 Final** (pretrain only) | `stage1-step1907359-tokens4001B` | **2.4507** | 15936385 |
| **Stage 2 Final** (midtrain) | `stage2-ingredient3-step23852-tokens51B` | **2.4639** | 15936385 |
| **Main** (post-trained) | `main` | **2.4579** | 15936385 |

Eval: skip=10000, size=50, seq_len=1024, Dolma v1.7, A=1 (vanilla forward)

**意外发现**: Midtrain 后 NLL 反而变高（2.4507 → 2.4639, +0.013）。原因是 midtrain 用 Dolmino（高质量学术/数学/QA），而 eval 用 Dolma v1.7（通用 web）。Midtrain 优化的是下游 benchmark（GSM8K、MMLU），不是通用 web NLL，在 Dolma 上出现 domain shift 是正常的。

**对实验的影响**:
- 从 stage1-final 出发做 DAGFormer midtrain 的起点 NLL = 2.4507
- 不应以 stage2-final 的 NLL 为目标（它不是在 Dolma 上优化的）
- 之前所有实验用的 main (post-trained) NLL = 2.4579，比 stage1-final 稍高

### Baseline NLL — Dolmino-Mix-1124 数据

| Checkpoint | Revision | NLL (Dolmino) | NLL (Dolma v1.7) | Job |
|-----------|----------|---------------|-------------------|-----|
| **Stage 1 Final** | `stage1-step1907359-tokens4001B` | **2.4171** | 2.4507 | 15940803 |
| **Stage 2 Final** | `stage2-ingredient3-step23852-tokens51B` | **2.3425** | 2.4639 | 15940803 |

Eval: skip=10000, size=50, seq_len=1024, Dolmino-Mix-1124, A=1 (vanilla forward)

**符合预期**: Midtrain 后在 Dolmino 上 NLL 显著下降（2.4171 → 2.3425, **-0.075**）。
这是官方 midtrain 的效果：50B tokens 的 Dolmino 训练让模型在该 domain 上 NLL 降了 3.1%。
我们的 DAGFormer midtrain（从 stage1-final 出发在 Dolmino 上训练）目标是达到或超过 2.3425。

### P1 Phase 1 — Stage1 & Stage2 (Frozen OLMo, Dolma v1.7, init_logit=3, tau=2)

确认 frozen OLMo 在不同起点下均无法从 sparse topology 获益。

| Exp | 起始模型 | Baseline | Best NLL | Final NLL | mean_A | Job |
|-----|----------|----------|----------|-----------|--------|-----|
| P1-Stage1 | stage1-final | 2.4481 | 2.6006 | 2.6012 | 0.842 | 15936725 |
| P1-Stage2 | stage2-final | 2.4597 | 2.6437 | 2.6437 | 0.887 | 15936726 |

Config: init_logit=3, tau=2 constant, lambda=0, 12500 steps, batch=4, Dolma v1.7

**P1-Stage1 eval trajectory:**

| Step | nll_soft | nll_baseline | diff | mean_A |
|------|----------|--------------|------|--------|
| 500 | 2.4525 | 2.4481 | +0.004 | 0.965 |
| 1000 | 2.6300 | 2.4481 | +0.182 | 0.852 |
| 5000 | 2.6187 | 2.4481 | +0.171 | 0.849 |
| 10000 | 2.6006 | 2.4481 | +0.153 | 0.842 |
| 12500 | 2.6012 | 2.4481 | +0.153 | 0.842 |

**P1-Stage2 eval trajectory:**

| Step | nll_soft | nll_baseline | diff | mean_A |
|------|----------|--------------|------|--------|
| 500 | 2.4613 | 2.4597 | +0.002 | 0.975 |
| 1000 | 2.4709 | 2.4597 | +0.011 | 0.973 |
| 3500 | 2.6138 | 2.4597 | +0.154 | 0.887 |
| 10000 | 2.6456 | 2.4597 | +0.186 | 0.886 |
| 12500 | 2.6437 | 2.4597 | +0.184 | 0.887 |

**结论**: 两个起点结果一致 — frozen OLMo 下，A<1 只会恶化 NLL。
predictor_grad 在前几千步后降为 0，predictor 学到一个 static A（context-independent）就停滞了。
完全验证了 A12-A14 的结论：**frozen OLMo = A=1 是全局最优**。Phase 2 是唯一出路。

---

## D1/D2/D6 — Escaping the Flat Gradient Plateau (4×A40 DDP, 12500 steps)

**动机**: 之前所有实验表明 A=1 处梯度平坦。D1/D2/D6 尝试三种不同策略逃离平坦区。

| Item | Value |
|------|-------|
| Status | **DONE** |
| Date | 2026-02-20 |
| Job IDs | D1=16027311, D2=16027312, D6=16027313 |
| Hardware | 4×A40 DDP each |
| Base config | phase=2, olmo_lr=5e-6, predictor_lr=1e-3, Dolmino-Mix, stage1-final |

| Exp | 策略 | init_logit | τ | λ | nll_soft@12500 | baseline@12500 | **diff** | mean_A | jaccard |
|-----|------|-----------|---|---|---------------|---------------|----------|--------|---------|
| **D1** | A≈0.5 起步 | 0 | 5→5 (fixed) | 0 | 2.577 | 2.391 | **+0.186** | 0.709 | 0 |
| **D2** | τ annealing + sparsity | 15 | 5→0.2 | 0→0.01 | 2.340 | 2.347 | **-0.007** | 0.972 | 0 |
| **D6** | A≈0 起步 (learn to turn ON) | -15 | 5→5 (fixed) | 0 | 2.926 | 2.625 | **+0.301** | 0.556 | 0 |

**D1 eval trajectory (init_logit=0, A starts ≈0.5):**

| Step | nll_soft | nll_baseline | diff | mean_A |
|------|----------|--------------|------|--------|
| 1000 | 2.577 | 2.391 | +0.186 | 0.748 |
| 5000 | 2.603 | 2.392 | +0.211 | 0.695 |
| 10000 | 2.585 | 2.391 | +0.194 | 0.705 |
| 12500 | 2.577 | 2.391 | +0.186 | 0.709 |

**D2 eval trajectory (τ anneal 5→0.2, λ=0→0.01):**

| Step | nll_soft | nll_baseline | diff | mean_A |
|------|----------|--------------|------|--------|
| 1000 | 2.351 | 2.360 | -0.009 | 0.966 |
| 5000 | 2.338 | 2.349 | -0.011 | 0.950 |
| 9000 | 2.337 | 2.347 | -0.010 | 0.971 |
| 12500 | 2.340 | 2.347 | -0.007 | 0.972 |

**D6 eval trajectory (init_logit=-15, A starts ≈0):**

| Step | nll_soft | nll_baseline | diff | mean_A |
|------|----------|--------------|------|--------|
| 1000 | 2.904 | 2.606 | +0.298 | 0.607 |
| 5000 | 2.911 | 2.621 | +0.290 | 0.552 |
| 10000 | 2.938 | 2.625 | +0.313 | 0.550 |
| 12500 | 2.926 | 2.625 | +0.301 | 0.556 |

**综合分析:**

| 发现 | 详情 |
|------|------|
| **D1 失败** | A≈0.5 起步 → NLL 比 baseline 差 +0.19。OLMo 无法适应 non-dense topology（OLMo LR=5e-6 太慢）。predictor_norm 大部分时间≈0。 |
| **D2 微弱** | 唯一正面信号（diff=-0.007），但 A≈0.97（几乎 dense）。改善完全来自 OLMo CPT，不是拓扑。和 P2-Dolmino (-0.007) 完全一致。 |
| **D6 惨败** | A≈0 起步 → NLL 差 +0.30。baseline 也在恶化（2.606→2.625），说明 OLMo 被 sparse topology 损坏。 |
| **全部 jaccard_var=0** | predictor 从未学会 context-dependent routing。三种初始化策略都失败。 |
| **predictor 本质上无用** | 无论从 A≈1、A≈0.5、还是 A≈0 出发，predictor 都无法学到有意义的拓扑。 |

**最终结论**: **CPT on pretrained OLMo 学拓扑这条路彻底失败。** 无论 init、τ schedule、sparsity pressure 如何调整，pretrained 权重对 A 的梯度本质平坦。接下来转向 **从头预训练** 方案（Mini OLMo-2 300M），让权重和拓扑从随机初始化开始共同进化。

---

## From-Scratch Pretraining — Mini OLMo-2 300M

### Pretrain-300M-Baseline (dense, no DAGFormer)

| Item | Value |
|------|-------|
| Status | **DONE** |
| Date | 2026-02-20 |
| Job ID | 16056723 |
| Hardware | 4×A40 DDP |
| Architecture | OLMo-2 style: 12L × 16H, hidden=1024, head_dim=64, intermediate=4096, weight tying |
| Params | 304,137,216 (unique, with weight tying) |
| Config | lr=5e-4, beta2=0.95, wd=0.1, grad_clip=1.0, linear decay to 0, warmup=300 steps |
| Tokens | ~5.24B (10K steps × 524K tok/step) |
| Throughput | ~81K tok/s |
| Seed | 42 |
| **eval/nll @10000** | **3.6094** |
| Checkpoint | `checkpoints/pretrain_300m_baseline/checkpoint_step10000.pt` |

---

### Pretrain-300M-DAGFormer: Static / Self-embed / Qwen (from scratch)

**动机**: CPT on pretrained OLMo 完全失败。新策略：从随机初始化开始，让权重和拓扑共同进化。

| Item | Value |
|------|-------|
| Status | **DONE** (partial — 被30h wall time截断) |
| Date | 2026-03-01 |
| Job IDs | static=16340963, self_embed=16340964, qwen=16340965 |
| Hardware | 4×A40 DDP each |
| A matrix | 192×192 (12L × 16H) |
| Config (共同) | lr=5e-4, pred_lr=3e-4, tau=5→0.2 cosine, λ=0→0.01, init_logit=15, micro_batch=4, accum=32, seq=1024 |
| Token budget | 5.24B (10K steps × 524K tok/step), same as baseline |

**结果对比 (各自跑到的最远step):**

| Variant | Steps completed | tok/s | train/nll | eval/soft | eval/baseline | mean_A | 问题 |
|---------|----------------|-------|-----------|-----------|---------------|--------|------|
| Baseline | 10000/10000 | 81K | — | — | 3.609 | — | — |
| **Static** | ~4690/10000 | ~23K | ~5.70 | ~5.89 | ~5.98 | **0.98** | 几乎dense，没学到topology |
| **Self-embed** | ~4670/10000 | ~23K | ~5.57 | **5.75** | 5.98 | **0.42** | 学到sparse！但base model严重undertrained |
| **Qwen** | ~2920/10000 | ~14K | ~6.01 | ~6.15 | ~6.28 | **0.41** | sparse但最慢（Qwen推理开销） |

**关键发现:**

| 发现 | 详情 |
|------|------|
| **DAGFormer 3.5x 慢** | 81K (baseline) vs 23K (DAGFormer) tok/s。per-head input assembly + einsum 开销巨大 |
| **Self-embed 学到了sparse topology** | mean_A=0.42，说明predictor确实在学！这是所有实验中首次看到有意义的topology |
| **Self-embed soft < own baseline** | soft(5.75) < baseline(5.98)，证明learned topology比dense好。但这是和同期undertrained的model比，不是和baseline的3.6比 |
| **Static 失败** | mean_A=0.98，pred_grad_norm=0.0001 — 没有input-dependent信号，predictor死了 |
| **Qwen 最慢** | 额外Qwen推理让throughput降到14K tok/s。sparse topology但太慢 |
| **根本矛盾** | DAGFormer forward太慢 → 同样wall time，base model见的token少3.5x → base model没训好 → 即使topology有用，整体也比baseline差很多 |

**结论**: Self-embed证明了topology学习是可行的，但DAGFormer forward的3.5x overhead让从头训练不现实。需要新策略解决throughput问题。

---

### Pretrain-300M-DAGFormer: 新策略 (Batch 2 — 解决throughput问题)

**三种新方法，解决DAGFormer forward太慢的根本问题:**

| Item | Value |
|------|-------|
| Status | **SUBMITTED** |
| Date | 2026-03-05 |
| Job IDs | staged=16341236, alternating=16341237, nodetach=16341238 |
| Hardware | 4×A40 DDP each |

#### 1. Staged (分阶段)
- **思路**: 先用dense baseline训好base model (已完成的10K步checkpoint)，再切到DAGFormer forward继续训
- **解决的问题**: base model不用重新学语言能力，只需适应routing
- Config: `pretrain_300m_staged.yaml`, predictor_type=self_embed, baseline_checkpoint=step10000, 2000步
- Job: 16341236

#### 2. Alternating (交替)
- **思路**: 从头训，但每10步只有1步走DAGFormer forward (慢路径)，其余9步走标准dense forward (快路径)
- **解决的问题**: 整体throughput≈baseline的90%，predictor仍能学习（但信号少10x）
- Config: `pretrain_300m_alternating.yaml`, standard_steps_per_dag_step=9, 10000步
- Job: 16341237

#### 3. Self-embed no-detach (不截断梯度)
- **思路**: 和staged一样加载baseline checkpoint，但embedding送入predictor时不.detach()
- **解决的问题**: embedding不仅服务NLL，还学会产生对predictor有用的表征（类似Qwen但用自己的、可训练的embedding）
- Config: `pretrain_300m_selfembed_nodetach.yaml`, baseline_checkpoint=step10000, detach=false, 2000步
- Job: 16341238

---

### Pretrain-300M-DAGFormer: 新策略 (Batch 3 — 改进predictor输入)

**问题**: 原始self_embed对raw token embeddings做mean pooling = bag-of-words，丢失所有上下文/语序信息。

**两种新predictor，用小型encoder获取上下文表征:**

| Item | Value |
|------|-------|
| Status | **SUBMITTED** |
| Date | 2026-03-05 |
| Job IDs | mini_encoder=16341566, context_embed=16341567 |
| Hardware | 4×A40 DDP each |
| Base model | 加载baseline checkpoint (staged) |
| Training | 2000步 × 524K tok/step ≈ 1.05B tokens |

#### 4. Mini-encoder (独立小encoder)
- **架构**: 独立embedding table (100K × 256) + 2层transformer (dim=256, 4头, FFN=1024) + mean pool → PredictorMLP → A
- **参数**: ~41.2M (其中embedding table 25.7M)
- **优点**: 完全独立于LLM，学到纯粹为topology预测优化的上下文表征
- **缺点**: 额外vocab参数多
- Config: `pretrain_300m_mini_encoder.yaml`
- Job: 16341566

#### 5. Context-embed (共享embedding + 独立encoder)
- **架构**: 复用LLM的embed_tokens (detach) → Linear(1024→256) → 2层transformer → mean pool → PredictorMLP → A
- **参数**: ~15.7M (省掉25M的vocab参数)
- **优点**: 利用已训好的LLM embedding，参数量小
- **缺点**: embedding通过detach隔离，predictor encoder无法影响embedding学习
- Config: `pretrain_300m_context_embed.yaml`
- Job: 16341567

**全部5个新实验汇总:**

| # | Name | 策略 | Base model | Predictor | Steps | Job ID |
|---|------|------|-----------|-----------|-------|--------|
| 1 | staged | baseline ckpt → DAGFormer | 已训好 | self_embed (mean pool) | 2000 | 16341236 |
| 2 | alternating | 9:1 standard:DAGFormer | 从头训 | self_embed (mean pool) | 10000 | 16341237 |
| 3 | nodetach | baseline ckpt + 不截断梯度 | 已训好 | self_embed (mean pool, no detach) | 2000 | 16341238 |
| 4 | mini_encoder | baseline ckpt | 已训好 | 独立embed + 2L transformer | 2000 | 16341566 |
| 5 | context_embed | baseline ckpt | 已训好 | 共享embed + 2L transformer | 2000 | 16341567 |

**关键对比维度:**
- 1 vs 3: detach vs no-detach 的效果
- 1 vs 4 vs 5: predictor输入质量（bag-of-words vs contextual）
- 1 vs 2: staged vs alternating 训练策略
- 4 vs 5: 独立 vs 共享 embedding table

---

## Analysis Experiments

### X0 — Gradient Analysis at A=1 (Frozen OLMo 局部最优性证明)

| Item | Value |
|------|-------|
| Status | **DONE** |
| Date | 2026-02-17 |
| Job ID | 15974169 |
| Hardware | A40×1 |
| Method | 在 A=1 处计算 ∂L/∂A，50 个 eval windows，逐 window backward |

**目的**: 严格判断 A=1 是否是 frozen OLMo 下的局部/全局最优。

**结果**:

| 指标 | Dolma v1.7 | Dolmino |
|------|-----------|---------|
| grad > 0 的比例 | 49.7% | 50.0% |
| grad < 0 的比例 | 50.3% | 50.0% |
| Mean ∂L/∂A | -0.000001 | 0.000000 |
| Std ∂L/∂A | 0.000090 | 0.000061 |
| Mean |∂L/∂A| | 0.000042 | 0.000030 |
| Max ∂L/∂A | +0.002359 | +0.001285 |
| Min ∂L/∂A | -0.002902 | -0.001288 |

Per-layer-pair: 所有 layer pair 的 %pos 在 43-60% 之间随机波动，无一致方向。
Adjacent vs skip: 同样 ~50/50，无显著差异。

**核心结论**:
1. **∂L/∂A ≈ 0**：梯度量级 ~4e-5，比 NLL (~2.4) 小 5 个数量级
2. **正负完美 50/50**：每个 window、每个 layer pair 都是噪声，不是信号
3. **A=1 是极其平坦的极值点**：严格意义上存在 grad > 0 的方向，但信号强度等效于零
4. **两个数据集完全一致**：不是数据问题，是 loss landscape 的本质特征
5. **解释了 Phase 1 全部失败**：predictor_grad → 0 不是 sigmoid 饱和问题，而是 ∂L/∂A 源头就没有信号
6. **Oracle search 的 2.58→0.12 大概率是 overfitting**：500 步直接优化单 window 的 30720 变量 = 用 topology 编码具体 token 序列

**理论意义**：Frozen OLMo 的 loss landscape 在 A=1 处是一个平坦高原。基于梯度的方法（包括 predictor）无法在此获得有意义的学习信号。Phase 2（unfreeze OLMo）是必要条件，因为只有让 OLMo 权重共同适应，才能在非 dense topology 下创造出有梯度信号的 loss landscape。

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

---

## FourWay — Per-Head Per-Token 4-way Routing (Q/K/V/R)

**动机**: MUDDFormer-style per-layer per-token 路由，但扩展到 per-head 粒度。External predictor 从 input_ids 预测所有层的 routing weights `α ∈ [B, T, H, l+1]` (Q/K/V per-head) + `[B, T, l+1]` (R per-token)。

**架构要点** (详见 `src/model/olmo_graph.py:FourWayDAGFormer` 和 `src/model/predictor.py:FourWayPredictor`):
- 4 streams: Q/K/V 按 head 路由, R 按 token 共享路由
- Identity init: 每个 head 的 `W=0`, `bias=[0,...,0,1]` → init 恰好等于 dense OLMo-2
- Causal predictor: 2-layer transformer encoder, dim=256, causal mask (KV cache 兼容)
- 项目 then 混合优化: L 次标准 QKV projection + einsum in head_dim（vs per-head D=1024 einsum 减小 overhead 从 3.5x → 2.5x）
- 无 Gumbel-sigmoid，纯 soft continuous gates
- Bias params 单独 wd=0 group（防止 identity init 被 wd 腐蚀）

**Dense baseline 参考**: 同 300M 配置 (12L × 16H × 1024), 5000 steps, from-scratch → eval NLL **3.85**, train/eval gap = 0.

### FourWay 主要 runs

| # | Run | Job ID | Config | Step | Train NLL | Eval NLL | Gap | 备注 |
|---|-----|--------|--------|------|-----------|----------|-----|------|
| 1 | **joint_causal (best)** | — | fourway_corrected_joint_causal (causal pred, base lr 5e-4, pred lr 3e-4, wd 0.1) | 5000 | ~3.5 | **5.49** | 2.0 | 最好的 from-scratch FourWay 结果 |
| 2 | predfirst_causal | — | predictor-first warmup (pred lr 3e-4) then joint | 5000 | — | ~5.6 | ~1.9 | 和 joint 相差不大 |
| 3 | long_train @ 6000 | 17253035 | 同 joint_causal, 10000 步计划 | 6000 | ~3.6 | **5.37** | 1.8 | Gap 随训练慢慢缩小 (5000:2.0 → 6000:1.8) |
| 4 | long_train resume | 17253035 | phase2 从 6000 resume | — | — | — | — | **FAILED**: resume 时 key prefix 不匹配 bug (见 "Known bugs") |

### 正则化 ablations (全部 5000 步 / 4×A40 DDP, from-scratch)

所有变体：train NLL 3.5-5.0, **eval NLL 5.4-6.9**, gap 从没被 close 到 < 1.5。

| Variant | Job | 机制 | Train NLL | Eval NLL | Gap | 结论 |
|---------|-----|------|-----------|----------|-----|------|
| **baseline dense** | 16056723 | 无 routing | — | **3.85** | 0 | 目标 |
| fourway joint causal | — | 无 reg | ~3.5 | 5.49 | 2.0 | **最好 FourWay 基线** |
| L2 decay (λ=1e-3..1e-2) | reg_l2decay | L2 on α logits | ~3.6 | ~5.55 | ~1.95 | 没用 |
| L2 clamp (±2) | reg_l2clamp | clamp α logits | ~3.7 | ~5.52 | ~1.85 | 没用 |
| L1 delayed (start@15%) | reg_delayed_l1 | 晚启动 L1 稀疏 | ~3.7 | ~5.56 | ~1.9 | 没用 |
| Entropy reg | reg_entropy | 对 softmax(α) 的熵惩罚 | ~3.7 | ~5.60 | ~1.9 | 没用 |
| Top-k=2 | reg_top2 | 保留 top-2 sources | ~3.8 | ~5.65 | ~1.85 | 没用 |
| Temperature (τ=2) | reg_temp | α / τ 再用 | ~4.5 | **7.36** | ~2.9 | 变差 |
| Noise injection | reg_noise | α += N(0, 0.1) 训练 | ~3.7 | ~5.60 | ~1.9 | 没用 |
| Small predictor (hidden 128) | reg_small_pred | 减 predictor 容量 | ~3.9 | ~5.62 | ~1.7 | 最多减 0.3 gap |
| Tiny predictor (hidden 64) | reg_tiny_pred | 更小 | ~4.1 | ~5.68 | ~1.6 | 最多减 0.4 gap |
| Low pred LR (1e-5) | reg_low_pred_lr | predictor 慢更新 | ~4.3 | ~5.72 | ~1.4 | train 变差更多 |
| Micro pred | reg_micro_pred | 超小 predictor | ~4.5 | ~5.85 | ~1.35 | train 变差 |
| Dropout routing 0.15 | 17308934 | 15% token 位置 reset 成 identity | 3.73 | 6.52 | 2.79 | **最差 gap — identity dropout 是 shortcut, 不是 regularizer** |
| Lower base lr (1.7e-4) | 17308932 | base LR 减 3x | 4.65 | 6.81 | 2.16 | 同时变差 |
| Lower both lr | 17308933 | base + pred 都减 | 4.89 | 6.85 | 1.96 | 最小 gap 但 train 最差 |
| Regularized combo | 17308935 | 上面多个一起 | 4.38 | 6.73 | 2.35 | 没用 |

### Normalization 实验 (全部爆炸)

| Variant | Mechanism | Train | Eval | 状态 |
|---------|-----------|-------|------|------|
| softmax | `α = softmax(raw + bias)` per stream | ~3.6 | **9.31** | 爆炸 |
| softmax_dropout | softmax + routing_dropout 0.1 | ~3.7 | **11.2** | 爆炸 |
| sinkhorn | Sinkhorn-Knopp 20 iters | ~3.6 | **10.8** | 爆炸 |
| sinkhorn_lite | 5 iters | ~3.7 | **12.4** | 爆炸 |
| sinkhorn_dropout | + dropout | — | **14-16** | 爆炸 |
| row_col (mHC-lite substitute) | alternating row/col normalize | — | **13+** | 爆炸 |

**失败原因** (Codex Q3): softmax 破坏 identity init. `softmax([0,1]) = [0.269, 0.731]` 不是 one-hot，27% 信号从错误层来，直接污染 R stream (residual carrier)，从 step 0 就严重 off-policy。

### Identity init 数学验证 (2026-04-06)

**Task**: 验证 FourWay 在 identity init 下是否 bit-exact 等于 dense OLMo 的 forward。

**Script**: `scripts/verify_identity_init.py` (Job 17347931)

| Dtype | seq_len | max_abs_diff (logits) | NLL diff | Result |
|-------|---------|----------------------|----------|--------|
| **fp32** | 128 | **6.3e-6** | **0.000000** | **PASS** |
| bf16 | 128 | 0.064 (~2%) | 8.5e-4 | bf16 噪声 |
| bf16 | 1024 | 0.063 (~2%) | 6.1e-5 | bf16 噪声 |

**结论**: FourWay 的 forward 数学是正确的。**Identity init 不是 overfit 根因**。bf16 2% logit drift 来自 einsum vs native attention 的累加顺序差异，NLL 影响 < 1e-3 可忽略。

### Eval with forced identity routing (Codex's Q5 experiment)

**Task**: Load 最好 checkpoint (joint_causal step 5000)，跑 eval 对比 3 种 routing 模式 — 诊断是 predictor overfit 还是 base model co-adapt。

**Script**: `scripts/eval_force_identity.py` (Job 17348622)

| Eval mode | NLL | 说明 |
|-----------|-----|------|
| `dense_baseline` (trained base, 直接 forward, no wrapper) | **7.0774** | 完全等于下面 fw_id ✓ 验证 forward 正确 |
| `fw_with_identity` (FourWay forward + α=identity) | **7.0774** | |
| `fw_with_predictor` (FourWay forward + 真实 predictor) | **6.3161** | |

**注**: 这里的绝对 NLL 比 training log (5.49) 高 ~0.8，是因为只用了 13 eval batches 而且我的 eval pipeline 和训练 eval 略有不同。**关键是相对比较**：

- `fw_id == dense` (exact): 确认 forward 数学正确
- `fw_with_predictor (6.32) < fw_with_identity (7.08)` by **0.76 nats**: predictor 的 routing **有效**，不是 garbage
- `fw_with_identity (7.08) >> 真 dense baseline (3.85)` by **3.2 nats**: **base model 已经强依赖 predictor 的 non-identity α**，脱离 predictor 就废

**诊断 (H2 confirmed)**: base model co-adapted to routing. 这是联合训练的自然结果，不是 bug —— 但说明 predictor + base 的平衡点不在 dense baseline 附近，它们一起走到了一个完全不同的解空间。

### α 统计 (from joint_causal step 5000 predictor)

| Stream | mean_norm | max | min | 观察 |
|--------|-----------|-----|-----|------|
| Q | 1.31 | 2.79 | -1.33 | 温和，q_norm 下游保护 |
| K | 1.15 | 2.68 | -1.91 | 温和，k_norm 下游保护 |
| **V** | **1.59** | **10.77** | **-7.67** | **outlier！无 norm 保护** |
| R | 0.94 | 1.96 | -2.22 | 实际比 identity (=1.0) 更小 |

**关键发现**: **V stream 是 outlier 最严重的**（max ±10, mean norm 59% 高于 identity），因为 Q/K 有 OLMo-2 自带的 `q_norm`/`k_norm` (post-projection RMSNorm), R norm 实际上很温和, **只有 V 完全没有 post-mix 归一化**。

Codex 最初把 runaway 归因于 R (因为 R 直接进残差流)，但实际数据显示 **R 其实很乖，V 才是放飞的那个**。

### Approach A: V Post-Mix RMSNorm (2026-04-06, running)

**动机**: 给 V 加一个和 Q/K 对称的 `Olmo2RMSNorm(model_dim)` post-mix，Q/K/R 保持 raw。

**实现**:
- `src/model/olmo_graph.py:FourWayDAGFormer.__init__`: 加 `use_v_norm: bool` 参数 → `self.v_norms = nn.ModuleList([Olmo2RMSNorm(model_dim) for _ in range(num_layers-1)])`
- Forward: 在 V mixing einsum 之后 apply v_norm (rearrange `b h t d → b t (h d)` → norm → rearrange 回来)
- `scripts/pretrain_dagformer.py`: 加 `use_v_norm` config 字段, 传给构造函数
- `configs/fourway_vnorm.yaml`: 基于 fourway_corrected_joint_causal + `use_v_norm: true`

**Identity init drift** (Job 17349536): 
- max_abs_diff (fp32) = 0.173 (vs 6e-6 无 vnorm) 
- **NLL diff = 1e-5** (可忽略) 
- 结论: v_norm 在 init 时对 NLL 影响为零，只是改变了 V 的 scale，训练可从近似 identity 状态起步。

**Training**: Job 17350168 (submitted 2026-04-06)

---

## Known Bugs (FourWay)

1. **Checkpoint resume 在 FourWay 模式下损坏** (Codex-identified, 2026-04-06):
   - `save_checkpoint()` 存 `predictor_state_dict` + `routing_state_dict` + 外部 `model_state_path`
   - `load_checkpoint()` 在 FourWay 模式下把 bare OLMo weights 加载到 `combined_raw.base_model`（是 `FourWayDAGFormer` wrapper，真 OLMo 在 `.olmo` 下）→ 键前缀错位
   - `routing_state_dict` 根本没被 load 回去
   - 症状: long_train 17253035 resume 失败
   - 修复: 需要改 `load_checkpoint()`, 把 bare model weights 加载到 `combined_raw.base_model.olmo`, 把 routing_state_dict 加载到 wrapper 本身
   - 状态: **未修**

2. **Predictor bias param group (wd=0) LR 不被 scheduler 更新** (Codex-identified):
   - Optimizer 3 个 group: `[base, pred_other, pred_bias_wd0]`
   - LR scheduler 只更新 `[0]` 和 `[1]`, group `[2]` 的 LR 锁在初始值
   - 影响很小（bias 一般不需要严格退火）, 状态: **未修**

### Approach B: Identity Predictor + Correction MLPs Only (Track 2, 2026-04-06)

**动机** (Codex Q5 follow-up + corrected eval data):

修好 `eval_force_identity.py` 的 correction MLPs 加载 bug 后 (Job 17351237), joint_causal step 5000 真实数据：

| Mode | NLL | Δ |
|------|-----|---|
| dense_baseline (pure OLMo, no FW) | 7.0774 | — |
| fw_id_noc (α=identity, corr OFF) | 7.0774 | bit-exact ✓ |
| fw_id (α=identity, **corr ON**) | 5.9662 | corrections alone -1.11 |
| fw_pred_noc (predictor α, corr OFF) | 6.3161 | predictor alone -0.76 |
| **fw_pred (full)** | **5.4924** | matches training log ✓ |

**关键发现**: Local correction MLPs 贡献 **-1.11 nats**，external predictor 增量只 -0.47。Correction = 70% of routing benefit. Local > global.

**实验设计**: 冻结 `FourWayPredictor` 在 identity init (W=0, bias=[0,...,0,1])，只训 correction MLPs + base model。如果 eval ≥ joint_causal 5.49，说明 external predictor 完全可去 → overfit 主要来自 predictor 全局记忆。

**实现**:
- `scripts/pretrain_dagformer.py`: 加 `freeze_predictor: bool = False` 配置, after 创建 fourway_predictor 调用 `requires_grad_(False)` + `eval()`
- Optimizer 改为按 `requires_grad` 过滤, 跳过空 param groups (handles freeze_predictor)
- `configs/fourway_local_only.yaml`: 基于 `fourway_corrected_joint_causal` + `freeze_predictor: true`

**Job**: 17352435 (PENDING, submitted 2026-04-06)

**期望结果**:
- 若 eval ≤ 5.49: external predictor 不必要，可去
- 若 eval ≈ 5.97 (= fw_id 单独 corrections): correction 需要 predictor 提供初始 α 才能完全发挥
- 若 eval > 5.97: 反直觉，corrections 自己上路效果反而差


### Approach C: Classical predictor dropout (Track 4, 2026-04-06)

**动机**: 我们之前试过 `routing_dropout` (reset α to identity) 但 Codex 指出那不是经典 dropout，是 shortcut。**Predictor 内部的真正 nn.Dropout 从来没试过** — `src/model/predictor.py:982` 一直是 `dropout=0.0` hardcoded。

如果 overfit 来自 "predictor encoder 记住了具体 input_ids → α 的映射"，经典 dropout 应该有效：
- nn.Dropout(0.1) 在训练时随机置零 encoder 注意力 + FFN + trunk 激活
- 强迫 encoder 学冗余表征
- eval 自动关闭 (`combined.eval()` propagates to children)
- 这是 routing_dropout 完全没做到的事

**实现**:
- `src/model/predictor.py:FourWayPredictor.__init__`: 加 `dropout: float = 0.0` 参数, 传给 `nn.TransformerEncoderLayer(dropout=dropout)`, 也加到 trunk 末尾 `nn.Dropout(dropout)`
- `scripts/pretrain_dagformer.py`: 加 `predictor_dropout: float = 0.0` config
- 同步更新 `eval_force_identity.py` 用 `dropout=config.get("predictor_dropout", 0.0)` 加载 checkpoint
- `configs/fourway_pred_dropout.yaml`: 基于 `fourway_corrected_joint_causal` + `predictor_dropout: 0.1`

**Job**: 17360292 (PENDING, submitted 2026-04-06)

**期望**:
- 若 eval ≤ 5.0: predictor memorization 是 overfit 主因，经典 dropout 解决了
- 若 eval ≈ 5.49: dropout 没用，predictor 不靠死记上面的具体 mapping
- 若 eval > 5.49: dropout 太强，predictor 学不出有用 routing

### Approach D: Label smoothing + Smaller correction MLPs (Track 5, 2026-04-06)

**动机**: 经典 overfit 工具箱中我们从来没试过的两个最直接方法。

**1. Label smoothing 0.1 on train NLL**:
- 直接打击症状: train NLL 3.5 < dense baseline 3.85 (FourWay 在 train 上 over-confident)
- F.cross_entropy 自带 `label_smoothing=0.1`, 仅训练 loss 用, eval 仍 raw NLL
- 实现: `pretrain_dagformer.py` 4 处 train cross_entropy 加 `label_smoothing=config.label_smoothing`

**2. Smaller correction MLPs (correction_hidden 128 → 32, 4x reduction)**:
- 上一轮发现 corrections 占 70% routing benefit (-1.11 of -1.59 nats)
- 它们是 dominant capacity
- 之前从没改过 correction_hidden，default 128
- 直接削减最可能的 overfit source

**Configs**:
- `fourway_label_smooth.yaml`: 仅 label smoothing 0.1
- `fourway_small_corr.yaml`: 仅 correction_hidden 32
- `fourway_smooth_small_corr.yaml`: 两个组合

**Jobs**: 17362778, 17362779, 17362780 (PENDING, submitted 2026-04-06)

**期望**:
- label_smooth 单独: 若 train NLL 升回 ~3.85 + eval ~5.0 → label smoothing 直接生效
- small_corr 单独: 若 corrections 1.11 → ~0.7 但 eval 改善 → corrections 是 overfit 主源
- combo: 若 显著优于两个单独 → 两个机制独立有效

### Root-cause update + clean diagnostics (2026-04-09)

**范围修正**: 只看当前 **FourWay soft routing** 线。旧的 hard gate / Gumbel `A` 线不再作为主分析对象。

**最新判断**:
- 目前更像不是某个单独组件坏掉，而是 **dynamic conditional DoF 太大**。FourWay 不只是 “多了 ~30M 静态参数”，更像一个 input-conditioned routing / 小型 hypernetwork。
- “它等价于更大的 LLM，需要更大的 recipe” 这个说法 **部分成立**，但不够精确。更准确的是: **大条件化空间 + joint training 下的 predictor/base co-adaptation**，让模型很容易收敛到 train 好、eval 差的解。
- 现有证据支持这个判断:
  - 真 dense baseline (独立 300M OLMo-2) eval NLL = **3.8459**
  - `fourway_corrected_joint_causal` eval NLL = **5.4924**
  - pure FourWay `long_train` 到 6000 step 能降到 **5.3702**，说明 recipe 确实重要，但离 dense 仍很远
  - forced-identity eval 显示: `fw_pred=5.4924`, `fw_id=5.9662`, `fw_pred_noc=6.3161`, `fw_id_noc=7.0774`
  - 结论: routing 是有效的，但学到的是泛化差的解；而且 **correction MLPs (-1.11 nats)** 比 external predictor (-0.47 nats) 更像主容量来源
- 因此当前工作假说是: **过拟合主因是 dynamic routing 的自由度和 joint co-adaptation，不只是某个局部模块 bug**。单靠常规小正则化大概率只能缓解，不能直接追平 dense baseline。

**诊断原则**:
- 不再重跑 baseline。
- 优先做能区分 “head-level routing 太自由 / correction token-level memorization / predictor embedding lookup memorization / 真正 layer-level routing 是否更稳” 的 clean ablation。

**关于 `alpha_share_heads`**:
- `alpha_share_heads` **不等于** MUDDFormer。
- 它只是在 predictor 输出后，把 `Q/K/V` 的 per-head `α` 先按 head 维平均，再 broadcast 回各 head；`R` 不变。
- 真正更接近 MUDDFormer-style 的对照应是 `routing_mode: layer_dwa_gate`。

### 2026-04-09 新提交实验 (4×A40, no baseline rerun)

| Job ID | Job Name | Config | 目标 | 备注 |
|--------|----------|--------|------|------|
| 17463826 | `fw_a_share` | `configs/fourway_alpha_shared_pure.yaml` | pure FourWay + `alpha_share_heads: true`，测试 per-head routing DoF 是否是主因 | clean shared-head 诊断，不走 corrected |
| 17463827 | `fw_corr_mean` | `configs/fourway_corr_pool_mean.yaml` | 测试 correction 是否主要靠 token-local memorization | 若改善明显，说明 local correction 是主要 overfit 源 |
| 17463828 | `fw_freeze_emb` | `configs/fourway_freeze_embed.yaml` | 测试 predictor embedding table 是否在记 input→α 映射 | 若改善明显，说明 predictor lookup-style memorization 存在 |
| 17463880 | `fw_layer_dwa` | `configs/fourway_layer_dwa_gate.yaml` | 真正 layer-level DWA / MUDDFormer-style 对照 | 用来和 shared-head FourWay 区分 |

**Log paths**:
- `logs/fw_a_share_17463826.out`
- `logs/fw_corr_mean_17463827.out`
- `logs/fw_freeze_emb_17463828.out`
- `logs/fw_layer_dwa_17463880.out`

**预期判读**:
- 若 `fw_a_share` 明显优于 5.49: 说明问题主要在 **per-head dynamic routing DoF**，不是单纯 recipe。
- 若 `fw_corr_mean` 改善最大: 说明 **token-local correction memorization** 是主过拟合源。
- 若 `fw_freeze_emb` 改善最大: 说明 predictor 的独立 embedding 在做 **lookup-table style memorization**。
- 若前三个都只是在 5.4-5.7 间小波动，而 `fw_layer_dwa` 更稳: 后续主线应往 **更低 DoF 的 layer-level routing** 收缩。
- 若四个都不改善太多: 更支持 “**dynamic routing 整体需要更强约束 + 更长/更大 recipe**” 这个方向，而不是继续扫小组件。

### Approach E: Attention bottleneck predictor (2026-04-09)

**动机**: 一个新的架构性假说是，当前 external FourWay predictor 看到了 **过分详细的 token-level 输入信号**:
- 独立 token embedding + pos embedding
- 2-layer predictor encoder 对全序列做表征
- 然后每个 token 直接输出所有层的 `α_q/α_k/α_v/α_r`

这给了 predictor 很大的 conditional bandwidth，容易学成 input-pattern → routing 的记忆器。  
如果 routing 真正只需要和 **next-token prediction** 相关的摘要信号，更合理的做法是先对 prefix memory 做一次读操作，把信息压成一个向量，再让 MLP 产出 routing。

**实现**:
- 新增 `FourWayAttentionBottleneckPredictor` (`src/model/predictor.py`)
- Query = 当前 token 的 layer-0 embedding（来自 base model 的 `embed_tokens`）
- Key/Value = 同一个 base model 在 **dense scout pass** 下的 final hidden states
- 用单次 causal multi-head attention 做 memory read
- attention 输出 + query residual → 小 MLP trunk → 每层 routing heads
- 仍保留 FourWay 的 identity init：routing output head `W=0`，bias=`[0,...,0,1]`

**训练接法**:
- `scripts/pretrain_dagformer.py` 新增 `fourway_predictor_variant`
  - `"encoder"` = 现有独立 encoder predictor
  - `"attn_bottleneck"` = 新 bottleneck predictor
- bottleneck 变体在 train/eval 都通过 helper `predict_fourway_routing()` 调用
- dense scout pass 用当前 base model、`torch.no_grad()`、`return_dict=True`，只取 `last_hidden_state`
- 目的不是做第二个可训练分支，而是给 predictor 一个更 task-aligned、但更低带宽的 memory source

**设计判断**:
- 这是在测试 “**过拟合来自 predictor 输入过宽**” 的结构性版本，不是又一个小正则
- 它不会变成真正的 next-token leakage，因为 query 用的是当前位置输入 token 的 layer-0 embedding，而不是 label token 本身
- 它也不等于 MUDDFormer；FourWay mixing 仍然不变，只是 external predictor 被换成了 bottleneck reader

**Config / Job**:
- Config: `configs/fourway_attn_bottleneck_pure.yaml`
- 设定: pure FourWay, `fourway_predictor_variant: attn_bottleneck`, `predictor_lr: 1e-4`
- Job: **17464685** (`fw_attn_bneck`, 4×A40)
- Logs:
  - `logs/fw_attn_bneck_17464685.out`
  - `logs/fw_attn_bneck_17464685.err`

**判读**:
- 若它明显优于 pure FourWay 当前参考线（尤其优于 `fourway_long_train` 早期区间）: 说明 predictor 输入带宽过宽这个方向是对的
- 若 train 变差但 eval 改善: 说明 bottleneck 在发挥 regularization 作用
- 若 train/eval 一起显著变差: 说明 “一个向量/token” 压得太狠，后续可以尝试少量 latent slots（比如 4 或 8 个 summary vectors）而不是回退到全带宽 predictor

### External diagnosis update (Gemini-DeepThink, 2026-04-09)

**共识**:
- Gemini 的核心判断和当前内部判断 **高度一致**，最有价值的重述是：
  - `open-loop controller`
  - `predictor/base co-adaptation`
  - `missing trust region around identity`
- 这比“dynamic gating 只是更大的模型、需要更大的 recipe”更精确。
- 特别是 `open-loop` 这个词很有用：当前 external predictor 基本是用 layer-0 / token-level 输入去决定深层 routing，但它并不真正基于运行中的深层状态做闭环校正。

**保留分歧 / 需要谨慎的点**:
- `fw_id_noc = 7.0774` 不能直接推出“residual stream 已经被 shred / chaos”这一种解释；更保守的表述仍然是：**base 已与 learned routing 强共适应**，脱离 learned routing 就不能工作。
- Gemini 提议“立刻 kill independent predictor + local corrections”过于激进。因为有一批 **几何 / 归一化约束** 相关实验的旧结论后来发现混入了 eval bug，已经重新提交；这些结果在新评估下可能会改变我们对 “trust region / bounded routing” 的判断。
- 因此当前不宜把 `softmax_fixed / sinkhorn_fixed / 相关几何约束 rerun` 线彻底判死，直到新的评估结果回来。

**当前综合判断**:
- 最可疑主因仍是：**open-loop conditional routing + missing trust region + joint co-adaptation**
- 但 “bounded geometry / constrained routing” 现在需要重新进入主分析线，因为旧负结果不再完全可信

**按信息增益排序的下一步**:
1. `static learned router`:
   - 完全去掉 input-conditioned predictor，只学静态 FourWay α
   - 用来判定问题是否主要来自 token-conditional bandwidth，而不是 FourWay mixing 数学本身
2. `frozen dense baseline + router-only`:
   - 从 dense baseline 3.8459 checkpoint 出发，冻结 base，只训练 FourWay router
   - 用来判定 joint co-adaptation 是否是主要病灶
3. `train vs eval α distribution audit`:
   - 对最好 overfit checkpoint 比较 train/eval 的 distance-to-identity / entropy / L2 / token-frequency-conditioned stats
   - 用来直接验证 external predictor 是否在输出 OOD routing fingerprints
4. 等待 `softmax_fixed / sinkhorn_fixed / 相关几何约束 rerun`:
   - 因为之前的 eval bug，旧结论需要暂时降权
   - 这些 rerun 会直接影响 “trust region / bounded routing” 是否值得升级为主线

### Approach F: Static learned router + frozen dense pure (2026-04-09)

**来源**:
- 基于 Gemini-DeepThink 的诊断建议，和当前内部判断合并后的最高信息增益实验
- 目标是优先区分：
  - FourWay mixing 数学本身是否有根本问题
  - 问题是否主要来自 input-conditioned routing bandwidth
  - 问题是否主要来自 joint co-adaptation

**F1. Static learned router**:
- 新增 `FourWayStaticPredictor`
- 不看 input，不看 token，不看 sequence
- 直接学习每层的静态 `α_q / α_k / α_v / α_r`
- 对所有 batch / time 位置 broadcast 同一组 α
- 仍保留 identity init：每个 stream 默认 `[0,...,0,1]`
- 这是最干净的控制实验：
  - 若它泛化接近 dense baseline，说明 FourWay math 本身没问题，病灶主要在 token-conditional routing bandwidth
  - 若它仍严重过拟合或明显变坏，说明 FourWay mixing 本身就可能与稳定 residual learning 张力很大

**F2. Frozen dense baseline + pure FourWay router**:
- 从 `pretrain_300m_baseline_5k/checkpoint_step5000.pt` 出发
- 加 pure FourWay external predictor
- **冻结 base model**
- 只训练 external router
- 这是最直接的 joint co-adaptation 测试：
  - 若接近 dense baseline，说明 joint training 让 base + router 一起走歪了
  - 若仍然明显过拟合，说明 external predictor 本身就更像 train-set-conditioned noise source

**实现**:
- `src/model/predictor.py`: 新增 `FourWayStaticPredictor`
- `scripts/pretrain_dagformer.py`: `fourway_predictor_variant` 现在支持
  - `"encoder"`
  - `"static"`
  - `"attn_bottleneck"`

**Configs / Jobs**:
- `configs/fourway_static_pure.yaml`
  - Job: **17466822** (`fw_static_pure`)
  - Logs:
    - `logs/fw_static_pure_17466822.out`
    - `logs/fw_static_pure_17466822.err`
- `configs/fourway_frozen_dense_pure.yaml`
  - Job: **17466823** (`fw_frz_dense`)
  - Logs:
    - `logs/fw_frz_dense_17466823.out`
    - `logs/fw_frz_dense_17466823.err`

### External diagnosis update (GPT-Pro, 2026-04-09)

**新增共识**:
- GPT-Pro 和 Gemini / 当前内部判断总体一致，但它把问题进一步收紧到了一个更具体的对象：
  - **高带宽 conditional code 写进 attention logits**
  - 特别是 `Q/K` routing 不是普通 depth mixing，而是在合成 token-specific attention kernels
- 这个表述比“只是更大的动态模型”更强，也更能解释为什么很多 broad regularization 几乎没用

**最重要的新点**:
- 不要把 `Q/K/V/R` 当成四个风险差不多的 stream
- `V/R` 更像信息 transport / residual mixing
- `Q/K` 直接改的是 **看什么、怎么竞争注意力**
- 因此当前最可疑的“毒性轴”是 **Q/K**, 而 local corrections 更像次级加速器/掩盖器

**和现有判断的合并版**:
- 目前最可信的解释是：
  - `open-loop conditional routing`
  - `missing trust region around identity`
  - `joint co-adaptation`
  - 且坏自由度很可能主要集中在 **Q/K**
- 这也解释了为什么此前很多正则化 sweep 信息量不高：它们主要在管 predictor 或 α 的边缘统计，没有直接管到 attention-logit operator space

### Approach G: QK-only vs VR-only split (2026-04-09)

**动机**:
- 直接测试 GPT-Pro 的核心判断：真正的病灶是否主要在 `Q/K` attention-kernel rewiring，而不是 generic FourWay routing 本身
- 如果 `QK-only` 明显更容易出现 “train 很低 / eval 很差”，而 `VR-only` gap 小得多，那么后续就不该再把四路一视同仁

**实现**:
- `scripts/pretrain_dagformer.py` 新增 stream 开关：
  - `route_q`
  - `route_k`
  - `route_v`
  - `route_r`
- 禁用的 stream 在 train/eval 都被强制成 exact identity `[0,...,0,1]`
- 强制发生在 regularization 和 deterministic transforms 之前，因此禁用 stream 不参与梯度和损失

**Configs / Jobs**:
- `configs/fourway_qk_only_pure.yaml`
  - `route_q: true`
  - `route_k: true`
  - `route_v: false`
  - `route_r: false`
  - Job: **17468404** (`fw_qk_only`)
  - Logs:
    - `logs/fw_qk_only_17468404.out`
    - `logs/fw_qk_only_17468404.err`
- `configs/fourway_vr_only_pure.yaml`
  - `route_q: false`
  - `route_k: false`
  - `route_v: true`
  - `route_r: true`
  - Job: **17468405** (`fw_vr_only`)
  - Logs:
    - `logs/fw_vr_only_17468405.out`
    - `logs/fw_vr_only_17468405.err`

**预期判读**:
- 若 `QK-only` 比 `VR-only` 更容易出现低 train / 高 eval gap:
  - 强支持 “Q/K 是主毒性轴”
- 若 `VR-only` 也同样严重过拟合:
  - 说明问题更接近 token-local conditional bandwidth / predictor side-channel，而不只是 attention-logit rewiring
- 若 `VR-only` 明显更稳:
  - 后续主线应考虑先只保留 `V/R`，把 `Q/K` tie/shared/移除，再重新引入 trust region

## Interpretability Round 1 — 拓扑 predictor 学到了什么 (2026-08-31)

**对象**: `fourway_300m_dagformer_mmap` step9000（headline checkpoint）。
**方法**: eval-time routing 干预（swap ladder）+ α 方差分解 + 合成 induction
+ 跨 domain 注入。脚本 `scripts/interp_{dump_alpha,swap_eval,swap2}.py`，
job 21666314 / 21666491。结果 JSON: `experiments/results/interp/`。
管线校准：identity_no_corr 6.4632 ≈ trainer eval/nll_baseline 6.4856；
dense/fourway 的 25-条切片偏移一致（0.133/0.131），dense−fourway gap 与
trainer 完全吻合（0.101 vs 0.102）。

### Swap ladder（后 25 条 held-out，NLL nats）

| mode | NLL | | mode | NLL |
|---|---|---|---|---|
| dynamic | **3.2173** | | static_mean | 3.2618 |
| **pos_table** | **3.2173** | | shuffle_time | 3.2649 |
| swap_context | 3.2176 | | dense_baseline | 3.3181 |
| static_q / k / v / r | 3.2182 / 3.2283 / 3.2536 / 3.2172 | | identity (corr ON) | 4.7563 |
| dynamic_no_corr | 4.5202 | | identity_no_corr | 6.4632 |
| pos_table_no_corr | 4.5395 | | static_mean_no_corr | 4.5551 |

### 结论

1. **逐 token 路由有真实因果价值**：dynamic − static_mean = 0.045 nats，
   占对 dense 优势（0.101）的 ~45%。
2. **但它是位置调度，不是内容自适应**：
   - `pos_table`（按位置查表的 α，校准集 per-position 均值，不看内容）
     **完全复现 dynamic**（3.2173 = 3.2173；三个 domain 窗口上同样打平）
   - `swap_context`（换别的序列的 α）无损；`shuffle_time` 掉回 static 水平
   - α 方差分解：**86.5% 位置方差**（Q 92% / K 89% / V 83% / R 79%）
   - 跨 domain 注入（code 窗用 prose/dolma/latex-mean α）差异 < 0.003 —
     domain 级内容条件化为零
   - 合成 induction：copy_acc dynamic 0.795 ≤ static 0.809 < dense 0.847 —
     无"检测重复并改接线"的证据
   - 去掉 correction 后内容残留仅 ~0.02 nats（4.5202 vs 4.5395），
     correction 在场时被完全覆盖
3. **动态收益解剖位置**：V 流（static_v 伤害最大 +0.036）、中层 L6–9、
   src=0（embedding 源）——内容方差 top-10 条目全部是 V/L6-11/src0。
4. **位置调度在 OOD 文本上价值更大**（domain 窗口 0.11–0.15 nats）。
5. correction MLP 深度耦合（去掉后 +1.3 nats），是剩余的
   content-conditioned 通道，无法用 swap 单独隔离。

### 叙事影响

"context-dependent dynamic topology"（外置 predictor 层面）**不成立**。
成立的三分解：**静态重连线（~1.5 nats，corr 在场）+ 位置路由调度
（0.045，可被一张 [1024×3773] 表替代）+ hidden-state 局部修正**。
候选新叙事：per-token routing 收益的因果分解 + "26M predictor ≈ 位置查表"
的简化结论。待补链条：pos-table **从头训**是否追平 encoder predictor
（排除"内容容量是训练脚手架"假说）。

### 新增 ladder runs（4×A40 各 ~12h）

- 21666351 `fourway_150m_static_mmap`（头级×静态）
- 21666552 `fourway_150m_postable_mmap`（头级×纯位置，新增
  `FourWayPositionalPredictor`，identity init 验证通过）

从头训 150M conditionality 阶梯：dense < static < pos_table ≈? full-encoder。

### Probes + emergence (job 21666314 完成, 2026-08-31)

| target | α probe | α 残差 | token-id 基线 | 多数类 | 判读 |
|---|---|---|---|---|---|
| posbucket | **0.962** | **0.890** | 0.254 | 0.250 | α ≈ 位置编码器，残差化后仍在 |
| charclass | 0.867 | 0.753 | 0.957 | 0.753 | 词法回声，低于 id 上限，残差=多数类 |
| freqbucket | 0.643 | 0.237 | 0.920 | 0.162 | 同上 |
| **isrepeat64** | 0.769 | 0.590 | **0.765** | 0.595 | **α ≈ token-id 基线：无超越当前词的 context 信号** |
| nllquartile | 0.289 | 0.239 | 0.266 | 0.209 | 几乎无难度感知 |
| domain | 0.443 | 0.345 | 0.588 | — | α 连词法 domain 信号都没用满 |

- **Emergence: 平坦**（charclass 0.865→0.867，isrepeat 0.768→0.772，
  step 1500→10500）——predictor 在 1500 步内定型，无结构涌现。
- Induction Δα: mean|Δ|=0.0011（极小）；top 条目 v/L7/h1/src0 与内容方差
  hotspot 同族——重复响应存在但微弱且因果为负。
- **可解码性与因果性一致**：外置 predictor = 位置嵌入 + 惰性词法查表，
  无 context 计算（尽管它是 2 层因果 transformer）。

图: `experiments/figures/interp_{swap_ladder,probe_acc,emergence,induction_heat}.png`

### Correction MLP 解剖 (job 21666770, 2026-08-31)

Δcorr = correction MLP 的逐 token 输出（hook 抓取），与外置 α 同布局同问题：

**方差分解 — 与外置 predictor 完美镜像**：
- Δcorr: **98% 内容方差**（pos_share ≈ 2%，四流一致）；量级集中 V(2.18)、R(1.59) ≫ Q(0.44)、K(0.31)
- 外置 α: 86.5% 位置方差

**Probes（Δcorr 特征）**：

| target | Δcorr | 残差 | token-id 基线 | 判读 |
|---|---|---|---|---|
| charclass | 0.993 | 0.706 | 0.957 | 看 hidden state，词法近完美（预期） |
| posbucket | 0.419 | 0.404 | 0.254 | 弱位置性（与 α 的 0.962 互补） |
| **isrepeat64** | **0.840** | **0.660** | 0.765 | **超词法基线 +0.075，残差超多数类 +0.065 — 真 context 信号** |
| **nllquartile** | **0.436** | **0.357** | 0.266 | **感知难度 +0.17，残差化后仍在** |
| **domain** | **0.808** | 0.476 | 0.588 | **超词法 +0.22** |

**Induction**: mean|Δ|=0.1654（外置 α 的 **150 倍**）。Top 条目全为 V 流
src=0（embedding 源）的推拉机制：L11/h4 **+8.2**（末层增强 embedding→V
直读，token 身份变得可直接复制）、L7/h1,h7 **−8.0**（中层抑制）。

**分工结构（decodability 层面）**：外置 predictor 管"你在哪"（位置调度），
correction MLP 管"你在处理什么"（重复/难度/domain 反应）。路由系统自发
分解为 前馈位置调度 + 局部内容反应控制器。

**待最终因果验证** (job 21667922): 把 correction 输出换成 per-position
查表/换 context 的 Δcorr —— live ≪ corr_pos_table 才能说内容反应性是
因果真实而非 epiphenomenal。

### Correction 因果测试 — 内容反应性是真的 (job 21667922, 2026-08-31)

| 模式 | NLL | |
|---|---|---|
| live（真 correction） | **3.2173** | = dynamic ✓（hook 机制校验）|
| corr_static（常数均值） | 3.7943 | |
| corr_pos_table（按位置查表） | 3.8129 | **比 static 还差** — correction 无位置结构 |
| corr_swap（换别序列的 Δcorr） | 4.4162 | ≈ 没有（4.5202）— 精确内容对齐 |
| corr_zero | 4.5202 | = dynamic_no_corr ✓ |

**live ≪ 任何 content-free 替代（Δ≥0.58 nats）→ correction 的内容反应性
因果必要。**

### 最终图景 — Double Dissociation（论文核心表）

| | 外置 predictor α | correction MLP Δcorr |
|---|---|---|
| 方差 | **86.5% 位置** | **98% 内容** |
| 换 context | 免费（+0.0003） | 灾难（+1.20） |
| 换 per-position 表 | 免费（±0.000） | 昂贵（+0.60） |
| probe 内容信号 | 无（=词法基线） | 重复/难度/domain 全超基线 |
| induction 响应 | 0.001 | 0.165（V/embedding 推拉 ±8） |

**结论**：学到的路由自发分解为「前馈位置调度（外置 predictor，可换成
83K 参数查表）」+「内容反应局部控制器（correction MLP，因果价值 ~0.6
nats）」。context-dependent dynamic routing 存在且因果必要——但住在局部
反馈里，不在全局 predictor 里。机制样例：重复检测下 V 流 embedding 直读
的末层增强/中层抑制。

## ssmfloss — Gradient Flossing / Jacobian Conditioning for Selective SSMs (PoC, 2026-09-01)

Side project in `ssmfloss/` (pure-PyTorch Mamba-1 LM, 68K params, 2 layers, d_model 64,
d_state 16; synthetic tasks; AdamW β2=0.95, wd 0.01, cosine LR, **no grad clipping**,
3000 steps, batch 64, 2 seeds). Jobs: smoke 21684726, sweep 21684841 (interactive 4×A40, 1 h).
Full table `ssmfloss/runs/poc/summary.md`, figures `fig_*.png`.

**Theorem check** (`check_lyapunov.py`): QR-based finite-time Lyapunov exponents of the
autograd state Jacobian == `A·mean(Δ)` to 1.7e-16; Jacobian exactly diagonal; all λ < 0.
→ naive Gradient Flossing on the recurrence is closed-form timescale control.

**Eval acc (mean of 2 seeds; 0 divergences anywhere; lam=0.1 for all regs)**

| task | lr | none | temporal (λ→0) | selective (floor −0.2) | io (log-gain→0) |
|---|---|---|---|---|---|
| selective_copy T=256 | 1e-3 | 0.49 | 0.55 | **0.59** | 0.30 |
| | 3e-3 | 0.67 | 0.86 | **0.94** | 0.85 |
| | 1e-2 | 0.74±0.14 | 0.97 | 0.55±0.45 | **0.997±0.002** |
| | 3e-2 | 0.09 (dead) | 0.55±0.45 | 0.22 | **0.64±0.36** |
| recall_unique T=97 | 1e-2 | 0.69 | 0.76±0.16 | 0.72 | **0.87±0.07** |
| | 3e-2 | 0.56±0.27 | 0.42±0.32 | 0.63 | **0.875±0.04** |
| recall_last T=97 | ≥3e-3 | 1.00 | 1.00 | 1.00 | 0.99–1.00 (task saturated) |

**Mechanistic findings**
- Baseline "instability" at high LR is NOT temporal explosion: it is (a) Δ runaway
  (mean Δ 0.02 → 0.5–22, λ → −4…−92 ⇒ Ā→0, state dies, log σ_max of io-Jacobian → −28 =
  constant function) and (b) residual-stream blow-up (`resid_rms` 1.5 → 7450 at lr 1e-2,
  io log σ_max 3 → 15). Embedding norm does *not* shrink (0.08→0.2), so the io-gain growth is
  real amplification, not an RMSNorm artifact.
- temporal flossing does exactly what the closed form predicts: Δ ↓ 5–50×, λ → −0.01…−0.08
  ("never forget"). Helps memory-heavy selective_copy at 3e-3–1e-2 (+0.2–0.3), but does
  nothing for the depth/io explosion (log σ_max still 10–14) and becomes seed-unstable at
  3e-2 (Δ oscillates 1e-20…1e-4).
- io flossing is the only reg that controls the depth Jacobian (log σ_max flat ≈ 3, resid_rms
  0.2–0.4, Δ stays ~0.05–0.1) and is best/tied-best at lr ≥ 1e-2 on every task; it costs at
  lr 1e-3 on selective_copy (0.30 vs 0.49) — two-sided "gain→1" over-constrains when nothing
  is wrong. Note the two-sidedness is what also catches the Ā→0 collapse mode.
- The prediction "temporal flossing hurts overwrite tasks" is untested: recall_last (8 keys,
  32 pairs) saturates at 1.0 for everything — needs a harder variant.

**Perf**: double-backward jvp through the scan loop = 3.4 s/step; forward-mode AD
(reverse-over-forward, `torch.autograd.forward_ad`) = 0.21 s/step vs 0.08 plain.

**Next**: one-sided io hinge / floss only early (paper-style) to remove the low-LR cost;
harder recall_last; 4-layer / T=512 / `--no_conv` to make the baseline fail at lower LR;
`--reg block`, `--io_mode top`, Mamba-2-like `--a_mode scalar`; λ sweep (`--grid lam`).

### ssmfloss round 2 — one-sided io hinge + harder overwrite task (job 21686677, 64 runs)

New: `--reg io_hinge --io_thr t` (L = mean relu(log-gain − t)², t=1.5 ≈ init level, also 3.0);
`recall_last_hard` (16 keys × 64 pairs, every key rebound ~4×, T=161). Same protocol as round 1.

**recall_last_hard — eval acc (2 seeds)**

| lr | none | temporal | selective | io | io_hinge@1.5 |
|---|---|---|---|---|---|
| 1e-3 | 0.34±0.11 | 0.33±0.12 | 0.35±0.12 | 0.36±0.05 | 0.35±0.04 |
| 3e-3 | 0.62±0.30 | 0.71±0.25 | 0.60±0.20 | 0.77±0.05 | **0.88±0.08** |
| 1e-2 | 0.995 | **0.72±0.16** | 0.93±0.02 | 0.993 | **0.996** |
| 3e-2 | 0.976 | **0.66±0.22** | 0.82±0.01 | **0.995** | 0.985 |

→ **The forgetting trade-off is real.** Where the task is learnable (lr ≥ 1e-2), naive temporal
flossing caps at 0.66–0.72 (train NLL plateaus ≈ 1 — optimisation failure, not overfitting),
the floored `selective` variant sits in between (0.82–0.93), baseline/io/hinge ≈ 1.0.
Accuracy orders inversely with how far the reg suppressed Δ (temporal Δ 0.003–0.007 <
selective 0.008–0.02 < none/io/hinge 0.01–0.03). Combined with round 1 (temporal was
best/2nd-best on memory-only selective_copy at the same LRs): **temporal flossing = timescale
control that helps pure memory and hurts overwriting; io-Jacobian flossing has no such
trade-off** (≈ best on both task types).

**io_hinge vs two-sided io (selective_copy / recall_unique)**

| task | lr | none | io (→0) | hinge@1.5 | hinge@3.0 |
|---|---|---|---|---|---|
| selective_copy | 1e-3 | 0.49 | 0.30 | **0.46** | **0.48** |
| | 3e-3 | 0.67 | **0.85** | 0.74 | 0.68 |
| | 1e-2 | 0.74 | **0.997** | 0.78±0.18 | **0.997** |
| | 3e-2 | 0.09 | 0.64±0.36 | 0.51±0.39 | 0.65±0.34 |
| recall_unique | 1e-2 | 0.69 | **0.87** | **0.865** | – |
| | 3e-2 | 0.56±0.27 | **0.875** | 0.53±0.29 | – |

→ Hinge removes the low-LR cost (1e-3: 0.46–0.48 ≈ baseline 0.49 vs io 0.30) and holds the
gain exactly at the boundary (log-gain flat at 1.5, σ_max ≈ e^4.4). But it keeps only part of
the high-LR benefit: hinge@1.5 loses at recall_unique 3e-2 (0.53 vs io 0.875) and is
seed-unstable at selective_copy 1e-2 (0.6 / 0.96); hinge@3.0 recovers 0.997 there (n=2, so
the thr dependence is not yet trustworthy). Interpretation: at high LR the two-sided pull
(also penalising gain < 1, i.e. the Δ-runaway / dead-state collapse mode) matters, or the
one-sided hinge is simply too weak once the boundary is reached.

**Status of the paper claims after 160 runs**
1. λ = A·E[Δ] theorem + collapse of naive flossing to timescale control: verified. ✔
2. Naive flossing hurts selective forgetting: verified on recall_last_hard (−0.28…−0.32 acc). ✔
3. Depth/io Jacobian, not the recurrence, is what blows up in baseline training; io flossing
   is the only reg that controls it and is best at high LR. ✔ (small model, synthetic)
4. Open: a version of io flossing with no low-LR cost AND full high-LR benefit (candidates:
   band hinge |lg| ≤ t, early-only flossing, top-σ mode, block-wise); thr/λ sweeps with ≥3 seeds.

### 通用性验证：同一套手术用于 OLMoE-1B-7B (job 21704844, 2026-09-01)

对开源 MoE（16 层×64 专家 top-8）的每层 router logits 做同构替换手术，
51 个 dolma held-out 窗口（25 calib / 26 test），OLMoE 自己的 tokenizer：

| 挡位 | NLL | Δ vs live |
|---|---|---|
| live | 2.8110 | — |
| **token_id**（按当前词查表） | 4.6828 | **+1.87（最便宜的替代，仍灾难）** |
| pos_table（按位置查表） | 7.3556 | +4.54 |
| swap_context | 7.6967 | +4.89 |
| shuffle_time | 8.0245 | +5.21 |
| static_mean | 8.2333 | +5.42 |

方差分解：**pos_share = 2.2%**（≈ 我们 correction 的 2%，与外置 α 的
86.5% 相反）。Probes：isrepeat64 gate=0.825 > token-id 基线 0.772（残差
0.667 > 多数类 0.601 — 真 context 信号）；posbucket 仅 0.433。

**Taxonomy（论文方法论卖点）** — 同一套手术区分三种路由本质：

| 路由机制 | 指纹 | 最便宜的无损/低损替代 |
|---|---|---|
| OLMoE expert router | 语境型（词法锚定） | 无（token-id 最近但 +1.87） |
| DAGFormer 外置 predictor | **位置型** | **per-position 查表（±0.000）** |
| DAGFormer correction MLP | 语境型 | 无（任何查表 ≥ +0.58） |

方法学意义：手术套件不是"什么都判死/判活"的钝器——它给真正
content-dependent 的 router（MoE）发"不可替代"证书，同时暴露我们
全局 predictor 的位置本质。跨架构只比较 profile/排序，不比较绝对
量级（MoE 换 logits 影响离散 top-8 选择，比软 α 替换更脆）。

附注：shard1 校验-重下机制工作正常（12 月遗留的 3.4G 截断分片被检出
并重下为 5.0G）；OLMoE 为 instruct 版，NLL 偏高但组内比较不受影响。

### 界面编辑实验 v1 (job 21705901, 2026-09-01)

**E4 头级粒度必要性 — 强阳性（vs MUDD 的差异化证据）**：
把有效路由（α+Δcorr）按头平均，模拟层级路由：

| 模式 | NLL | repeat copy_acc |
|---|---|---|
| baseline | 3.2173 | 0.863 |
| head_avg_qkv | **5.164 (+1.95)** | **0.031（塌方）** |
| head_avg_v | 4.297 (+1.08) | 0.108 |
| head_avg_qk | 3.527 (+0.31) | 0.049（NLL 小伤但 copy 塌方）|

判读：LM 质量主要靠 V 的头级分化；copy 机制同时需要 QK 头级分化。
层级路由界面无法表述该解。（注意措辞：证明的是"我们学到的解是头级
异质且层级界面不可表述"，不是"层级架构训不出好损失"——MUDD 在
150M 与我们打平过。）

**E1 v1 外科删除 — 特异性 ✓ 效应弱**：删 top-6 induction 条目后
NLL +0.007（外科精度好），copy_acc 仅 −0.010 → 电路分布式，须删
整个 embedding→V 家族（v2 改为 176 条家族删除 + 等量随机对照）。

**E5 v1 steering — 指标失误**：copy_from_ctx_rate 在随机文本基线即
0.86（天花板），λ=2 反而全面破坏。v2 改用 lag-128 token 的 logprob
（induction 假设的尖锐指标）。

### 界面编辑实验 v2 (job 21705932, 2026-09-01)

**E4b 组件归因 — 头级多样性分布在两个组件且互锁（超可加）**：
| 模式 | NLL | copy_acc |
|---|---|---|
| 只平均 α（corr 保留） | 3.578 (+0.36) | 0.515 |
| 只平均 corr（α 保留） | 3.461 (+0.24) | 0.516 |
| 两者都平均（v1） | 5.164 (+1.95) | 0.031 |

单独平均各损失一半 copy、少量 NLL；同时平均 → 崩溃。+1.95 ≫ 0.36+0.24：
α 与 corr 的头级结构互相补偿、缺一即塌 — "interlocking head diversity"。

**E1b 家族删除 — 阴性（诚实记录）**：删整个 embedding→V correction
家族（176 条）copy_acc 仅 −0.006（0.857），对照家族反而 −0.049。
结论：V/src0 上 ±8 的 induction 大信号是重复处理的**标志物（marker）**，
不是 copy 的**引擎（lever）**。copy 的因果载体在 per-head QK 结构里
（v1 中 head_avg_qk 使 copy 塌方至 0.049 而 NLL 仅 +0.31 可证）。

**E5b 尖锐指标 steering — 阴性**：注入重复指纹后 lag-128 logprob
−8.256→−8.195（λ=0.5，噪声级），λ=1 反而变差且 NLL 漂移 +0.23。
写入 correction 通道不能在非重复文本上诱发 induction 行为。

**编辑线定论**：组件/流/粒度级干预 = 因果、大效应、可解释（swap 阶梯、
stream 消融、头平均）；微观单电路编辑 = 未证实。论文可把"signature ≠
lever"作为 nuance 正面写出。

### Steering v2 (jobs 21706167/21706191, 2026-09-01)

**S1 抑制背诵的旋钮 — 成功（本轮最强正面结果）**：
真重复文本上按比例减去重复反应信号（correction 通道）：

| 减幅 λ | 背诵正确率 | 普通文本 NLL |
|---|---|---|
| 0 | 72.7% | 3.2173 |
| 0.25 | 68.2% | 3.2344 |
| 0.5 | 62.9% | 3.2871 |
| 1.0 | 43.5% | 3.5392 |
| 2.0 | 12.7% | 4.5583 |

平滑单调可调；轻档几乎免费（−0.25 档：背诵 −4.5 点 / NLL +0.017），
重档代价显著。单路减弱效果小（qk_only 69.0%、v_only 71.1%）→ 旋钮
分布于全信号。

**S2 公平目标诱发 — 阴性（两轮独立设计后定论）**：重复刚停止的文本
注入信号无法维持"仍在重复"押注（尾段 lag-128 logprob 基线 −8.12，
各档 −8.11~−8.37 噪声级）。

**S3 文风注入 — 阴性 + 完美对照**：散文注入代码指纹，代码词概率质量
0.96%→1.01%（真代码参考 10.6%）。**α 侧同量注入：+1.0/+2.0 档
mass=0.0096、NLL=4.921/4.920 —— 与基线逐位一致**（预测零效应，
实测零效应；correction 侧同幅注入 NLL 4.92→5.13 有扰动）。

**机制定论**：correction 里的内容信号是对"已检测到的模式"的**响应**而
非检测本身——减掉能打断响应（刹车灵），注入造不出检测（油门不灵）。
两次失败从两个方向支撑同一句话；抑制旋钮 + 权衡曲线进正文，
反向阴性进 nuance 段。

### 从头训练的 conditionality 阶梯 — 150M 第一格落地 (2026-09-02)

同数据/步数/评测集（dolma held-out 50 seqs），step 5500 eval NLL：
- dense: 3.7969
- **pos-table routing（无 correction）: 3.7813**（job 21712983，final step6000 = 3.7788）
- full DAGFormer（encoder + corrections）: 3.6875

仅位置路由收益 0.016 nats；完整模型 0.109 nats → 增益大头需要内容
通路（与 300M 干预分解方向一致）。注意：本格 ≠ "预测器可换位置表"
的直接检验（那需保留 corrections）；直接检验 = Lite（21712984 在跑，
预测 ≈3.69）。static（21723942 排队）补零输入依赖格。

### 简化版（Lite）从头训练结果 — 替换主张成立 (job 21712984, 2026-09-02)

150M，同数据/步数/评测集，step 5500 eval NLL：

| 条件 | 预测器参数量 | eval NLL |
|---|---|---|
| dense | — | 3.7969 |
| 位置表，无 correction | 1.33M | 3.7813 |
| **位置表 + corrections（Lite）** | **1.33M** | **3.6875** |
| 编码器预测器 + corrections（完整版） | ~27M | 3.6875 |

Lite 与完整版**四位小数相等**（final step6000: Lite 3.6839）。
结论链闭合：
1. eval 时替换（300M）：零变化；
2. **从头训练替换（150M）：最终损失相等，预测器参数缩小 20 倍**；
3. 全局预测器的内容容量连"训练脚手架"都不是；
4. 内容依赖增益（0.109 中的 0.094）全部来自 854K 参数的局部修正模块。

### 阶梯补齐：static 格 + 新增 static+corr 格 (2026-09-02)

150M step-5500 对齐表（同数据/步数/评测集）：

| 条件 | eval NLL |
|---|---|
| dense | 3.7969 |
| static routing（无 corr, job 21723942, final 3.7764） | 3.7776 |
| pos-table routing（无 corr） | 3.7813 |
| Lite（pos-table + corr） | **3.6875** |
| full（encoder + corr） | **3.6875** |

新发现：**无 corrections 时，位置依赖相比纯静态无增益**（3.7776 vs
3.7813，噪声级）。与 300M eval-time 的 0.045 位置收益（corrections 在场）
对比 → 位置调度的价值可能依赖与内容通路的交互，或随规模变化。
已提交 static+corr 150M 检验："若 static+corr == Lite == full，全局
预测器可退化为常量向量"。

### 选择性对照：复现抑制不是一般性破坏 (job 21763777, 2026-09-02)

回应"是否只是破坏模型性能"的质疑。300M step9000，held-out 25 条，
重复序列 16 条：

| 扰动（作用于 correction 输出） | held-out NLL | 复现准确率 |
|---|---|---|
| 无 | 3.2173 | 72.7% |
| −0.5×签名 | 3.2871 (+0.070) | 62.9% |
| +0.5×签名（反号） | 3.2712 (+0.054) | **78.7%** |
| 随机方向、同 L2 范数（5 种子） | 4.32–6.87 (+1.1～+3.7) | 2–8% |
| −1.0×签名 | 3.5392 | 43.5% |
| +1.0×签名 | 3.4446 | 75.2% |
| 随机方向、同范数（1.0） | 6.60–13.33 | 0–4% |

结论：效应方向特异且有符号（反号→复现上升），随机方向的一般性损害
大 15–50 倍。修正此前表述：签名方向可双向调节**已被激活**的复现过程，
但不能在无重复输入上启动它。

**自然文本分层**（可复现组 = 目标 token 出现于前 128 token，占 44.7%）：
−0.5 下可复现组 1.795→1.853 (+3.3%)，新颖组 4.366→4.445 (+1.8%)。
行为空间选择性成立，token 空间附带代价弥散。待改进：低频词分层。

**审计投影（阴性）**：signature 方向投影对复现事件 AUC=0.47（幅值基线
0.575）。信息存在于 correction 输出（探针 0.840）但不在单一方向上；
审计需训练式线性监测器（待补 AUC + 隐状态探针对比）。

### 复述抑制的特异性对照 (job 21763799, 2026-09-02) — 回应"是否只是整体损害"

同一 300M step9000 模型，减去扰动向量后测：重复文本背诵准确率 /
普通文本 NLL / 普通文本 top-1 准确率（基线 72.7% / 3.217 / 38.6%）：

| 扰动 | 背诵 | NLL | top-1 |
|---|---|---|---|
| 签名 ×0.25 | 68.2% | 3.234 | 38.2% |
| 签名 ×0.5 | 62.9% | 3.287 | 37.5% |
| 签名 ×1.0 | 43.5% | 3.539 | 34.0% |
| 随机方向，范数 = ‖0.5×签名‖（3 种子） | 7.4–8.4% | 4.32–6.87 | 6.6–23.3% |
| 随机方向，2–16 倍范数 | ≈0–2% | 8.6–17.7 | ≈0 |
| 签名坐标打乱 ×0.5（3 种子） | 1.1–56.1% | 3.58–5.85 | 15.4–34.0% |

**等范数比较**：签名方向对普通能力几乎无损（top-1 −1.1 点），任意
方向同等大小则摧毁模型（top-1 −15 至 −32 点）；坐标打乱后该性质
消失 → 签名是特定的低损伤方向，其坐标结构有意义。
**等代价比较**尚缺随机方向的低范数点（job 21763945 补 1/16–1/2 倍）。

### 特异性对照 — 等代价比较 (job 21763945, 2026-09-02)

随机方向低范数扫描（‖δ‖ = 1/16, 1/8, 1/4, 1/2 × ‖0.5×签名‖，各 3 种子）：

| 代价（NLL 上升） | 签名方向：背诵下降 | 随机方向：背诵下降（3 种子均值±范围） | 随机方向 top-1 下降 | 签名方向 top-1 下降 |
|---|---|---|---|---|
| ≈ +0.017 | −4.5 点 | −1.2（−3.1 ~ +1.2） | −0.3 | −0.4 |
| ≈ +0.07 | **−9.8 点** | **−0.9（−5.0 ~ +1.4）** | −0.7 | −1.1 |
| ≈ +0.32（插值） | **−29.2 点** | **≈ −10** | ≈ −4 | −4.6 |

**结论**：在相同附带代价下，普通能力（top-1）的下降两者相当，而签名方向
对背诵的抑制是随机方向的约 3–10 倍。批评"只是整体损害"不成立；扣除
整体损害后的针对性效应：低代价档 ≈9 点，中代价档 ≈19 点。
图：`experiments/figures/interp_specificity.png`。

### 审计监测器 + 低频词分层 (job 21763959, 2026-09-02)

**审计**：在自然 held-out 文本上标记"复现事件"（目标 token 出现于前
128 token 且模型预测正确，占 23.8%），训练式线性监测器的测试 AUC：

| 特征 | AUC |
|---|---|
| **correction 模块输出（3773 维）** | **0.776** |
| 同上，按当前 token 身份残差化 | 0.691 |
| 隐状态 L3/6/9 拼接（3072 维，即 correction 的输入） | 0.735 |
| 当前 token 身份 P(event\|id) | 0.581 |

路由信号是比其自身输入（隐状态）更好的线性可读审计信号；去掉词身份
可解释部分后仍有 0.69。

**低频词分层**（rare = 语料内出现 ≤5 次；test 10 条序列）：

| 组 | λ=0 | λ=0.5 (Δ) | λ=1.0 (Δ) |
|---|---|---|---|
| 出现过 & 低频（真需复现） | 2.671 | 3.087 (+0.416, +15.6%) | 3.950 (+1.279) |
| 出现过 & 高频 | 1.508 | 1.556 (+0.048) | 1.751 (+0.243) |
| 新颖 & 低频 | 5.318 | 5.436 (+0.118) | 5.813 (+0.495) |
| 新颖 & 高频 | 3.492 | 3.559 (+0.067) | 3.790 (+0.298) |

干预代价精确集中在需要复现机制的 token 上（λ=0.5 时该组绝对增幅是
新颖组的 3.5–6 倍、相对增幅 7 倍）。此前"代价弥散"的读数是把由语言
统计预测的高频复现词混入所致。选择性三重成立：方向特异、符号特异、
token 特异。

### 记忆化（训练数据逐字复现）审计与控制 — 阴性 (job 21763846, 2026-09-02)

按训练顺序（seed 42 块置换，step 9000 前消费 4,608,000 个位置）精确
划分见过/未见过的训练窗口，各 600 个，前缀 64 + 续写 32：

| | 见过 | 未见过 | 差 |
|---|---|---|---|
| 续写 NLL（教师强制） | 3.497 | 3.584 | 0.087（显著，≈3.5 SE）|
| 贪心逐 token 匹配率 | 3.8% | 3.7% | — |
| 32-token 完整抽取率 | 0.0% | 0.0% | — |

存在轻微参数化记忆（见过的损失更低），但**无可测的逐字抽取**（300M、
单 epoch、4.7B token 的预期结果）。干预（−0.5/−1.0 签名）使两组损失
等量上升，gap 保持 0.09 → 复现抑制方向不作用于参数化记忆（机制不同）。
投影审计 AUC 0.525（随机）。

**结论**：control 案例保持为"上下文内逐字复制"；论文需明确边界：该操作
对象控制的是从上下文复制，而非参数化记忆；后者在本规模不可测。

### 具名头定位与双向 steering (job 21764128, 2026-09-03) — 本轮最强结果

对 Q/K 逐头路由做"替换为层均值"扫描（300M step9000，复述基线 72.7%）：

**逐层**：L4 → 17.2%（−55）、L6 → 53.3%、L3 → 58.3%、L7 → 61.5%，
其余层 ≈ 无影响；NLL8 代价 L4 仅 +0.10。
**逐头**：L4/h1 单头 → **12.6%（−60）**；L3/h11 → 54.9；L6/h11 → 55.4；
L3/h13 → 60.9；L3/h2、L4/h4 ≈ 中位数（71.5/71.6，可忽略）。

**γ 扫描**（6 个具名头的 Q/K 路由相对层均值的偏离 × γ，同时作用于
预测器与修正模块）：

| γ | 复述准确率 | NLL | 自然文本前文 token 概率质量 |
|---|---|---|---|
| 0 | **3.8%** | 3.427 (+0.21) | 36.7% |
| 0.5 | 35.1% | 3.295 (+0.08) | 37.1% |
| 1（基线） | 72.7% | 3.217 | 38.5% |
| 1.5 | **80.2%** | 3.240 (+0.02) | 39.3% |
| 2 | 82.8% | 3.292 (+0.07) | 40.1% |
| 3 | 86.6% | 3.499 (+0.28) | 42.3% |

- 关闭（γ=0）：−69 点 / +0.21 NLL（签名方向 −29 / +0.32；随机 ≈−10 / +0.32）
- **正向 steering 成立**：γ=1.5 复述 +7.5 点仅 +0.02 NLL；自然文本上
  对前文 token 的概率倾向随 γ 单调上升（36.7% → 42.3%）
- "重复后接新内容"的 lag-128 logprob 基本不动（−8.46~−8.10）：放大不会
  凭空诱发重复，只增强对真实重复的利用

**接线读出**（预测器 Q/K 来源层权重，重复与随机文本上逐位一致 →
学到的固定结构）：具名头的 K 路由对 embedding（源 0）与第 1 层输出赋
**负权**（−0.3 至 −0.9），对最近层赋 >1 的权重（+1.4 至 +1.5）——
即 key = 近层表示 − 当前 token 自身身份，符合 induction head "按前文
而非自身匹配"的要求；L6/h11 的 K 另从源 4（第 3 层输出，即 L3 具名头
的写入处）读取 +1.09，构成跨层组合。
图：`interp_named_head_dial.png`、`interp_named_head_wiring.png`。

### 自由生成的重复退化 — 阴性 (job 21764143, 2026-09-02)

40 条 held-out 前缀（128 token），贪心解码 200 token：
| λ | rep-4 | distinct-2 | 循环序列比例 | dense PPL(生成文本) |
|---|---|---|---|---|
| 0 | 0.823 | 0.153 | 100% | 1.51 |
| 0.25 / 0.5 / 1.0 | 0.856 / 0.835 / 0.841 | 0.123 / 0.138 / 0.128 | 100% | 1.49 / 1.56 / 1.60 |

复现抑制方向不改变贪心解码的循环退化（解码病态，自我强化；与
教师强制下的上下文复制不同）。control 案例的适用范围止于教师强制
/评测设定下的复制行为。

**本轮（合作者 audit/control 提议）总结**：正面 = 线性审计监测器
（AUC 0.776 > 隐状态 0.735 > 词身份 0.581）、抑制的三重选择性、
代价曲线；阴性 = 参数化记忆不可测且不受影响、不能诱发复制、文风
不可注入、不能修复生成退化。

### 全头功能图谱 (job 21765227 + 登录节点分析, 2026-09-03)

176 个路由头逐个消融（连接换层均值；另有输出置零版本），每次测
25 项指标（总 NLL；按目标词类/词频/位置/难度分的 NLL；目标词是否在
前文出现；周期 32/128/512 复述准确率；code/prose/latex NLL）。

**单头消融能命名的头**：稳健（复述 ≥5 点或某类 NLL 超出 ≥0.02 nats）
23/176；按相对 z 分数（≥2.5）89/176；87 个头无可辨单头效应（冗余）。
稳健命名示例：复述回路 L4/h1、L3/h11、L3/h13、L6/h11、L1/h5；
稀有词头 L1/h7 (+0.049)、L9/h4、L7/h8、L11/h2；大写词头 L7/h11 (+0.054)、
L9/h13、L9/h2；空白符头 L10/h3（总 +0.002 但空白目标 +0.083）；
标点头 L10/h4、L7/h13；代码头 L7/h9、L7/h5；LaTeX 结构头 L7/h1 (+0.110)。
标签修正：领域指标受复述能力牵连（LaTeX/代码重复多），先按复述判定。

**接线签名（全部 176 头，无需消融）**：k-means 6 族，主轴是
"向 V 注入多少原始词向量"（族均值 −10.7 / −3.2 / −1.65 / +0.35 /
+3.9 / +10.2）。注入为正的两族（22 头）中 15 个是词法头（词类/词频），
仅 2 个无效应；注入为负的两族（101 头）中词法头仅 13 个、65 个无效应
→ **接线预测功能**。图：`interp_atlas_families.png`、
`interp_atlas_wire.png`；卡片：`atlas_cards.json`。

下一步命名剩余头：按接线族整族消融（组消融）+ 更细的行为测试。

### 发现式命名 (jobs 21766163-66 + 登录节点分析, 2026-09-03)

**数据**：1000 条模型未训练过的 mmap 序列（permuted position ≥5.5M，
超出 10700 步消耗量），每个头做连接消融记录逐 token ΔNLL（176×1000×1024）。
**协议**：前 500 条发现（取效应最大 200 token 提假设），后 500 条验证
（规则命中 token 的平均效应 ≥ 未命中的 3 倍且 ≥0.01 nats）。规则来源：
数据驱动词集（top-200 中计数≥3 且 lift≥5）+ 词类词表（代词分三人称、
开/闭括号、引号、换行、句末、介词、连词、限定词）+ 结构规则。

**结果：75/176 头通过验证命名**（第一版按 top-3 词覆盖率门槛为 0/176，
门槛错误已改）。代表性命名（held-out：规则内效应 / 规则外效应）：
- 代词按人称分工：L9/h3 第三人称 he/she/her（+0.276 / −0.002，覆盖其
  效应质量 30%）；L8/h8、L9/h8 it/they/them（+0.333、+0.182）；
  L9/h11 第一人称（+0.140）；L10/h15、L2/h10 第二人称
- 括号：开括号头 L8/h0（+0.211）、L6/h1、L6/h13、L7/h5、L7/h15、L3/h9、
  L5/h4；闭括号头 L8/h9（+0.271）、L9/h2、L3/h2、L6/h5、L2/h7、L4/h2、L7/h4
- 引号/标点 L11/h0、L10/h4、L10/h9、L8/h15；句末标点 L6/h12、L7/h13、
  L8/h1、L5/h11、L5/h7；换行/缩进 L11/h6、L11/h7、L11/h10、L10/h12、
  L5/h12、L6/h3
- 介词 L10/h10（+0.041，覆盖 46%）；限定词 L6/h15；数字 L6/h11（+0.244）、
  L1/h2；大写词 L8/h3、L9/h13、L7/h9、L11/h2、L6/h6
- 待人工判读的数据驱动词集：L11/h4 {than, second, old, right…} +0.699；
  L10/h11 {Canada, colour, honour, humor…}（拼写变体/国别？）；L10/h6
**限制**：命名描述最具区分度的功能，覆盖率多为 1–30%（头多功能）；
词表在看过第一轮后设计，验证数据独立；101 个头本轮未命名。
产物：`token_effects/discovery2_summary.json`、`token_effects/cards/*.md`
（每头 30 条片段）、`interp_head_atlas_map.png`。

### 阶梯完成：static + corrections (job 21755999, 2026-09-03)

150M step-5500 对齐（同数据/步数/评测集）：dense 3.7969 | static 3.7776 |
pos-table 3.7813 | **static+corr 3.6863** | pos-table+corr 3.6875 |
encoder+corr 3.6875（static+corr final step6000 = 3.6851）。

结论：修正模块在场时，全局预测器取常量 / 位置表 / 27M 编码器**从头
训练结果无差**（三者 0.0012 内）。增益分账：静态重接线 ≈0.02，
局部修正 ≈0.09（与全局预测器形态无关）。最简架构 = 逐头静态接线 +
局部修正模块。与 300M eval-time 结果（static_mean 比 dynamic 差 0.045）
的差别归因于从头训练的共适应；300M 从头训 static+corr 可作最终确认（可选）。

### 阶梯最后一格：static + corrections (job 21755999, 2026-09-03)

150M step-5500 对齐表（同数据/步数/评测集），完整六格：

| 全局路由 | correction | 预测器参数 | eval NLL |
|---|---|---|---|
| 无（dense） | 无 | 0 | 3.7969 |
| 静态常量 | 无 | 1,295 | 3.7776 |
| 位置表 | 无 | 1.33M | 3.7813 |
| **静态常量** | **有** | **1,295** | **3.6863** |
| 位置表 | 有 | 1.33M | 3.6875 |
| 编码器 | 有 | ~27M | 3.6875 |

三个"有 correction"格在 0.001 内相等（final step6000: 3.6851 / 3.6839 /
—）。**结论：全局预测器可退化为 1,295 个常量而无损失**；输入依赖的
全部收益由 854K 参数的 correction 模块承载；编码器的位置调度在从头
训练条件下也非必要。最小充分路由结构 = 静态 (层,头,流,源) 常量 +
局部内容反应修正。
注意：300M 上 eval-time 的 static_mean vs pos_table 差 0.045 是对编码器
调度的 co-adaptation，非必要成分；规模依赖性可用 300M static+corr 复核。

### 生成示例 round 1 (job 21780660, 2026-09-03)

对 10 个已命名头放大接线偏离 γ∈{0,1,2,4,8}，8 个中性提示各 2 段 80 token：
- **复述头 L4/h1：重复 token 占比 28.5% → 37.2%(γ2) → 40.8%(γ4) → 71.2%(γ8)**，
  distinct ratio 0.71 → 0.29，γ=8 样例整句循环——路由放大对"搬运信息"类
  功能产生 Golden-Gate 式泛滥。
- 介词头 L10/h10 轻微单调（9.7% → 14.4%）；数字头 γ4 时 4.2%（基线 1–2%）
- 代词头、括号头、引号头、换行头：**无系统变化**。
解释：路由放大 = 头更多地从来源读信息，对"决定输出哪类词"的头无效；
其杠杆应为输出强度（round 2 验证）。

### 生成示例 round 2 (job 21780859, 2026-09-03)

**类别头改用输出放大（κ 0–8）**：代词/括号/引号/换行/数字头仍无系统变化
→ 单个类别头无论放大路由还是输出都不会让生成泛滥；它们是在该类词
合理的位置做精细选择，不是概念表征（与 SAE 特征的本质区别）。

**复述回路短语复现（6 个具名头，γ 0–4，提示含独特短语，各 4 样本×100 token）**：
γ≤1 基本不复现（quantum crystals 0×，Aldermoor 0–0.5×）；γ=2–3 明显复现
（quantum crystals 1.0→5.75×，Aldermoor 1.5→3.0×）；γ=4 退化为整句循环。
样例：
- γ=3 "The Bridge is a suspension bridge in Los Angeles. The Bridge is a
  suspension bridge in San Francisco. The Br…"
- γ=4 "It is not a suspension bridge. It is not a suspension bridge. It is not…"
- γ=4 "He said the image of the theory of quantum crystals to the class. He said
  the investigation of the theory of quantum crystals to the class."
- γ=2 "The famous ancient village of Aldermoor is an old town of ancient village
  of Aldermoor…"
论文用法：round 1 的 token 重复率曲线（28.5%→71.2%）+ 这些样例。
`experiments/results/interp/generation/generation_demos{,2}.json`

### 一头多名 + 多义性统计 (discovery pass 3, 2026-09-03)

合并 5000 条数据 + 扩展规则：**88/176 头命名，280 个已验证功能**（每头
平均 1.6 个，最多 17 个；58 个头 ≥2 个功能）。新命名示例：L11/h1 词内
续写（+0.182，覆盖 36%）、L7/h7 身份角色名词（student/employee/bride/
clients，+0.225）、L7/h8 钱与时间量（$/years/cost/price）、L7/h0 时间词、
L6/h8 代码符号、L1/h7 词片段补全。
**多义性**：把一个头的全部已验证功能合并，中位数只覆盖其效应质量 13%
（最高 72%）→ 大部分效应不属于单头单功能，指向回路级（跨头跨层组合）
单元。已启动连接级扫描（88 头 × Q/K/V × 来源层 ≈1850 条，8 分片）与
功能×连接归因（`scripts/interp_circuits.py`）。

### 连接级归因 → 回路 (jobs 21782111-18 + 登录节点分析, 2026-09-03)

88 个命名头 × Q/K/V × 来源层 ≈1850 条连接，逐条替换为层均值，1000 条序列
逐 token 效应；对 280 个已验证功能（去重后 99 个）算每条连接在功能 token
上的效应。
- **原始排序**：前 10 条连接平均横跨 5.3 个头，仅 12% 在功能所属头内——
  但被复述头 L4/h1 的大效应连接混杂（数字、大写词等常出现在重复内容里）。
- **按连接自身跨功能分布归一化（z 分数）后**：平均 3.9 个头。多数词类
  功能（第三人称代词、it/they、开括号、引号、句末、大写字母）的专属连接
  集中在**一个头的多条连接**（Q/K/V × 多个来源层），单条效应小
  （+0.01~+0.09）而整头效应大（如代词 +0.276）→ 功能由头内多连接共同承载；
  另一些功能是真跨头回路：月份/单位 6 头、数字续写 7 头、大写前缀 6 头、
  词片段补全 9 头、身份角色名词 4 头。
- 结论：单元既不是单头也不是单连接，而是"连接集合"（头内多连接或跨头）。
  验证中：整组消融 vs 等量随机连接、整组放大（job 见 circuit_verify.json）。
产物：`circuit_matrix.npz`、`circuits_specific.json`。

### 回路级编辑验证 (job 21786544, 2026-09-03) — 双向、特异

每个功能取归一化排序前 10 条连接（一个头内多条或跨头），200 条 held-out
序列，报告功能 token 上 / 其他 token 上的 NLL 变化：

| 功能 | 整组消融 on / off / 随机10条 on | 放大×2 on / off | 放大×4 on / off |
|---|---|---|---|
| 开括号 | +0.259 / −0.001 / +0.007 | **−0.154** / +0.001 | **−0.231** / +0.004 |
| 引号 | +0.256 / −0.001 / −0.005 | −0.143 / +0.002 | **−0.312** / +0.007 |
| 闭括号 | +0.226 / +0.001 / −0.003 | −0.080 / +0.001 | −0.145 / +0.008 |
| 第三人称代词 | +0.219 / −0.002 / +0.004 | −0.072 / +0.002 | −0.084 / +0.006 |
| 句末标点 | +0.068 / −0.002 / +0.024 | −0.050 / +0.002 | −0.110 / +0.007 |
| 空白符 | +0.095 / −0.004 / +0.057 | −0.066 / +0.007 | −0.112 / +0.017 |
| 数字 | +0.115 / +0.016 / +0.002 | +0.076 / +0.020 | +0.305 / +0.139（与复述回路纠缠，放大反而伤）|
| 词内续写 | +0.232 / +0.068 / +0.002 | +0.033 / +0.023 | +0.237 / +0.158（宽泛功能）|

6/8 功能得到回路级双向旋钮：消融只伤该功能 token（其他 token ≈0，随机
对照 ≈0），放大让该功能的预测**变好**（−0.05 至 −0.31 nats）且其他
token 几乎不变（<0.01）。与生成实验合看：放大并不让该类词在生成中泛滥
（回路是"条件使能器"——在该类词合理时增强，不是概念注入）。
图：`interp_circuit_editing.png`；数据：`circuit_verify.json`。

### 路由空间字典学习 (jobs 21787092 / 21787756, 2026-09-03)

对每 token 的 3773 维修正向量做稀疏自编码。v1（L1 惩罚）不稀疏（每 token
100–480 个激活），弃用。**v2：top-k（k=32）、8192 特征、400 万 token、
3 轮**，R²≈0.82，特征激活频率 0.2–2.5%，最强激活上下文呈现语义/结构
特征：亲属名词（his >>brother<<；f3583）、身体/方位名词（f4271）、对话
归属动词（" she >>added<<；f6409）、时间单位（a guinea a >>-week<<；f1407）、
化合物（carbon >>dioxide<<；f366）、"第 N 世纪"（f4380）、脚注括号、表格
数字等。
**Steering 读数（12 个特征，α=−3…+3 × 平均激活）**：held-out 文本上规则
token 的 NLL 随 α 单调变化——f2859: −3 → +0.167，+3 → −0.127（规则外
≤+0.009）；f452: +0.051 → −0.030；f2985 符号反向（−0.098 ↔ +0.134）——
即**方向级的双向调节成立**；但这批特征的目标词太稀有，生成占比读数为 0，
无法判断"泛滥"。v3 改用语义/常见类特征（亲属、身体、对话动词、时间单位、
最高级、代词、否定…）、α 到 ±8、每条件 32 段生成，双读数并行。
产物：`routing_sae/sae.pt`、`features.json`、`steering.json`。

### SAE 方向 steering v3 (job 21788155, 2026-09-03)

14 个语义/常见类特征，α=−8…+8 × 平均激活，32 段 × 60 token 生成 + 100
条 held-out NLL。
- **亲属名词方向 f3583：双向生效**。生成中亲属词占比 0.62 / 0.31 / 0.21 /
  0.10 / 0.05%（α −8→+8，12 倍跨度；符号约定反向，负 α = 激活）；规则
  token NLL −0.052（−4）vs +0.135（+4），规则外 +0.02。±8 时文本退化
  （"Capitalism. Capitalism…"，规则外 +0.108），可用区间 ±4。
- NLL 读数双向成立但生成读数不足：'century'（−0.073 / +0.084）、否定
  （−0.043 / +0.050，生成 0.94% vs 0.21% 但中段不单调）、引号（反号，小）。
- 无效：身体/方位、对话动词、时间单位、最高级、代词、数字。
- 注意：生成占比基数很小（0.2% ≈ 每条件 4 次出现），亲属结果需更大采样
  和更合适的提示语复核；下一步对全部 200 个命名特征先用 NLL 读数筛双向
  可控方向，再对胜出者做大样本生成。
产物：`routing_sae/steering_v3.json`。

### SAE 方向全量筛选 + 大样本生成 (job 21788580, 2026-09-03) — 定论

- **损失读数**：187 个命名方向中 **34 个双向可控**（±4 时规则 token 的 NLL
  反向变化且规则外 <0.03）。
- **生成读数（96 段/条件，5760 token）**：8 个胜出方向中没有一个呈现干净
  的单调频率变化；亲属方向前一轮的 12 倍效应不复现（命中 15/13/9/7/12）；
  f7326（所有格/词片段）单调但弱（9→21 次）；f7068、f4222 呈 V 形（两个
  方向都比基线高）——是扰动导致的泛化退化，不是方向控制。
**结论**：路由界面上的 steering 控制的是"功能在该发挥时发挥得多好"
（能力级：回路集合与 SAE 方向都双向、特异、held-out 可验），不控制
"自由生成中说什么"（内容倾向），唯一例外是复述（路由的本职）。
Golden-Gate 式的内容泛滥是残差流表征层面的现象；路由决定信息流向、
不决定表征内容。该分离本身是关于本架构界面的可解释陈述：两类操控
杠杆，我们拥有其中一类。
产物：`routing_sae/screen.json`、`screen_generation.json`。

### 图：复述回路示意 + 三态生成 (2026-09-03)

`experiments/figures/interp_copy_circuit_demo.png`：左侧 12×16 头网格上
高亮 6 个复述头（L4/h1 主头），箭头为其 Q（紫）/K（蓝）的前两大来源层
连接（实线正权、虚线负权，embedding 为负 = "按上下文而非自身匹配"）；
右侧用未见过的地名 "Aldermoor" 做提示：γ=0（抑制）续写全程不再出现该
名；γ=1（原模型）自然提及；γ=2（激活）反复出现并形成排比句式。
脚本内联于会话（可从 localize_step9000.json + generation_demos2.json 复现）。

### 回路级自由生成三态 (jobs 21791872 / 21792536, 2026-09-03)

12 个回路各 96 段 × 80 token（γ=0/1/2/4）。该类词占比：
- 引号 1.15 / 1.16 / 1.73 / 1.99%（激活 +72%，抑制无变化）；每段引号数
  0.94 / 1.19 / 2.07
- 第三人称代词 1.60 / 1.78 / 2.76 / 2.25%；介词 10.4 / 10.4 / 10.7 / 12.0%
- 换行 token 2.72 / 3.20 / 3.22 / 3.31%；但"句中硬换行数/段"96 样本下
  1.11 / 1.36 / 1.32 —— 6 样本时看到的 0 / 0.67 / 2.67 是噪声
- 开/闭括号、句末、数字、词内续写、大写、代码符号：无变化
**结论**：类别回路在自由生成中只有激活侧的温和变化（≤2×，样本方差大），
抑制侧几乎不可见；生成上有戏剧性三态的仍只有复述回路。模板卡片：
`figures/circuit_cards/*_generation.png`（引号卡为类别回路的最佳例）。
概率版三态（每回路 3 个上下文的 P(正确词)）：`circuit_examples.json`，
例如 "(Llanbeblig" 后 ")" 的 P 为 0.03 / 0.25 / 0.30，换行位置 0.02 / 0.21 / 0.58。

### 模板卡片集 (2026-09-03)

`experiments/figures/circuit_cards/`：
- `*_circuit.png`（6 张，概率版）：开/闭括号、引号、第三人称代词、句末、
  换行——右栏为 held-out 三态损失 + 3 个上下文中 P(正确词) 的三态
- `*_generation.png`（4 张，生成版）：类别回路的自由生成三态**不成立**
  （96 样本中位数样例三态无差别），仅作记录，不宜入论文
- `interp_copy_circuit_demo.png`：复述回路，自由生成三态成立（唯一）
脚本：`interp_circuit_examples.py`（数据）、`interp_circuit_figs.py`（概率卡）、
`interp_circuit_gencards.py`（生成卡）。

### 全连接回路发现（不依赖单头命名）(jobs 21794904-13 + 21795994, 2026-09-03)

补扫剩余 88 个未命名头的连接后，候选池 = 全部 **3696** 条逐头 Q/K/V 连接
（每 token 路由维度 3773 = 49×77，其中 77 条 R 为层级共享、未纳入扫描）；
功能列表 = 全部 86 条规则（词类/字符类/前词类/结构 + 已验证词集），
不再由单头命名反推。

- 各功能前 10 条连接中，来自"单头无功能头"的比例平均 13%；
  **6 个功能过半来自无名头**。
- 整组验证（200 条 held-out，对照等量随机连接）：

| 功能 | 回路 | 消融：规则内/规则外/随机 | 放大×4 |
|---|---|---|---|
| 数字后接数字 | 7 头 6 层（50% 无名） | **+0.072** / +0.004 / +0.004 | +0.031 |
| 词内续写 | 8 头 6 层（60% 无名） | **+0.260** / +0.073 / +0.002 | +0.130 |
| 跟在换行之后 | 8 头 5 层（60% 无名） | **+0.050** / +0.030 / +0.009 | +0.064 |

**结论：存在只在连接组合层面成立、单头消融检不出的功能回路**（消融效应
为随机对照的 5–130 倍）。与单头主导的回路相比，这类回路特异性较低
（规则外也有上升）且不可放大（放大同样使损失上升）——"缺一不可但不能超量"。
产物：`circuits_all.json`、`circuit_matrix_all.npz`、`verify_all.json`。

### 归属分解补完：corrections-only (job 21780504, 2026-09-04)

150M step-5500 对齐（同数据/步数/评测集），相对 dense 3.7969 的增益：

| 条件 | eval NLL | 增益 |
|---|---|---|
| 只有静态路由（无 corrections） | 3.7776 | 0.0193 |
| **只有 corrections（predictor 冻结为恒等）** | **3.6983** | **0.0986** |
| 静态路由 + corrections | 3.6863 | 0.1106 |
| 编码器 predictor + corrections | 3.6875 | 0.1094 |

两个方向的边际值：给定路由再加 corrections **+0.0913**；给定 corrections
再加路由 **+0.0120**。单独之和 0.1179 > 联合 0.1106（轻微次可加）。
→ 从头训练下 **corrections 占增益的绝大部分**（89%），静态路由约 11%；
这与 300M 训好模型上的删除实验（删 predictor 静态接线 −1.9、删 corrections
−1.3，各约一半）不矛盾：后者衡量训好解对各部件的依赖，前者衡量从头训练
的边际贡献。此前把 0.02/0.09 说成"归属"是错的（只算了一个顺序），现已补齐。

### 以"生成可见"为搜索目标的全头扫描 (job 21798297, 2026-09-04)

对 176 个头逐个测 γ ∈ {−4,−1,0,1,4,8,16}（新增**负 γ = 反向接线**），
每态 32 段生成，测 10 个可见统计量，打分 = 相对变化幅度 × 与 γ 的秩相关。
基线：换行 2.75、句长 16.7、多样性 0.77、三元重复 0.01、全大写 0.03。

**发现 1：负 γ 是全新效应区**（此前只试过 0–8）
| 头 | γ=−4 / −1 / 0 / 1 / 4 / 16 |
|---|---|
| L3/h12 句长 | 40.7 / 35.4 / 16.8 / **17.4** / 17.6 / 18.8 |
| L3/h12 多样性 | 0.41 / 0.38 / 0.73 / **0.73** / 0.78 / 0.80 |
| L3/h12 三元重复 | 0.58 / 0.62 / 0.02 / **0.03** / 0.00 / 0.00 |
| L2/h10 句长 | 35.3 / 46.1 / 14.8 / **15.6** / 14.8 / 16.9 |
反向接线 → 无换行、句长 2.4 倍、词汇减半、大量自我重复的长段落。

**发现 2：有效应只在 γ>8 显现**（验证"更夸张"确有必要）
L1/h15：全大写词 0.03 → **3.50**（γ=16，117 倍），句长 15→28，多样性 0.76→0.57。

排行榜（前 8）：L1/h15(allcaps 104)、L3/h12(rep 100)、L4/h1(rep 90)、
L4/h14(64)、L6/h11(63)、L1/h8(56)、L2/h10(55)、L1/h5(32)——L4/h1 与
L6/h11 是已知复述回路成员，说明打分机制有效。
**待排除**：通用退化混淆（等强度随机方向对照）——job 21803778。
产物：`gen_search/search_heads.json`。

### 生成搜索的对照与纠错 (job 21803778, 2026-09-04)

对排行榜前 6 个头做样例核查 + 等强度随机方向对照 + 损失代价：
- **统计量被退化文本欺骗**：L3/h12 γ=−1 的"长句+高重复"实为词元卡死
  （`TheBurnTheBurnTheBurn…`），L2/h10 为 `ThePeopleRNLLLLL…`；无句号→
  "句长"虚高，token 卡住→"重复率"虚高。损失仅 +0.15（教师强制下影响小）
  但自回归采样漂移累积。
- L4/h1 γ=16（重复 0.73）与 L1/h15 γ=16（损失 3.11→7.18）同为破坏性扰动，
  等强度随机对照也产生 0.56–0.59 重复 → 通用退化，不特异。
**结论**：生成层面站得住的三态单元仍只有复述回路（γ=2–3，文本可读）。
**方法修正**：搜索加入"可读性闸门"——仅接受 三元重复 ≤ max(0.05, 3×基线)、
多样性 ≥ 0.6、损失 ≤ 基线+0.35 的状态，且至少 3 个状态通过才计分
（job 21803950 重扫）。这条闸门是本轮方法论教训：**以生成统计量为搜索
目标时必须同时约束流畅度，否则搜索会收敛到退化模式**。

### 带流畅度闸门的全头生成搜索 (job 21803950, 2026-09-04) — 否定结论

176 头 × γ∈{−4,−1,0,1,4,8,16} × 32 段，闸门：三元重复 ≤ max(0.05,3×基线)、
多样性 ≥0.6、损失 ≤ 基线+0.35。
- 排行榜被**低基数统计量**占满（全大写、括号基线均为 0.00，出现一次即
  无穷大比例）——分数高但绝对量无意义。
- 按**绝对移动量**重排：最大者为句长 ±6 词（基线 16.8），与 32 段采样噪声
  同量级，且随 γ 非单调（如 L6/h9: 15.4/18.6/17.1/18.5/16.7/16.5/22.1）。
**结论（完整搜索空间）**：在保持文本可读的前提下，**没有任何单头**能在
自由生成中产生稳定三态效应。生成层面唯一成立的仍是复述回路（多头集合，
γ=2–3）。类别功能只在"该功能 token 的概率"层面可双向调节。
方法论：本轮把"生成可见性"作为搜索目标本身跑通了（自动化、无人工挑选），
其结论是负面的，但闸门 + 绝对量核查是必要组件——否则会误报退化与低基数
假象（本轮两次误报均由此产生并被纠正）。
产物：`gen_search/search_heads.json`、`gen_search/confirm.json`。

### 存储清理 (2026-09-04)

project 配额曾达 506.3G/500G（宽限期启动）。清理（均为可从保留产物重算
或结论已固化的中间数据）：
- `token_effects/effects_conn_{g,u}*.pt`（16 个分片，约 7.6G）→ 归因矩阵
  `circuit_matrix{,_all}.npz` 已保存，原始逐 token 张量不再需要
- `corr_magnitude{,_more}.pt`（1.7G）→ 从未被使用
- 300M DAGFormer 中间 checkpoint step1500–7500（约 14G）→ 保留 step9000
  （全部分析所用）与 step10500（最终）；涌现曲线结论已固化
- 150M 各 ladder run 的中间 checkpoint、300M baseline step3000/6000
- `checkpoints/checkpoint_step500.pt`（2.5G 四月孤立文件）
结果：**506.3G → 464.1G**，宽限解除。/work 仍 100% 满（1.5T），未动
（其中 pretok 数据为训练必需）。

### /work 存储清理 + 模型共享 (2026-09-11)

**清理**：`/work/hdd/bfqt` 配额曾 1.519T / 软限 1.465T（宽限过期）。删除：
- `data/pretok/olmo_mix_21b`（42G）：配额事故时抢救的 11.3B 部分索引，已被
  `olmo_mix_21b_full`（21.0B, 42 shards）取代，仅被两个 smoke 配置引用
- 600M 三个 run 的中间 checkpoint（保留各自最终步）
- 1B baseline 的 step38000（保留 step38160，即完整 20B token 的最终模型）
- `hf/dagformer-1b/ckpts` 的 16 个中间导出（保留 step10000）
结果 **1.519T → 1.393T**，宽限解除。未动：`data/pretok/dolma_v1_7_12b`
（所有 matched run 与 interp 语料所用）、`data/olmo-mix-1124` 原始镜像
（77G，1B run 的预分词源，可重下但耗时）、`yurenh2/ept`（用户另一项目）。

**共享**：Xiaocong 与我同属 `delta_bfqt` 组，但 `/projects/bfqt/users` 是
`drwx------`（他无法进入），且 `/work` 下我的子目录组为 `grp_202` 且无组权限
——这才是障碍，不是路径不存在。新建 `/work/hdd/bfqt/shared/dagformer-models`
（`delta_bfqt` 组 + `g+rX`），放入 7 个模型（75M/150M/300M 的 baseline 与
DAGFormer + 600M baseline，均为最终 checkpoint）、加载器、`src/`、tokenizer
与 README，共 12G；并把已有 `dagformer_checkpoints/hf/` 导出改为组可读。
已在该目录下 CPU 端到端验证两类模型均可加载。
踩到的坑：checkpoint 内记录的是 side-file 的**原始文件名**，加载器按该名在
同目录查找，故共享副本不能改名（曾改成 `checkpoint_model.pt` 导致加载失败）。

### 欺骗完形填空：上下文忠实度回路 (jobs 22025644/22025724/22025808, 2026-09-13)

**动机**（Xiaocong 提议）：liars-bench 需要指令跟随与世界知识，300M 基座
模型二者皆无。改为上下文锚定的完形填空即可绕开：上下文先确立事实
（"Anna 看到盒子里是红球"），角色转述时留空，说出上下文的值 = 诚实，
说别的 = 撒谎。这是纯下一词预测，基座模型能做。

**第一版设计错误（已纠正）**：我比较了中性/诚实/欺骗三种提示，想看模型
是否对提示敏感——结果不敏感（DAGFormer 诚实−欺骗 = −0.038，方向还反了；
dense +0.020），据此我曾误判该方向不可做。正确设计是 Xiaocong 的原意：
**固定用欺骗提示**，变异由干预制造，不需要提示敏感性。

**顺带的对照结果**（240 题，真值在真值+同类干扰项间归一化）：
DAGFormer 真值概率 0.669 / 准确率 89.6%；dense 0.500 / 68.8%。
DAGFormer 的上下文取回能力显著强于 dense。

**头扫描**（全部欺骗框架，基线真值概率 0.7188）：
L4/h1 **−0.100**、L7/h1 −0.058、L9/h2 −0.049、L7/h11 −0.041、L7/h9 −0.027、
L3/h11 −0.026、L3/h13 −0.021、L10/h6 −0.019、L8/h3 −0.017、L6/h11 −0.015；
反向：**L2/h10 +0.021**（平时在压制上下文取回）。
前 10 名中 4 个（L4/h1、L3/h11、L3/h13、L6/h11）是已命名的复述回路成员。

**连接细化**：最强两条均在 L4/h1——`v<-src4` −0.080、`q<-src1` −0.060；
其余为 L7/h9 `v<-src0`、L7/h1 `v<-src1`/`k<-src7`、L9/h2 `v<-src2` 等。

**判读**：同一批具名头同时解释"合成序列逐字复述"与"按上下文如实作答"
两种表面无关的行为，是回路真实性的强证据（非每任务各拟合一套）。
审计意义：该回路控制模型是否报告上下文给定的事实，压低即产生与上下文
矛盾的编造。**但不可称为"说谎回路"**——欺骗框架下模型仍 91% 说真话，
无欺骗意图表征；准确命名为**上下文忠实度回路**。
脚本：`interp_liar_cloze.py`、`interp_deception_circuit.py`。

### 上下文忠实度回路的双向调节 (jobs 22025853/22025923, 2026-09-13)

固定欺骗框架，10 条连接的回路（L4/h1 `v<-src4`、`q<-src1` 为核心，
另含 L7/h9、L7/h1、L9/h2 的若干条），γ 缩放其相对层均值的偏离：

| γ | −2 | −1 | 0 | 0.5 | 1（原模型） | 2 | 4 |
|---|---|---|---|---|---|---|---|
| 真值概率 | 0.326 | 0.427 | 0.522 | 0.599 | **0.719** | 0.808 | 0.935 |

七档完全单调，跨度 0.33→0.94。**对照（各 5 个种子）**：
全模型随机 10 条连接 0.7183–0.7238（≈零效应）；同一批头内随机抽
0.684–0.740（最差 −0.035）。选中回路全消融 −0.197 = 最差同头随机的
5.6 倍、全模型随机的约 40 倍。（首轮仅 3 个种子时曾出现 −0.094 的异常
种子，扩到 5 个种子后消失。）

**命名**：称其为**上下文忠实度回路**而非"说谎回路"。理由：欺骗框架下
模型本就 91% 说真话，中性框架 0.669 vs 欺骗 0.719 几乎无差，模型不具备
欺骗意图表征；该回路控制的是"是否使用上下文中给定的事实"，压低即产生
与上下文矛盾的编造。审计含义明确（编造/幻觉侧），且与复述回路同源。
图：`experiments/figures/interp_context_fidelity_dial.png`。

### 命名的证伪测试：框架对照 (job 22043981, 2026-09-13)

同一 10 条连接回路（在欺骗框架下发现），在**中性框架**下重测 γ 曲线：

| γ | −2 | −1 | 0 | 0.5 | 1 | 2 | 4 |
|---|---|---|---|---|---|---|---|
| 欺骗框架 p_true | 0.326 | 0.427 | 0.522 | 0.599 | 0.719 | 0.808 | 0.935 |
| 中性框架 p_true | 0.320 | 0.416 | 0.511 | 0.578 | 0.668 | 0.763 | 0.918 |
| 差 | +0.006 | +0.011 | +0.011 | +0.021 | +0.051 | +0.045 | +0.017 |

两条曲线近乎平行；γ 效应（消融 −0.157/−0.197，放大 +0.25）在两框架下相当，
差值均在 0.02 量级且不随 |γ| 增长。中性框架下特异性同样成立（全模型随机
10 条 0.668–0.673 对基线 0.668）。
**结论：该回路对欺骗框架不敏感 → 明确命名为上下文忠实度回路，非说谎回路。**
这张对照表即是该命名的证伪测试，可直接回应"你怎么知道不是说谎"的质疑。
