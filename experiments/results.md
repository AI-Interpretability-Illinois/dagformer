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
