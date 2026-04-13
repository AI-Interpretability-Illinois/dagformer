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
