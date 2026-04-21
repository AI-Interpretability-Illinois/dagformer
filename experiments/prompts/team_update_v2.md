# DAGFormer: Project Update (v2) — from hard-gate A to FourWay

## tl;dr

Last update you got was the old **hard-gate 256×256 adjacency matrix** version (Qwen encoder → Gumbel-Sigmoid → binary A). We've since rewritten the routing to a **per-token, per-head, 4-stream soft mixing** scheme inspired by MUDDFormer (Pagliardini et al., 2025) but operating at **head level** instead of layer level. First 300M results show the method beats a dense OLMo-2 baseline by **~0.22 nats at matched token count**. 1B training is underway on rented A100s, checkpoints will land in ~2 days. Tasks follow — the high-priority ones are "run these commands verbatim"; model-dependent interp / visualization tasks are parked until checkpoints exist.

---

## 1. Architecture evolution

### Old (what you last saw)

- Qwen encoder (frozen) → MLP → **single 256×256 binary A matrix per window**
- Gumbel-Sigmoid with temperature annealing + cascading gate
- All tokens in a window share the same A

### New (FourWay)

A standard OLMo-2 transformer forward, but each attention layer's Q/K/V inputs are a **learned soft mixture** of all prior layer outputs rather than just the previous residual:

```
For layer l with l+1 sources {X_0 = embedding, X_1, ..., X_{l-1}}:
  Q_mixed[t, h] = Σⱼ α_q[t, h, j] · q_proj(X_j)     per head, per token
  K_mixed[t, h] = Σⱼ α_k[t, h, j] · k_proj(X_j)     per head, per token
  V_mixed[t, h] = Σⱼ α_v[t, h, j] · v_proj(X_j)     per head, per token
  R[t]          = Σⱼ α_r[t, j] · X_j                 shared across heads, used as residual
```

- **Routing weights α** come from a small independent **causal transformer encoder** reading `input_ids`, fully parallel, per-token output
- **Identity init**: predictor weights W=0, bias=[0,…,0,1] — at step 0 α puts all weight on the most recent source, exactly reproducing vanilla OLMo-2 (verified numerically, diff=0)
- **No Gumbel, no hard gating, no binary thresholding** — the entire routing is continuous real-valued

### How it differs from the competitive set

- **MUDDFormer (ICML'25)** routes at **layer** level (one α per layer) → we route at **head** level (finer granularity)
- **MoH (ICML'25)** gates head **outputs** (which heads contribute to the final sum) → we gate head **inputs** (what each head sees)
- **Hyper-Connections (ICLR'25)** maintains n parallel residual streams → we have explicit Q/K/V/R stream separation
- Nothing in the literature uses a **separate, dedicated predictor MLP** for per-token head-level routing — that's our claimed niche

---

## 2. Where we are right now

| Data point | Value |
|---|---|
| 300M DAGFormer @ 1K steps (0.52B tokens) | train NLL **4.02**, eval NLL 4.55 |
| 300M dense baseline @ 1K steps (same tokens) | train NLL 3.89–4.12 |
| Mean gap steps 100–750 | **+0.22 nats (DAGFormer ahead)** |
| DAGFormer wins at 71/100 sampled checkpoints |  |
| 1B DAGFormer (rented A100, in progress) | step 3.7K, train NLL ~2.82, resuming from a crash |
| 1B baseline | to run after DAGFormer finishes |

Four critical bugs were found and fixed recently (all post-fix):
1. DDP gradient bypass (params diverged across GPUs)
2. Wrong attention kernel (manual f32 softmax → SDPA with is_causal)
3. Bias filter in optimizer (grabbed wrong param group)
4. LR scheduler missing a param group

A "combo" regularization (V post-mix RMSNorm + softmax-normalized R/V routing weights) is the 75M ablation winner. Sticking with V-norm only for 1B to test if softmax helps at scale.

---

## 3. Project timeline

| Milestone | When | Notes |
|---|---|---|
| 1B DAGFormer complete | ~2 days | checkpoint available for interp |
| 1B dense baseline | ~2–4 days after that | A100 rental still alive |
| 75M / 150M / 300M baselines + reg tests on SLURM | queued, ~1–2 days once they start | ACCESS (NCSA Delta) |
| 75M / 150M baselines on TPU | awaiting TPU TRC capacity | 30-day free trial window |
| 3B training | planned on 64× TPU v6e | needs JAX FourWay port first |

---

## 4. Tasks — pick what looks interesting

Tasks fall into **three categories**. Xiaocong's recommendation (which I agree with) is to start with **Category A** since that's operational work that unblocks other things, doesn't need you to have deep architecture context yet, and gets you comfortable with ACCESS.

Once 1B checkpoints drop in ~2 days, **Category B** becomes the high-leverage work.

### Category A: RUN PRE-WRITTEN CODE (start here — no checkpoints needed)

All commands below are literal — copy-paste, don't paraphrase. Goals: get you running on ACCESS and produce data we need anyway.

**A1. Pick up the queued reg comparison jobs** (plain vs vnorm vs softmax_rv at 300M, 5K steps each).

The jobs are already in SLURM queue (jobs 17717990, 17717991) but scheduled for tomorrow night due to cluster pressure. If you have priority on your ACCESS account, you can submit the same configs yourself and they might start sooner:

```bash
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
git pull

# 300M vnorm (~18h on 4× A40)
sbatch --time=24:00:00 -p gpuA40x4 scripts/slurm_fourway.sh configs/fourway_vnorm_full_fix.yaml

# 300M softmax_rv (~18h)
sbatch --time=24:00:00 -p gpuA40x4 scripts/slurm_fourway.sh configs/fourway_softmax_rv_full_fix.yaml
```

**A2. Pick up the queued baseline jobs** (75M, 150M, 300M dense):

```bash
# 75M baseline (~2h on 4× A40)
sbatch --time=06:00:00 -p gpuA40x4 scripts/slurm_pretrain_300m.sh configs/pretrain_75m_baseline.yaml

# 150M baseline (~5h)
sbatch --time=12:00:00 -p gpuA40x4 scripts/slurm_pretrain_300m.sh configs/pretrain_150m_baseline.yaml

# 300M dense at Chinchilla-optimal tokens (~15h)
sbatch --time=36:00:00 -p gpuA40x4 scripts/slurm_pretrain_300m.sh configs/pretrain_300m_chinchilla.yaml
```

If our jobs start first (you can check `squeue -u yurenh2`), cancel yours so we don't double-run.

**A3. Once any job finishes, pull `metrics.csv` and plot**:

```bash
# In the repo root after job completes:
python3 scripts/plot_300m_quick_comparison.py  # reference plot script
# Or write your own — see experiments/figures/300m_quick_dense_vs_dagformer.png
```

### Category B: VISUALIZATION & INTERP (start ~2 days from now, when 1B checkpoint exists)

Xiaocong's explicit suggestion: **"look at whether the encoder learns meaningful structure"**. These are the highest-value analysis tasks.

**B1. Domain routing heatmap** — does the predictor route differently for different content?
- Sample N documents each from: math, code, wiki, fiction, legal
- Forward through the trained 1B FourWayPredictor, collect α_q / α_k / α_v / α_r
- Produce per-layer mean routing heatmaps per domain + pairwise difference heatmaps
- **Deliverable**: a figure that answers "is routing context-dependent or collapsed to near-identity?"

**B2. Encoder representation analysis** — what does the causal encoder actually encode?
- Extract the predictor's last hidden state (before layer heads)
- UMAP / t-SNE colored by domain, POS, position, token identity
- If clusters emerge, the encoder has learned meaningful structure → routing is data-driven
- If no structure, routing is noise → collapse

**B3. Routing weight statistics** — is routing actually using the extra sources?
- Per layer, per stream: entropy of α distribution, mean, variance
- Fraction of mass on the "identity" source (last one) vs. all others combined
- If mass is >99% on identity → model is using FourWay as vanilla residual, no benefit from architecture
- **Deliverable**: a table + bar chart showing "how much does routing actually deviate from identity per layer?"

**B4. Per-stream importance** — do we need all four Q/K/V/R streams?
- Ablate: freeze α_q at identity, train/eval
- Repeat for α_k, α_v, α_r
- Measures which stream carries the improvement
- **Can start without new training** — just zero out the deviation and run eval on existing 1B checkpoint

### Category C: CODE WORK (lower priority, flexible schedule)

**C1. MUDDFormer baseline at 300M** — reviewers will ask. Official pytorch code is cached at `docs/reference_papers/muddformer/modeling_muddformer.py`. Port the `MultiwayDynamicDenseBlock` into our training loop. Same config as our 300M DAGFormer for a head-to-head. Produces the main comparison-table entry.

**C2. MoH baseline at 300M** — implement the parameter-free router (L2 norm of Q) + top-K head mask. Cleaner "head-level routing without a learned predictor" reference.

**C3. lm-eval-harness integration** — add a hook that runs HellaSwag / PIQA / ARC-E / LAMBADA / BoolQ on any saved checkpoint. Lets us report downstream numbers alongside perplexity.

**C4. JAX/Flax port of FourWayDAGFormer** — we have the JAX baseline working (`scripts/pretrain_baseline_jax.py`), but the DAGFormer forward needs a Flax port for TPU training. This is 1–2 weeks full-time but unblocks the 3B run on TPU TRC (30-day allocation, clock is ticking).

**C5. Data preprocessing pipeline** — pre-tokenize and pack Dolma locally so training doesn't depend on HF streaming reliability (we've had HTTP crashes kill multi-day runs).

---

## 5. Where to read

1. `CLAUDE.md` §1–4 — architecture spec (the rest is historical context)
2. `src/model/olmo_graph.py::FourWayDAGFormer` — the modified forward pass
3. `src/model/predictor.py::FourWayPredictor` — the routing predictor
4. `scripts/pretrain_dagformer.py::main` — training loop
5. `experiments/results.md` — chronological experiment log
6. `experiments/figures/300m_quick_dense_vs_dagformer.png` — the key result so far

## 6. What to reply with

Pick whatever looks interesting from A/B/C and reply. I'll send a concrete first-issue-scope for whatever you pick (exact command, expected output, success criterion, deadline).

If you're not sure — Xiaocong and I recommend starting with **A1 or A2** (just run jobs, minimal ramp-up) + one from **B1/B3** to start sketching once checkpoints arrive.
