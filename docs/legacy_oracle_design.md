# DAGFormer — original oracle / binary-gate design

Historical design document. The current FourWay architecture is described in
the [project README](../readme.md).

> **One-liner**: Train a lightweight neural network to predict per-token dynamic
> DAG topologies for OLMo2-1B, replacing expensive oracle search with a
> single forward pass.

## Project Context

The oracle search (separate codebase) proved that **context-dependent
topologies exist** which dramatically reduce NLL (2.58 → 0.12 median,
100% improvement rate across 50 windows). However, oracle search takes
~500 optimization steps per window and cannot scale. This codebase trains
a **structure predictor** end-to-end: it reads the current context,
predicts a soft adjacency matrix, and the language model executes under
that topology — all differentiable, no oracle data needed.

---

## Architecture Overview

```
Context tokens
     │
     ├──────────────────────────┐
     ▼                          ▼
 Qwen-3-Embedding-0.6B     OLMo2-1B
 (frozen encoder)           (16L × 16H = 256 nodes)
     │                          │
     ▼                          │
 Structure Predictor            │
 (trainable MLP)                │
     │                          │
     ▼                          │
 Gumbel-Sigmoid                 │
 + Cascading Gate               │
     │                          │
     ▼                          │
 Soft A ∈ [0,1]^{256×256}  ───▶ │ (applied as gate mask)
   upper-triangular             │
   per-token                    ▼
                            NLL Loss
                                │
                    ◀── ∇ backprop to predictor
```

### Gate Structure (inherited from oracle search)

| Gate type        | Count   | Semantics                              |
|------------------|---------|----------------------------------------|
| Sequential       | 256     | 16 layers × 16 heads (residual → head) |
| Hyperconnection  | 30,720  | head_i → head_j for all j > layer(i)   |
| **Total**        | **~31K**| Binary decisions per context window     |

The adjacency matrix A is 256×256, upper-triangular (DAG constraint).
Row i, col j means "head i sends output to head j". Sequential gates
are the diagonal-block entries; hyperconnection gates are off-diagonal.

### Oracle Topology Statistics (reference targets)

- Sequential gates: ~91% ON
- Hyperconnection gates: ~70% ON  
- Jaccard similarity between windows: < 0.8 (topologies are context-dependent)

---

## Structure Predictor Architecture

```
Input:  context embedding e ∈ R^d  (from Qwen, pooled or [CLS])
                │
                ▼
        MLP: Linear(d, h) → GELU → Linear(h, h) → GELU
                │
                ▼
        Low-rank head: Linear(h, 256·r), Linear(h, 256·r)
                │               │
                ▼               ▼
            U ∈ R^{256×r}   V ∈ R^{256×r}
                │
                ▼
            Z = U V^T  ∈ R^{256×256}   (logits)
                │
                ▼
        UpperTriMask ⊙ σ((Z + G) / τ)  (Gumbel-Sigmoid)
                │
                ▼
        Cascading Gate:
            g_j = σ(k · Σ_i A[i][j])
            A[j, :] *= g_j
                │
                ▼
        Soft A ∈ [0,1]^{256×256}
```

**Key design choices:**

1. **Low-rank factorization** — Instead of predicting 65K entries, predict
   U,V ∈ R^{256×r} where r ∈ {8, 16, 32, 64}. Inductive bias toward
   structured topology. Also enables future diffusion head replacement.

2. **Gumbel-Sigmoid** — Continuous relaxation of binary gates.
   Temperature τ anneals from τ_init to τ_final via cosine schedule.
   - τ large → soft (exploration)
   - τ small → sharp (near-binary)
   - Training: soft; Inference: hard threshold at 0.5

3. **Cascading gate** — If a node has no incoming edges, it should have
   no outgoing edges (no information to propagate). Enforced as a
   differentiable soft constraint:
   ```
   incoming_j = Σ_i A[i][j]
   g_j = σ(k · incoming_j)        # k=5, fixed or learnable
   A[j, :] = A[j, :] * g_j        # kill outgoing if no incoming
   ```

---

## OLMo2-1B Modification

The base OLMo2-1B model needs a **modified forward pass** that:

1. Accepts a soft adjacency matrix A ∈ [0,1]^{256×256} per token
2. Gates the residual stream connections accordingly:
   - Each attention head output is multiplied by its gate value before
     being added to the residual stream
   - Hyperconnections: head_i's output is routed to head_j's input,
     weighted by A[i][j]
3. When A is all-ones (or not provided), behavior is identical to vanilla
   OLMo2-1B (this is the **sanity check** — must reproduce baseline NLL)

**Implementation strategy**: Monkey-patch or subclass OLMo2's attention
and residual logic. Do NOT fork the entire model — maintain compatibility
with HuggingFace `olmo2` model loading.

---

## Training Phases

### Phase 1: Learn Topology (Frozen OLMo)

| Component           | Status   |
|---------------------|----------|
| OLMo2-1B            | ❄ frozen |
| Qwen-3-Embedding    | ❄ frozen |
| Structure Predictor  | 🔥 trainable |

| Hyperparameter       | Value                |
|----------------------|----------------------|
| Data                 | Dolma (streamed)     |
| Tokens               | 5–10B               |
| Sequence length      | 1024                 |
| Batch size           | TBD (start 32)       |
| LR                   | 3e-4                |
| Optimizer            | AdamW (β1=0.9, β2=0.999, wd=0.01) |
| LR schedule          | Cosine decay to 0   |
| τ annealing          | 5 → 0.2 (cosine)    |
| Sparsity λ           | 0 → 0.01 (linear ramp over first 20% steps) |
| Hardware             | A40 × 4             |
| **Gate criterion**   | NLL ≤ dense baseline on held-out eval |

**Sparsity loss**: λ · mean(A) — encourages the predictor to turn off
unnecessary connections rather than leaving everything ON.

### Phase 2: Joint Continued Pre-Training (future)

| Component           | Status   |
|---------------------|----------|
| OLMo2-1B            | 🔥 unfrozen |
| Qwen-3-Embedding    | ❄ frozen  |
| Structure Predictor  | 🔥 trainable |

| Hyperparameter       | Value                |
|----------------------|----------------------|
| Tokens               | 20–50B              |
| LR (OLMo)           | 3e-5                |
| LR (Predictor)       | 1e-4                |
| τ continues          | → 0.1               |
| Hardware             | A100 × 4            |

Phase 2 is out of scope for initial implementation. Build the training
loop to support it (differential LR groups) but don't implement the
unfreezing logic yet.

---

## Directory Structure

```
dagformer/
├── CLAUDE.md               # This file — the spec
├── README.md               # Public-facing project description
├── pyproject.toml           # Dependencies and project metadata
│
├── configs/
│   ├── sanity_check.yaml    # Tiny run: 1K steps, verify NLL matches baseline
│   ├── ablation_rank.yaml   # Sweep r ∈ {8, 16, 32, 64}
│   ├── ablation_tau.yaml    # Sweep τ_init, τ_final
│   ├── ablation_lambda.yaml # Sweep sparsity coefficient
│   └── phase1_full.yaml     # Full Phase 1 training config
│
├── src/
│   ├── __init__.py
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── olmo_graph.py    # Modified OLMo2 forward with adjacency injection
│   │   ├── predictor.py     # Structure predictor (Qwen + MLP + Gumbel)
│   │   └── pipeline.py      # Combines predictor + OLMo into one forward call
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── dolma.py         # Streaming dataloader for Dolma
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py       # Pure PyTorch training loop
│   │   ├── schedulers.py    # τ annealing, sparsity ramp, LR schedule
│   │   └── checkpointing.py # Save/load with topology statistics
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging.py       # Wandb + console logging
│       └── topology.py      # A matrix analysis: sparsity, Jaccard, per-layer stats
│
├── scripts/
│   ├── train.py             # Entry point: python scripts/train.py --config configs/...
│   ├── eval.py              # Evaluate: compare NLL with/without predictor
│   ├── sanity_check.py      # Verify: A=all-ones reproduces baseline NLL
│   └── visualize_topology.py # Plot adjacency matrices, gate distributions
│
└── tests/
    ├── test_olmo_graph.py   # Forward pass matches baseline when A=1
    ├── test_predictor.py    # Output shapes, gradient flow
    └── test_gumbel.py       # Gumbel-Sigmoid properties at various τ
```

---

## Implementation Order

Build and **test each module in isolation** before combining.

### Step 0: Environment Setup
- [ ] `pyproject.toml` with deps: torch, transformers, datasets, wandb, pyyaml, einops
- [ ] Verify OLMo2-1B loads: `AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B")`
- [ ] Verify Qwen loads: `AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")`
- [ ] Verify Dolma streaming: `load_dataset("allenai/dolma", ..., streaming=True)`

### Step 1: `model/olmo_graph.py` — Modified OLMo Forward
- [ ] Load OLMo2-1B, identify where head outputs merge into residual
- [ ] Implement gate injection: multiply head outputs by A values
- [ ] **Sanity check**: A = all-ones → NLL matches vanilla forward within 0.01
- [ ] **Sanity check**: A = all-zeros → NLL is very high (model is broken)
- [ ] Verify gradients flow through A (A.requires_grad=True, check A.grad is not None)

### Step 2: `model/predictor.py` — Structure Predictor
- [ ] Qwen encoder wrapper (frozen, returns pooled embedding)
- [ ] MLP decoder: e → h → (U, V) → Z = UV^T
- [ ] Gumbel-Sigmoid: σ((Z + G) / τ) with configurable τ
- [ ] Upper-triangular mask
- [ ] Cascading gate
- [ ] **Test**: output shape is (batch, 256, 256), values in [0,1], upper-tri
- [ ] **Test**: gradient flows from output back to MLP parameters

### Step 3: `model/pipeline.py` — End-to-End Forward
- [ ] Combine predictor + OLMo: tokens → A → modified_forward → NLL
- [ ] Verify full gradient chain: NLL.backward() updates predictor params
- [ ] Profile memory: should fit on single A40 (48GB) for seq_len=1024, batch=1

### Step 4: `training/` — Training Infrastructure
- [ ] Config loading (yaml → dataclass)
- [ ] τ annealing schedule (cosine: τ_init → τ_final)
- [ ] Sparsity λ ramp (linear: 0 → λ_max over warmup fraction)
- [ ] LR schedule (cosine decay)
- [ ] Training loop: forward → loss → backward → step → log
- [ ] Wandb logging: NLL, sparsity(A), τ, gate statistics per layer
- [ ] Checkpointing: save predictor weights + optimizer + step + τ

### Step 5: Sanity Check Run
- [ ] `configs/sanity_check.yaml`: 1K steps, small batch, high τ
- [ ] Verify: loss decreases, A is not collapsing to all-ones or all-zeros
- [ ] Verify: gradient norms are reasonable (not exploding/vanishing)
- [ ] **Decision gate**: if loss doesn't decrease in 1K steps, debug before proceeding

### Step 6: Ablations
- [ ] Rank r ∈ {8, 16, 32, 64}: which gives best NLL-sparsity tradeoff?
- [ ] τ schedule: (5→0.2) vs (2→0.1) vs (10→0.5)
- [ ] Sparsity λ: 0 vs 0.001 vs 0.01 vs 0.1
- [ ] Cascading gate: with vs without

---

## Key Invariants (must always hold)

1. **Baseline reproduction**: When the predictor outputs A = all-ones,
   the NLL must match vanilla OLMo2-1B within 1%.

2. **DAG constraint**: A is always upper-triangular (no cycles).
   Enforced by mask, not by loss.

3. **Gradient flow**: `loss.backward()` must produce non-None gradients
   for all predictor parameters. Check after every architectural change.

4. **Memory budget**: Phase 1 must fit on 4× A40 (48GB each) with
   seq_len=1024. If it doesn't, reduce batch size before changing
   architecture.

5. **Deterministic eval**: At eval time, use hard threshold (A > 0.5)
   with no Gumbel noise. Eval NLL must be reported with hard gates.

---

## Logging & Monitoring

Log to **Wandb** at every step:

| Metric                 | What it tells you                        |
|------------------------|------------------------------------------|
| `train/nll`            | Primary objective                        |
| `train/sparsity_loss`  | λ · mean(A)                              |
| `train/total_loss`     | NLL + sparsity                           |
| `eval/nll_soft`        | NLL with soft gates (Gumbel, current τ)  |
| `eval/nll_hard`        | NLL with hard gates (threshold 0.5)      |
| `eval/nll_baseline`    | NLL with A=1 (should be constant)        |
| `topology/sparsity`    | 1 - mean(A)                              |
| `topology/seq_gate_on` | Fraction of sequential gates > 0.5       |
| `topology/hyp_gate_on` | Fraction of hyperconnection gates > 0.5  |
| `topology/jaccard_var` | Variance of Jaccard similarity across batch |
| `schedule/tau`         | Current temperature                      |
| `schedule/lambda`      | Current sparsity coefficient             |
| `gradients/predictor_norm` | Total gradient norm                  |

**Collapse detection**: If `topology/sparsity` < 0.01 or > 0.99 for
100 consecutive steps, something is wrong. Log a warning.

---

## Dependencies

```
torch >= 2.2
transformers >= 4.40
datasets
wandb
pyyaml
einops
```

No Accelerate, no Lightning. Pure PyTorch with `torch.nn.parallel.DistributedDataParallel`
for multi-GPU. Keep it simple and transparent.

---

## What NOT to Build (out of scope)

- Phase 2 (joint CPT) — design for it, don't implement yet
- Diffusion-based topology predictor — future work
- Custom CUDA kernels for sparse attention — use dense ops with masking
- Support for models other than OLMo2-1B — hardcode for now
- Fancy hyperparameter search — manual ablations are fine

---

## Code Style

- Type hints everywhere
- Docstrings on all public functions
- Config dataclasses, not raw dicts
- `assert` liberally for shape checks in forward passes
- No silent failures: if something is wrong, crash loudly
- Prefer explicit over clever: `for layer in layers` over `map(lambda ...)`
