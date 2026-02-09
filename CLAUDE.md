# CLAUDE.md — DAGFormer Project Specification

**Read this file in full before writing any code.** This document is the
single source of truth for all design decisions. If something contradicts
the README or any other file, this document wins.

---

## 1. What Is This Project?

**DAGFormer** trains a small neural network (the "structure predictor") to
predict, for each token, the optimal wiring diagram (a DAG) of a frozen
1B-parameter language model (OLMo2-1B). The predicted DAG controls which
attention heads talk to which other heads. The entire system is trained
end-to-end with language modeling loss — no labeled topology data needed.

### Why?

Standard transformers have a fixed sequential computation graph: layer 0 →
layer 1 → ... → layer 15. Every input sees the same wiring. We showed via
expensive oracle search that **context-dependent topologies** can reduce
next-token prediction loss (NLL) from 2.58 to 0.12 (median across 50
evaluation windows, 100% of windows improved). The oracle search is too
slow to use at scale (500 gradient steps per window), so we need a learned
predictor that produces the topology in a single forward pass.

### What is NOT this project?

- This is NOT the oracle search codebase (that exists separately)
- This is NOT a Mixture-of-Experts project (despite the repo history)
- This does NOT modify the OLMo2-1B weights in Phase 1
- This does NOT implement Phase 2 (joint training) yet — only the
  infrastructure to support it later

---

## 2. Architecture — Exact Specification

### 2.1 The Computation Graph of OLMo2-1B

OLMo2-1B (HuggingFace ID: `allenai/OLMo-2-0425-1B`) has:

- **16 transformer layers**, each with **16 attention heads**
- This gives **256 "nodes"** total: node `(l, h)` = layer `l`, head `h`
- We flatten to a single index: `node_id = l * 16 + h` (0-indexed)

**Standard forward pass** — each layer does:
```python
# Input: residual (shared across all heads in this layer)
normed = RMSNorm(residual)
attn_out = self_attn(normed)          # all 16 heads compute in parallel,
                                       # outputs concatenated and projected by o_proj
residual = residual + attn_out         # attention residual connection
normed2 = RMSNorm(residual)
mlp_out = MLP(normed2)
residual = residual + mlp_out          # MLP residual connection
```

The **residual stream** at the start of layer `l` is therefore:
```
residual_l = embedding + Σ_{l'<l} (attn_output[l'] + mlp_output[l'])
           = embedding + Σ_{l'<l} (Σ_h head_output[l',h] + mlp_output[l'])
```
where `head_output[l',h]` is head h's individual contribution to the
attention output (its slice of `o_proj`, see §2.2). ALL heads in layer l
see the SAME `residual_l` as input — there is no per-head differentiation
in standard transformers.

### 2.2 The Adjacency Matrix A

We introduce a **256×256 adjacency matrix A** that controls information
routing between attention heads across layers.

```
A[i][j] ∈ [0, 1]   where i = source node, j = target node
                     i = l_i * 16 + h_i,  j = l_j * 16 + h_j
```

#### Mask: Block-Upper-Triangular (NOT element-upper-triangular)

**CRITICAL**: The mask is based on LAYER indices, not node indices.

```python
# CORRECT: block-upper-triangular based on layer
mask[i, j] = 1  if  layer(j) > layer(i)    # i.e. j//16 > i//16
mask[i, j] = 0  if  layer(j) <= layer(i)   # same layer or backward

# WRONG: do NOT use torch.triu() — it would allow same-layer connections
# e.g. triu would set mask[0,15]=1, but both are in layer 0
```

Heads in the same layer execute in parallel and cannot see each other's
outputs. Only cross-layer forward connections are meaningful.

#### Connection Count

| Type | Definition | Count | Role |
|------|-----------|-------|------|
| **Adjacent-layer** | `layer(j) = layer(i) + 1`, all head pairs | 15 × 16 × 16 = **3,840** | These exist in standard transformer (via shared residual). When gated to 1, behavior matches baseline. |
| **Skip** | `layer(j) > layer(i) + 1`, all head pairs | 105 × 16 × 16 = **26,880** | These do NOT exist in standard transformer. They are additional direct routes that bypass intermediate layers. |
| **Total** | All entries where `layer(j) > layer(i)` | **30,720** | |

For logging and analysis, label connections as "adjacent" or "skip", but
the forward pass treats all 30,720 entries identically.

> **Note on "31K" count**: The oracle search reported "256 sequential +
> 30,720 hyperconnection ≈ 31K" using a different parameterization
> (separate per-head activity gates + routing gates). In our unified
> 256×256 matrix, there are exactly 30,720 free entries. Both represent
> the same underlying structure.

#### What "head output" means (resolves shape ambiguity)

In HuggingFace OLMo2, the `o_proj` layer concatenates all 16 heads and
projects back to model_dim:

```python
# Inside self_attn:
# Each head computes: attn_weights @ V → [batch, seq, head_dim]  (head_dim = 128)
# All heads concatenated: [batch, seq, 16 * 128] = [batch, seq, 2048]
# Then: o_proj([batch, seq, 2048]) → [batch, seq, 2048]
```

The `o_proj` weight matrix `W_o ∈ R^{2048 × 2048}` can be viewed as 16
column blocks: `W_o = [W_o^0 | W_o^1 | ... | W_o^15]`, each `W_o^h ∈
R^{2048 × 128}`. The full attention output is:

```
attn_output = Σ_h W_o^h @ head_value_h = Σ_h head_output[l, h]
```

So **`head_output[l, h]` is `W_o^h @ head_value_h`, shape `[batch, seq,
model_dim]` (= 2048)**. This is each head's contribution to the attention
output in MODEL DIM space. Heads can be summed directly because they're
already in the same 2048-dim space.

#### Per-Head Input Assembly (the core modification)

In DAGFormer, each head j = (l_j, h_j) has its **own input**, assembled
from three sources:

```python
input_j = embedding                               # (1) always present
        + Σ_{l' < l_j} mlp_output[l']             # (2) all prior MLP outputs, NOT gated
        + Σ_{i: layer(i) < l_j} A[i,j] * head_output[i]   # (3) gated head outputs
```

**Source (1) — Token embedding**: The output of the token embedding layer
(before any transformer layer). Always included for every head. This is
NOT part of A. Think of it as the "seed" of the computation.

**Source (2) — MLP outputs**: Each layer's MLP computes on a shared
aggregated state (see below) and its output is added to ALL downstream
head inputs equally. MLP outputs are **never gated by A**. This keeps
the MLP's contribution identical to the standard forward pass.

**Source (3) — Gated head outputs**: The only part controlled by A.
Each entry A[i,j] scales how much of head i's output reaches head j's
input. Different heads h within the same layer l_j can receive different
weighted combinations of prior head outputs — this is what makes the
256×256 per-head routing meaningful.

#### Baseline Reproduction Proof (resolves the "double-counting" concern)

When A[i,j] = 1 for ALL 30,720 valid entries:

```
input_j (where j is in layer l) 
    = embedding + Σ_{l'<l} mlp_output[l'] + Σ_{i: layer(i)<l} 1.0 * head_output[i]
    = embedding + Σ_{l'<l} mlp_output[l'] + Σ_{l'<l} Σ_h head_output[l',h]
    = embedding + Σ_{l'<l} (mlp_output[l'] + attn_output[l'])
    = residual_l   ← the standard residual stream!
```

All heads in the same layer l receive the SAME input (because A=1 gives
the same weighted sum for all target heads). This equals the standard
`residual_l`. Therefore **A=1 exactly reproduces vanilla OLMo2-1B**. ✓

There is NO double-counting because each head output appears exactly once
in the sum. Head (0,5)'s output is included via the gated sum (source 3),
and it flows to layer 8 only via that direct A[0*16+5, 8*16+h] entry, NOT
also through intermediate layers. The intermediate layers' head outputs
are separate entries in the sum.

#### MLP Execution (not gated, shared input)

After all 16 heads in layer l compute, the MLP runs:

```python
# MLP input = standard aggregation (same as baseline when A=1)
mlp_input_l = embedding
            + Σ_{l'<l} mlp_output[l']
            + Σ_{l'<=l} attn_output[l']    # includes current layer's heads
# where attn_output[l'] = Σ_h head_output[l', h]  (UNGATED sum of this layer's heads)

# IMPORTANT: MLP always sees the UNGATED sum of its own layer's head outputs.
# The gating by A only affects what OTHER layers' heads receive.
# This is necessary for baseline reproduction.

mlp_output[l] = MLP_l(RMSNorm(mlp_input_l))
```

The MLP input includes the **ungated** sum of the current layer's head
outputs. Gating only affects cross-layer routing. This design ensures
that A does not interfere with the MLP's computation within its own layer.

#### Layer 0 Special Case

Layer 0's 16 heads have no prior head outputs (no nodes with layer < 0).
Their input is simply:
```python
input_j (j in layer 0) = embedding   # no prior heads, no prior MLPs
```
This requires no special handling — the formula naturally produces
`embedding` when there are no prior sources.

### 2.3 The Structure Predictor

The predictor takes the current context and outputs A. Architecture:

```
Input: raw text → Qwen tokenizer → qwen_ids [batch, qwen_seq_len]
       (qwen_seq_len may differ from OLMo's seq_len — that's fine,
        because mean pooling collapses to a single vector per sequence)
         │
         ▼
   ┌─────────────────────────────────┐
   │  Qwen-3-Embedding-0.6B          │
   │  HF ID: "Qwen/Qwen3-Embedding-0.6B" │
   │                                  │
   │  This is a TEXT EMBEDDING MODEL, │
   │  NOT a generative LLM. It takes  │
   │  text and produces a single      │
   │  fixed-size vector per sequence. │
   │                                  │
   │  FROZEN — no gradients ever      │
   │  Uses its OWN tokenizer          │
   │  (separate from OLMo's)          │
   │                                  │
   │  Input: raw text → Qwen tokenizer│
   │  → Qwen forward → last_hidden   │
   │  Pooling: mean over seq_len dim  │
   │  → embedding e ∈ R^d             │
   │    (d = model.config.hidden_size │
   │     — do NOT hardcode, query at  │
   │     runtime from the model)      │
   │                                  │
   │  e is a SINGLE VECTOR per        │
   │  sequence — it summarizes the    │
   │  entire context. The predictor   │
   │  MLP then maps e → A matrix.     │
   │  Qwen has nothing to do with     │
   │  OLMo's vocabulary or tokens.    │
   └─────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────────────┐
   │  MLP Decoder (TRAINABLE)         │
   │                                  │
   │  Linear(d, hidden_dim)           │
   │  → GELU                          │
   │  → Linear(hidden_dim, hidden_dim)│
   │  → GELU                          │
   │  → two heads:                    │
   │      Linear(hidden_dim, 256 * r) │  → reshape to U ∈ R^{256×r}
   │      Linear(hidden_dim, 256 * r) │  → reshape to V ∈ R^{256×r}
   │                                  │
   │  hidden_dim = 1024 (default)     │
   │  r = rank hyperparameter         │
   │    ablate: r ∈ {8, 16, 32, 64}   │
   │    default: r = 32               │
   └─────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────────────┐
   │  Low-Rank Logits                 │
   │                                  │
   │  Z = U @ V.transpose(-1, -2)    │
   │  Z ∈ R^{batch × 256 × 256}      │
   │                                  │
   │  This is the logit matrix before │
   │  any gating or masking.          │
   └─────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────────────┐
   │  Block-Upper-Triangular Mask     │
   │                                  │
   │  mask[i,j] = 1 if j//16 > i//16 │
   │  (layer(j) > layer(i))          │
   │                                  │
   │  ⚠ Do NOT use torch.triu() —     │
   │  it allows same-layer connections│
   │                                  │
   │  Z_masked = Z * mask + (-1e9) * (1 - mask)  │
   │  (force invalid positions to     │
   │   -inf so sigmoid → 0)           │
   └─────────────────────────────────┘
         │
         ▼
   ┌─────────────────────────────────┐
   │  Gumbel-Sigmoid (3 modes)       │
   │                                  │
   │  MODE 1 — TRAINING:             │
   │    G ~ Logistic(0, 1)            │
   │      (= log(U) - log(1-U),      │
   │       U ~ Uniform(0,1))          │
   │    A = σ((Z_masked + G) / τ)     │
   │    Gumbel noise G adds stochastic│
   │    exploration. Gradients flow    │
   │    through σ naturally (this is  │
   │    continuous relaxation, NOT    │
   │    STE — no straight-through     │
   │    estimator needed here).       │
   │                                  │
   │  MODE 2 — EVAL SOFT:            │
   │    A = σ(Z_masked / τ)           │
   │    NO Gumbel noise. Deterministic│
   │    soft gates. Values in (0,1).  │
   │    Used for eval/nll_soft metric.│
   │                                  │
   │  MODE 3 — EVAL HARD:            │
   │    A = (Z_masked > 0).float()    │
   │    NO Gumbel noise. Binary 0/1.  │
   │    Threshold at logit=0 (= prob  │
   │    0.5). Used for eval/nll_hard  │
   │    and for final inference.      │
   │                                  │
   │  τ = temperature, annealed       │
   │  during training (see §3)        │
   └─────────────────────────────────┘

**WHY Gumbel-Sigmoid — the gradient problem and its solution:**

At inference we want binary A ∈ {0, 1} (discrete routing decisions).
But discrete decisions have zero gradient — you can't backprop through
`(Z > 0).float()`. The predictor MLP would receive no learning signal.

Gumbel-Sigmoid is a **continuous relaxation** (also called the "concrete
distribution" / "surrogate gradient" technique):

```
Training:  A = σ((Z + G) / τ)    ← continuous, differentiable
                                     ∂A/∂Z = A(1-A)/τ  ← well-defined gradient
                                     backprop: ∂Loss/∂Z = ∂Loss/∂A · ∂A/∂Z

Inference: A = (Z > 0).float()   ← discrete, but no gradient needed
```

The complete gradient chain during training:
```
Loss (NLL from OLMo)
  → ∂Loss/∂logits            (OLMo's output layer)
  → ∂logits/∂head_inputs     (through OLMo's frozen layers — computed
                               but NOT used to update OLMo weights)
  → ∂head_inputs/∂A          (the gate multiplication: input_j = Σ A[i,j] * head_out[i])
  → ∂A/∂Z                    (sigmoid derivative: A(1-A)/τ — THIS is the surrogate)
  → ∂Z/∂(U,V)                (low-rank matmul: Z = UV^T)
  → ∂(U,V)/∂MLP_params       (the trainable predictor MLP)
  → optimizer updates MLP_params
```

Key points:
- OLMo is frozen but its forward computation is ON the gradient tape
  (it's a differentiable function of A). Gradients flow THROUGH OLMo
  back to A, even though OLMo's own parameters don't get updated.
- The sigmoid σ acts as a differentiable surrogate for the discrete
  threshold. As τ → 0, σ((Z+G)/τ) approaches a step function, but
  during training τ is always > 0 so gradients are always nonzero.
- Gumbel noise G provides stochastic exploration: even if Z is small,
  the noise can push A above or below 0.5, letting the model discover
  which connections matter.
- NO straight-through estimator (STE) is needed. The sigmoid itself
  provides the gradient. STE would be needed if we hard-thresholded
  during training, but we don't.
         │
         ▼
   ┌─────────────────────────────────┐
   │  Cascading Gate                  │
   │                                  │
   │  Purpose: if node j has no       │
   │  incoming edges, it has no info  │
   │  to propagate, so kill outgoing. │
   │                                  │
   │  ONE-PASS computation (not       │
   │  sequential by layer):           │
   │                                  │
   │  1. Compute ALL incoming sums:   │
   │     inc_j = Σ_i A[i, j]  ∀j     │
   │  2. Compute ALL gates:           │
   │     g_j = σ(k * inc_j)   ∀j     │
   │  3. Apply ALL gates at once:     │
   │     A[j, :] *= g_j       ∀j     │
   │                                  │
   │  This is a single vectorized op, │
   │  NOT a layer-by-layer cascade.   │
   │  All incoming sums use the       │
   │  ORIGINAL A values (before any   │
   │  gates are applied).             │
   │                                  │
   │  k = 5.0 (fixed scalar)         │
   │  k can be made learnable later.  │
   │  This is fully differentiable.   │
   │                                  │
   │  EVAL HARD MODE: After hard-     │
   │  thresholding A to binary 0/1,  │
   │  the cascading gate is also      │
   │  hard: g_j = 1 if inc_j > 0,    │
   │  else g_j = 0. (Because σ(5*0)  │
   │  = 0.5 would be wrong for       │
   │  binary gates — a disconnected   │
   │  node should be fully killed,    │
   │  not half-alive.)                │
   │                                  │
   │  SOFT MODE NOTE: In training and │
   │  eval-soft mode, the cascading   │
   │  gate uses σ(k * inc_j) with the│
   │  ORIGINAL (pre-gate) A values.   │
   │  If all incoming gates are small │
   │  (e.g. 0.01 each), inc_j can    │
   │  still be > 0, giving g_j > 0.5.│
   │  This is INTENTIONAL: in soft    │
   │  mode, "weakly connected" is a   │
   │  valid state (the gradient can   │
   │  still flow). The cascading gate │
   │  is a soft penalty, not a hard   │
   │  kill, during training.          │
   └─────────────────────────────────┘
         │
         ▼
   Output: A ∈ [0,1]^{batch × 256 × 256}, block-upper-triangular
           (30,720 free entries, rest forced to 0)
```

### 2.4 End-to-End Pipeline

```
raw text ──→ [Qwen tokenizer] ──→ qwen_ids ──→ [Qwen encoder] ──→ e ──→ [Predictor MLP] ──→ A
   │                                                                                         │
   │                                                                                         ▼
   └──→ [OLMo tokenizer] ──→ olmo_ids ──→ [OLMo2-1B modified forward with A] ──→ logits ──→ NLL
                                                                                             │
                                                                         ∇ backprop to predictor MLP
```

**Tokenization**: Qwen and OLMo have DIFFERENT tokenizers and vocabularies.
The dataloader produces raw text strings. Each model tokenizes independently:
```python
# In the dataloader or pipeline:
raw_text = batch["text"]                              # list of strings
qwen_ids = qwen_tokenizer(raw_text, ...)["input_ids"] # Qwen's token IDs
olmo_ids = olmo_tokenizer(raw_text, ...)["input_ids"]  # OLMo's token IDs
# These are DIFFERENT tensors with DIFFERENT lengths and vocabularies.
# Qwen produces a pooled embedding (one vector per sequence).
# OLMo produces per-token logits for NLL computation.
```

- Qwen and OLMo are FROZEN (Phase 1). Only the MLP decoder trains.
- The same **raw text** goes to both Qwen and OLMo, but they use
  **separate tokenizers**. Qwen tokenizes independently to produce an
  embedding; OLMo tokenizes independently for language modeling.
  Token IDs are NOT shared between the two models.
- Qwen's output (a single pooled vector per sequence) goes to the
  predictor MLP → A matrix. OLMo uses A in its modified forward pass.
- Loss = NLL (from OLMo) + λ · mean(A)  (sparsity regularization)

---

## 3. Training — Exact Specification

### 3.1 Phase 1: Learn Topology (IMPLEMENT THIS)

| What | Frozen/Trainable |
|------|-----------------|
| OLMo2-1B (`allenai/OLMo-2-0425-1B`) | ❄ FROZEN, no grad |
| Qwen-3-Embedding (`Qwen/Qwen3-Embedding-0.6B`) | ❄ FROZEN, no grad |
| Structure Predictor (MLP decoder) | 🔥 TRAINABLE |

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Dataset | Dolma v1.7 (`allenai/dolma`, `name="v1_7"`, streamed) | Specify version explicitly |
| Token budget | 5–10B | Configurable |
| Sequence length | 1024 | OLMo token count, matches oracle search |
| Batch size | 32 (start) | Reduce if OOM |
| Learning rate | 3e-4 | For predictor MLP only |
| Optimizer | AdamW | β1=0.9, β2=0.999, weight_decay=0.01 |
| LR schedule | Cosine decay to 0 | Standard |
| τ (temperature) init | 5.0 | |
| τ final | 0.2 | |
| τ schedule | Cosine annealing | τ(t) = τ_f + 0.5(τ_i - τ_f)(1 + cos(πt/T)) |
| Sparsity λ max | 0.01 | |
| Sparsity ramp | Linear 0 → λ_max over first 20% of steps | |
| Hardware | 4× A40 (48GB each) | Use DDP |

**Loss function:**
```python
total_loss = nll_loss + lambda_t * A.mean()
```
where `lambda_t` ramps linearly from 0 to `lambda_max` over the first 20%
of training steps.

**τ schedule (exact formula):**
```python
tau_t = tau_final + 0.5 * (tau_init - tau_final) * (1 + math.cos(math.pi * step / total_steps))
```

### 3.1.1 Data Processing Pipeline

**Dolma version**: Use `allenai/dolma` with `name="v1_7"`. If v1_7 is not
available on HuggingFace, fall back to whatever default version loads.
Verify by printing the dataset info at startup and logging to wandb.

**Sequence packing** (critical for training efficiency):

Dolma contains documents of varying lengths. Do NOT pad short documents
or discard them. Instead, **pack multiple documents into fixed-length
sequences**:

```python
# Pseudocode for sequence packing:
buffer = []
for doc in dolma_stream:
    olmo_tokens = olmo_tokenizer(doc["text"], add_special_tokens=False)["input_ids"]
    buffer.extend(olmo_tokens)
    buffer.append(olmo_tokenizer.eos_token_id)  # document separator

    while len(buffer) >= seq_len + 1:  # +1 for labels (next-token prediction)
        chunk = buffer[:seq_len + 1]
        buffer = buffer[seq_len + 1:]
        yield {
            "olmo_ids": chunk[:seq_len],      # input
            "olmo_labels": chunk[1:seq_len+1], # shifted target
            "raw_text": olmo_tokenizer.decode(chunk[:seq_len])  # for Qwen
        }
```

This means:
- No padding ever. Every token contributes to NLL.
- A single training sequence may contain parts of multiple documents,
  separated by EOS tokens. The causal mask handles this correctly (each
  token can only attend to prior tokens, so cross-document leakage is
  minimal and standard practice).
- `raw_text` is decoded from OLMo tokens to feed to Qwen. This is a
  slight approximation (Qwen re-tokenizes the decoded text), but Qwen
  only needs to understand the gist of the context to produce a good
  embedding — exact token alignment doesn't matter.
- **Qwen sees the entire packed sequence** (which may contain multiple
  document fragments separated by EOS). This is intentional: the
  predictor should condition on exactly what OLMo will process. Qwen's
  mean-pooling produces a single summary vector of the full 1024-token
  window, which is the right granularity — one A matrix per window.
- Qwen's `seq_len` will differ from OLMo's 1024 (different tokenizer
  granularity), but this is fine because Qwen's output is mean-pooled
  to a single vector regardless of input length.

**Qwen input format**: Qwen3-Embedding-0.6B is an embedding model.
**Decision**: Use raw text directly (no prefix). Rationale: our input is
a packed sequence of document fragments, not a search query or passage
in the retrieval sense. Prefixes like "query:" or "passage:" are designed
for retrieval tasks and would be semantically misleading here. The Qwen
encoder just needs a general-purpose text representation.

Log `qwen_input_prefix: ""` to wandb config for reproducibility. If
future experiments show that a prefix helps, it can be changed via the
`qwen_input_prefix` config field.

### 3.2 Evaluation Data

Use a **fixed held-out subset of Dolma** for eval. Implementation:

```python
# At startup, skip the first N examples to get to a held-out region
eval_dataset = load_dataset("allenai/dolma", name="v1_7", split="train", streaming=True)
eval_dataset = eval_dataset.skip(1_000_000)  # skip 1M examples
eval_dataset = eval_dataset.take(1_000)       # use next 1K as eval set

# Pack these into fixed-length sequences using the same packing logic
# as training (§3.1.1), then cache in memory at startup
eval_batches = list(eval_dataloader)  # ~1K sequences, fits in RAM
```

**Eval caching**: The `skip(1_000_000)` on a streaming dataset is slow
(~minutes). To avoid this cost on every restart, **cache the packed eval
sequences to disk** after the first run:
```python
eval_cache_path = os.path.join(config.save_dir, "eval_cache.pt")
if os.path.exists(eval_cache_path):
    eval_batches = torch.load(eval_cache_path)
else:
    eval_batches = list(build_eval_dataloader(...))
    torch.save(eval_batches, eval_cache_path)
```

**Multi-GPU eval**: Run eval on **rank 0 only**. Other ranks skip eval
and wait at a barrier. This avoids redundant computation and ensures
consistent eval metrics (no need to reduce across GPUs).

Eval runs in **two modes** (both deterministic, no Gumbel noise):
- **Soft**: A = σ(Z / τ) at current temperature. Reports `eval/nll_soft`.
- **Hard**: A = (Z > 0).float(), binary 0/1. Reports `eval/nll_hard`.
Also report `eval/nll_baseline` with A = all-ones (should be constant).

### 3.3 Multi-GPU Strategy

**Standard DDP — each GPU holds a full replica of all three models.**

Memory budget per GPU (fp16/bf16):
```
OLMo2-1B parameters:  ~2.0 GB
Qwen-0.6B parameters: ~1.2 GB
Predictor MLP:         ~0.05 GB
Optimizer states:      ~0.1 GB (only predictor params)
─────────────────────────────
Model total:           ~3.4 GB

Activations (seq_len=1024, batch=8):
  OLMo per-head forward: ~12-16 GB (per-head inputs means 16 separate
    attention computations per layer instead of 1 batched MHA — this
    increases intermediate activation storage significantly vs standard)
  Qwen forward:          ~3 GB
  Per-head A storage:    ~0.5 GB (256×256 × batch × float32)
─────────────────────────────
Activations total:       ~16-20 GB

Grand total:             ~20-24 GB per GPU (with batch=8)
Headroom on 48GB A40:    ~24-28 GB
```

This fits but is tighter than a standard OLMo forward pass. If OOM,
reduce batch_size to 4 first (halves activation memory). If still OOM,
use **gradient accumulation** to maintain effective batch size:
```python
# Example: effective_batch=8, micro_batch=2, accumulation_steps=4
accumulation_steps = config.batch_size // config.micro_batch_size
for micro_step in range(accumulation_steps):
    loss = forward(micro_batch) / accumulation_steps
    loss.backward()
optimizer.step()
optimizer.zero_grad()
```
Add `micro_batch_size` to config (default: same as `batch_size`, i.e. no
accumulation). The per-head computation path creates ~2-3x more
intermediates than standard batched MHA because we cannot fuse across
heads when each has a different input.

**Gradient checkpointing**: Since OLMo is frozen (no gradients computed
for its parameters), we do NOT need to store OLMo's intermediate
activations for backprop through OLMo's weights. However, we DO need
gradients to flow through OLMo's forward pass back to A (and then to
the predictor). This means OLMo's forward activations are needed for
the chain rule through A's gate multiplications, but NOT for OLMo's
own parameter gradients.

If memory is still tight, apply `torch.utils.checkpoint` to the OLMo
layer loop — recompute each layer's forward pass during backward instead
of storing all intermediates. This trades compute for memory.
Gradient checkpointing is optional; try without it first.

**Storing head_outputs**: The forward pass accumulates `head_outputs` for
all 256 nodes (each `[batch, seq, 2048]`). At fp16 with batch=8,
seq=1024: 256 × 8 × 1024 × 2048 × 2 bytes ≈ 8.6 GB. This is the
dominant memory cost. Gradient checkpointing can reduce this by
recomputing head outputs layer-by-layer during backward.

Use `DistributedDataParallel` wrapping ONLY on the predictor MLP (the
only trainable component). The frozen Qwen and OLMo models do not need
DDP wrapping — just load them on each GPU.

```python
# DDP setup
predictor = StructurePredictor(config).to(device)
predictor = DDP(predictor, device_ids=[local_rank])

# Frozen models — no DDP needed
olmo = AutoModelForCausalLM.from_pretrained(...).to(device).eval()
qwen = AutoModel.from_pretrained(...).to(device).eval()

# Gradient disabled for frozen models
for p in olmo.parameters(): p.requires_grad_(False)
for p in qwen.parameters(): p.requires_grad_(False)
```

Data parallelism: Dolma is a streaming `IterableDataset` — do NOT use
`DistributedSampler` (which requires map-style datasets). Instead, shard
the stream manually:
```python
dataset = load_dataset("allenai/dolma", name="v1_7", split="train", streaming=True)
dataset = dataset.shard(num_shards=world_size, index=rank)
```
Each GPU processes a disjoint shard. Gradients are synchronized by DDP.

### 3.4 o_proj Bias and Weight Layout

OLMo2-1B uses **no bias** in its linear layers (`bias=False` for q_proj,
k_proj, v_proj, o_proj, and MLP layers). This is standard for modern LLMs.

**Verify this at runtime:**
```python
assert not model.layers[0].self_attn.o_proj.bias, \
    "Expected no bias in o_proj — update per-head splitting if bias exists"
```

If a future model version adds bias, the per-head split must also split
the bias: `bias_h = bias[h * head_dim : (h+1) * head_dim]` for input
projections, or `bias / num_heads` for o_proj (since it's additive).
But for OLMo2-1B, this is not needed.

**o_proj weight layout** for per-head splitting:
```python
# o_proj.weight: [model_dim, model_dim] = [2048, 2048]
# This maps concatenated head outputs → model_dim
#
# Split by INPUT dimension (head_dim chunks):
# o_proj_h.weight = o_proj.weight[:, h*head_dim : (h+1)*head_dim]
# Shape: [2048, 128]
#
# head_output[l, h] = attn_values_h @ o_proj_h.weight.T
# Shape: [batch, seq, 128] @ [128, 2048] → [batch, seq, 2048]
```

### 3.5 Phase 2: Joint CPT (DO NOT IMPLEMENT — FUTURE WORK)

In Phase 2, OLMo would be unfrozen and co-trained with the predictor using
differential learning rates (OLMo: 3e-5, predictor: 1e-4). The training
loop should be DESIGNED to support this (parameter groups with different
LRs) but the actual unfreezing logic should NOT be implemented yet.

---

## 4. OLMo2-1B Modification — Implementation Guide

This is the hardest part. The goal: intercept OLMo2-1B's forward pass to
apply the adjacency matrix A without forking the model code.

### 4.1 Strategy: Hook-Based or Subclass

**Option A (preferred): Monkey-patch the attention forward.**
- Load the model normally via HuggingFace
- Replace each layer's attention module's forward method with a wrapped
  version that accepts and applies A
- Pro: no code duplication, stays in sync with HF updates
- Con: fragile if HF changes internal API

**Option B: Subclass the model.**
- Subclass `OlmoForCausalLM` and override the relevant methods
- Pro: cleaner, more explicit
- Con: more code to maintain

Choose whichever is cleaner after inspecting the actual OLMo2 source.
The key requirement is: **when A is not provided or is all-ones, the
output must be IDENTICAL to vanilla OLMo2-1B.**

### 4.2 What Exactly to Modify

**Reference pseudocode — this is the exact semantics to implement:**

```python
def dagformer_forward(model, olmo_ids, A, input_norm="none"):
    """
    Args:
        model: OLMo2-1B loaded from HuggingFace
        olmo_ids: [batch, seq_len] — tokenized by OLMo's tokenizer
        A: [batch, 256, 256] — block-upper-triangular gate matrix
           (produced by the predictor from Qwen embeddings — but this
            function doesn't know about Qwen, it just takes A as input)
        input_norm: normalization method for assembled head inputs
    Returns:
        logits: [batch, seq_len, vocab_size]
    """
    batch, seq = olmo_ids.shape
    model_dim = 2048
    num_layers = 16
    num_heads = 16

    # Token embedding (always included, not gated)
    embedding = model.embed_tokens(olmo_ids)  # [batch, seq, 2048]

    # Storage for per-head outputs (in model_dim space)
    head_outputs = {}  # (layer, head) -> [batch, seq, model_dim]
    mlp_outputs = {}   # layer -> [batch, seq, model_dim]

    for l in range(num_layers):
        layer_module = model.layers[l]

        # === ASSEMBLE PER-HEAD INPUTS ===
        # Base: embedding + all prior MLP outputs (shared, ungated)
        base_l = embedding.clone()
        for prev_l in range(l):
            base_l = base_l + mlp_outputs[prev_l]

        # Per-head: add gated head outputs from earlier layers
        # This is the KEY difference from standard transformer:
        # each head h in layer l gets its OWN input.
        per_head_inputs = []
        for h in range(num_heads):
            j = l * num_heads + h
            assembled = base_l.clone()  # [batch, seq, model_dim]

            # Gated sum of all prior head outputs
            gated_sum = torch.zeros_like(base_l)
            for src_l in range(l):
                for src_h in range(num_heads):
                    i = src_l * num_heads + src_h
                    gate = A[:, i, j]  # [batch]
                    gated_sum = gated_sum + gate[:, None, None] * head_outputs[(src_l, src_h)]

            # Apply normalization ONLY to gated_sum, then add to base (see §6.1)
            assembled = base_l + apply_norm(gated_sum, method=input_norm)
            per_head_inputs.append(assembled)

        # === RUN ATTENTION WITH PER-HEAD INPUTS ===
        # This requires splitting the attention computation:
        # 1. Each head h gets its own Q from per_head_inputs[h]
        # 2. Each head h gets its own K, V from per_head_inputs[h]
        # 3. Each head computes attention independently
        # 4. Each head's output is projected by its slice of o_proj
        #
        # In practice: need to split q_proj, k_proj, v_proj into per-head
        # projections, run attention per-head, then apply per-head o_proj.
        #
        # Efficient batch implementation:
        # Stack per_head_inputs → [batch, num_heads, seq, model_dim]
        # Apply qkv projection in batch across heads
        # Run attention, apply o_proj per-head
        stacked = torch.stack(per_head_inputs, dim=1)  # [batch, 16, seq, 2048]
        normed = layer_module.input_layernorm(stacked)  # RMSNorm per-head
        # NOTE: RMSNorm(2048) operates on the last dimension. When applied to
        # [batch, 16, seq, 2048], it normalizes each head's input independently.
        # When A=1, all 16 heads have identical inputs, so RMSNorm produces
        # identical outputs — matching the standard single-RMSNorm behavior.
        # No numerical divergence occurs in the baseline case.

        for h in range(num_heads):
            q = q_proj_head(normed[:, h], head=h)  # [batch, seq, head_dim]
            k = k_proj_head(normed[:, h], head=h)
            v = v_proj_head(normed[:, h], head=h)
            attn_out_h = attention(q, k, v)         # [batch, seq, head_dim]
            head_outputs[(l, h)] = o_proj_head(attn_out_h, head=h)  # [batch, seq, model_dim]

        # === RUN MLP (ungated, standard) ===
        # MLP input = base_l + ungated sum of THIS layer's head outputs
        attn_output_l = sum(head_outputs[(l, h)] for h in range(num_heads))
        mlp_input = base_l + attn_output_l  # standard residual
        mlp_outputs[l] = layer_module.mlp(layer_module.post_attention_layernorm(mlp_input))

    # Final: assemble output state = embedding + all heads + all MLPs
    # IMPORTANT: final output uses UNGATED sum of ALL head outputs.
    # A only controls intermediate routing (which heads feed into which).
    # The final output is NOT gated — every head's output contributes
    # equally to the final state. This is a deliberate design choice:
    # (1) it matches the standard residual stream when A=1,
    # (2) it means A controls *how* heads compute (via their inputs),
    #     not *whether* their outputs are used in the final prediction.
    final_state = embedding
    for l in range(num_layers):
        final_state = final_state + mlp_outputs[l]
        for h in range(num_heads):
            final_state = final_state + head_outputs[(l, h)]

    logits = model.lm_head(model.norm(final_state))
    return logits
```

⚠ **This pseudocode is for SEMANTIC CLARITY. The actual implementation
MUST use batched operations for performance:**

```python
# Efficient version for the gated sum:
# Stack all prior head outputs into a tensor
prior_outputs = torch.stack([head_outputs[(l', h')]
                             for l' in range(l) for h' in range(16)], dim=1)
# prior_outputs: [batch, l*16, seq, model_dim]

# Slice A for connections into this layer
A_slice = A[:, :l*16, l*16:(l+1)*16]  # [batch, l*16, 16]

# Batched gated sum: einsum or matmul
# per_head_gated[h] = sum_i A_slice[:, i, h] * prior_outputs[:, i]
per_head_gated = torch.einsum('bih,bisd->bhsd', A_slice, prior_outputs)
# per_head_gated: [batch, 16, seq, model_dim]

# Apply normalization ONLY to per_head_gated, then add base (consistent with §6.1)
per_head_gated = apply_norm(per_head_gated, method=input_norm)
assembled = base_l.unsqueeze(1) + per_head_gated  # [batch, 16, seq, model_dim]
```

**Splitting Q/K/V projections per-head**: OLMo2's `q_proj`, `k_proj`,
`v_proj` are single linear layers projecting from `model_dim` to
`num_heads * head_dim`. To apply different inputs per head:

```python
# Option A: reshape the weight matrix and apply per-head
W_q = layer.self_attn.q_proj.weight  # [2048, 2048]
W_q_heads = W_q.view(num_heads, head_dim, model_dim)  # [16, 128, 2048]
# For each head h: q_h = assembled[:, h] @ W_q_heads[h].T

# Option B: run the full projection on each head's input separately
# Less efficient but simpler

# Option C (RECOMMENDED): run full projection on stacked inputs
# assembled: [batch, 16, seq, 2048]
# Reshape to [batch*16, seq, 2048], run q_proj, reshape back
```

**RoPE (Rotary Position Embeddings)**: OLMo2 applies RoPE to Q and K
AFTER projection, BEFORE the attention dot product. This is critical for
per-head computation:

```python
# Standard OLMo2 attention (simplified):
q = q_proj(hidden_states)    # [batch, seq, num_heads * head_dim]
k = k_proj(hidden_states)    # [batch, seq, num_kv_heads * head_dim]
q, k = apply_rotary_emb(q, k, cos, sin, position_ids)
# Then attention(q, k, v)

# DAGFormer per-head version:
# For each head h with its own input:
q_h = q_proj_head(assembled_h, head=h)   # [batch, seq, head_dim]
k_h = k_proj_head(assembled_h, head=h)   # [batch, seq, head_dim]
q_h, k_h = apply_rotary_emb(q_h, k_h, cos, sin, position_ids)  # SAME cos/sin/position_ids for all heads
v_h = v_proj_head(assembled_h, head=h)   # no RoPE on V
attn_out_h = attention(q_h, k_h, v_h)
```

The cos/sin cache and position_ids are **shared across all heads** (they
depend on sequence position, not on head identity). Extract them once from
the model's rotary embedding module at the start of each layer, then reuse
for all 16 heads. If the implementation fails to apply RoPE, the baseline
reproduction sanity check WILL fail because position information is lost.

**OLMo2-1B uses standard MHA (NOT GQA)**: OLMo2-1B has
`num_attention_heads = 16` and `num_key_value_heads = 16` (every Q head
has its own K and V head). This means per-head splitting is straightforward
— no KV sharing complications. **Verify at runtime:**
```python
config = model.config
assert config.num_attention_heads == 16
assert config.num_key_value_heads == 16, \
    f"Expected MHA (16 KV heads), got {config.num_key_value_heads} — GQA detected, update splitting logic"
```
If a future model uses GQA, the per-head splitting must account for shared
KV heads. But OLMo2-1B does not require this.

**Causal attention mask**: The standard causal mask within each head's
self-attention (preventing token t from attending to tokens t+1, t+2, ...)
is UNCHANGED. DAGFormer's adjacency matrix A controls **cross-layer**
routing (which heads' outputs feed into which heads' inputs). The causal
mask controls **within-sequence** attention (which token positions can
attend to which). These are orthogonal — A operates at the head/layer
level, causal mask operates at the token position level. Use OLMo's
existing causal mask implementation as-is.

### 4.3 Sanity Checks (MUST PASS before proceeding)

1. **Baseline reproduction**: Set A[i,j]=1 for all 30,720 valid cross-layer
   entries, `input_norm: "none"`. NLL must match vanilla OLMo2-1B within
   0.01 nats. This works because A=1 makes every head's input equal to the
   standard residual stream (see §2.2 proof). If this fails, the gate
   injection or per-head splitting logic is wrong.
2. **A = all-zeros** (all 30,720 entries = 0): Every head sees only
   embedding + MLP outputs, no cross-layer attention routing. NLL should
   be significantly higher than baseline.
3. **A = random in [0,1]**: NLL should be between the all-ones and all-zeros
   cases.
4. **Gradient check**: Create A as a leaf tensor with `requires_grad=True`,
   compute NLL, call `.backward()`, verify `A.grad` is not None and has
   nonzero entries at all 30,720 valid positions.
5. **Normalization smoke test**: For each `input_norm` in {gate_mean,
   rms_post, ln_post, rms_pre}, run forward with A=all-ones. NLL will NOT
   match baseline (normalization changes the scale), but must be finite
   (no NaN/Inf). This confirms the norm implementations don't crash.
6. **Per-head input divergence**: Set A to a matrix where different heads
   in the same layer have different gate values. Verify that the per-head
   inputs are actually different (not collapsed to the same tensor).
   This confirms the per-head routing works.

---

## 5. Directory Structure

```
dagformer/
├── CLAUDE.md                  # THIS FILE — the complete spec
├── README.md                  # Brief public description
├── pyproject.toml             # Dependencies
│
├── configs/
│   ├── sanity_check.yaml      # 1K steps, verify baseline NLL reproduction
│   ├── ablation_rank.yaml     # r ∈ {8, 16, 32, 64}
│   ├── ablation_tau.yaml      # τ_init/τ_final sweep
│   ├── ablation_lambda.yaml   # sparsity coefficient sweep
│   └── phase1_full.yaml       # Full Phase 1 run
│
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   ├── olmo_graph.py      # Modified OLMo2 forward with A injection
│   │   ├── predictor.py       # Qwen encoder + MLP decoder + Gumbel + cascade
│   │   └── pipeline.py        # Combines predictor + OLMo into single forward
│   ├── data/
│   │   ├── __init__.py
│   │   └── dolma.py           # Streaming dataloader: produces raw text,
│   │                          # tokenizes with BOTH Qwen and OLMo tokenizers,
│   │                          # returns {qwen_ids, olmo_ids, labels}
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training loop (pure PyTorch + DDP)
│   │   ├── schedulers.py      # τ annealing, λ ramp, LR schedule
│   │   └── checkpointing.py   # Save/load predictor + optimizer + schedule state
│   └── utils/
│       ├── __init__.py
│       ├── logging.py         # Wandb integration
│       └── topology.py        # A matrix analysis utilities
│
├── scripts/
│   ├── train.py               # Entry: python scripts/train.py --config configs/X.yaml
│   ├── eval.py                # Evaluate NLL with/without predictor
│   ├── sanity_check.py        # Verify A=1 reproduces baseline
│   └── visualize_topology.py  # Plot A matrices and gate distributions
│
└── tests/
    ├── test_olmo_graph.py     # Baseline reproduction test
    ├── test_predictor.py      # Shape and gradient tests
    └── test_gumbel.py         # Gumbel-Sigmoid correctness
```

**File responsibilities (be precise):**

- `olmo_graph.py`: ONLY handles injecting A into OLMo's forward. Does NOT
  know about the predictor, Qwen, or training. Exports a function or class
  that takes `(model, olmo_ids, A)` and returns logits.

- `predictor.py`: ONLY the structure predictor. Takes raw text (or
  pre-tokenized Qwen IDs), returns A. Contains Qwen loading, Qwen
  tokenizer, MLP, Gumbel-Sigmoid, cascading gate. Does NOT know about
  OLMo or training.

- `pipeline.py`: Glue. Takes raw text (or pre-tokenized batch dict with
  both qwen_ids and olmo_ids), calls predictor to get A, calls modified
  OLMo forward with A, returns loss. This is what the trainer calls.

- `trainer.py`: Pure training loop. Loads config, builds pipeline, runs
  forward/backward/step. Handles DDP, logging, checkpointing. No model
  logic here.

---

## 6. Implementation Order (FOLLOW THIS EXACTLY)

### Step 0: Project Scaffolding
- Create directory structure above
- Write `pyproject.toml`:
  ```
  dependencies: torch>=2.2, transformers>=4.40, datasets, wandb, pyyaml, einops
  ```
- Verify model loading:
  ```python
  from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
  olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-0425-1B")
  qwen = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
  ```
- Verify Dolma streaming:
  ```python
  from datasets import load_dataset
  ds = load_dataset("allenai/dolma", name="v1_7", split="train", streaming=True)
  print(next(iter(ds)))  # verify a document loads
  ```

### Step 1: `olmo_graph.py` — Modified OLMo Forward
**This is the foundation. Get it right.**

1. Load OLMo2-1B, inspect its architecture:
   - Find the attention layer class name
   - Find where head outputs are computed and merged
   - Find the residual connection logic
2. Implement adjacency injection
3. Run sanity checks (§4.3) — ALL FOUR must pass
4. Do not proceed to Step 2 until Step 1 passes all checks

### Step 2: `predictor.py` — Structure Predictor
1. Implement Qwen encoder wrapper (frozen, mean-pooled)
2. Implement MLP decoder with low-rank heads
3. Implement Gumbel-Sigmoid with 3 modes: (a) training: noise + τ,
   (b) eval soft: σ(Z/τ) no noise, (c) eval hard: (Z>0).float() no noise
4. Implement block-upper-triangular mask (based on layer index, NOT torch.triu)
5. Implement cascading gate
6. Test: output shape [batch, 256, 256], values in [0,1], block-upper-tri
7. Test: `loss = A.sum(); loss.backward()` produces gradients in MLP params

### Step 3: `pipeline.py` — End-to-End
1. Wire predictor output into OLMo modified forward
2. Verify full gradient chain: NLL.backward() updates predictor MLP
3. Profile memory on single GPU (must fit seq_len=1024, batch=1 on 48GB)

### Step 4: Training Infrastructure
1. YAML config → Python dataclass (not raw dicts)
2. `schedulers.py`: τ annealing, λ ramp, LR cosine decay
3. `trainer.py`: training loop
4. `logging.py`: Wandb metrics (see §7)
5. `checkpointing.py`: save/load

### Step 5: Sanity Check Training Run
- Config: `sanity_check.yaml` — 1K steps, batch=4, high τ=5.0
- Verify: loss decreases over 1K steps
- Verify: A is not collapsing (not all-ones, not all-zeros)
- Verify: gradient norms are reasonable
- **STOP HERE if loss doesn't decrease.** Debug before proceeding.

### Step 6: Ablations (later)
- Rank r, τ schedule, sparsity λ, cascading gate on/off
- **Input normalization** (see §6.1 below)

### 6.1 Input Normalization Ablation (IMPORTANT)

When head j receives gated inputs from multiple source heads across
different layers, the magnitudes of these representations can differ
significantly. Layer 0 outputs and layer 14 outputs live at different
scales. The choice of how to normalize the aggregated input to each
head is a critical design decision.

**The problem** — referring to the per-head assembly from §2.2:
```python
gated_sum = Σ_{i: layer(i) < l} A[i,j] * head_output[i]
# If 50 sources have A[i,j] > 0, gated_sum has much larger magnitude
# than any single head_output. Scale depends on sparsity pattern of A.

assembled = base_l + gated_sum  # base_l = embedding + prior MLPs
# The gated_sum component can dwarf or be dwarfed by base_l.
```

**Normalization is applied ONLY to the gated_sum, before adding to base_l:**
```python
assembled = base_l + normalize(gated_sum, method=config.input_norm)
```
This way, base_l (which is the standard, ungated component) is preserved
as-is, and only the novel gated routing is normalized.

**Ablation candidates (implement all, sweep in configs):**

| ID | Method | Formula | Learnable params | Rationale |
|----|--------|---------|-----------------|-----------|
| `none` | Raw weighted sum | `gated_sum` as-is | 0 | Baseline. When A=1 this reproduces vanilla OLMo. |
| `gate_mean` | Divide by gate sum | `gated_sum / (Σ_i A[i,j] + ε)` | 0 | Normalizes for fan-in. ε=1e-8. |
| `rms_post` | RMSNorm after sum | `RMSNorm(gated_sum)` | 2048 (one gain vector) | One shared `nn.RMSNorm(model_dim)` instance, applied to each head's gated_sum. |
| `ln_post` | LayerNorm after sum | `LayerNorm(gated_sum)` | 2×2048 (gain + bias) | One shared `nn.LayerNorm(model_dim)` instance. Affine params are trainable and counted as predictor params. |
| `rms_pre` | RMSNorm each source before sum | `Σ_i A[i,j] * RMSNorm_i(head_output[i])` | 256 × 2048 = 524,288 | One `nn.RMSNorm(model_dim)` **per source node** (256 total). Each head gets its own learnable gain, allowing fine-grained per-head scale correction before mixing. |

All learnable norm params (if any) are part of the predictor's parameter
group and trained with the same LR and optimizer as the MLP decoder.

**Default:** Start with `none` (to verify baseline reproduction), then
switch to `gate_mean` for actual training. If that doesn't work, try
`rms_post`.

**Implementation:** The normalization method is a config string
(`input_norm: "gate_mean"`). The `olmo_graph.py` code dispatches on this
config. All five methods must be implemented.

**Config example for ablation:**
```yaml
# In configs/ablation_norm.yaml
sweep:
  input_norm: ["none", "gate_mean", "rms_post", "ln_post", "rms_pre"]
```

---

## 7. Logging & Monitoring

Log ALL of these to Wandb at every training step:

| Metric | Formula / Source | Why |
|--------|-----------------|-----|
| `train/nll` | Cross-entropy loss | Primary objective |
| `train/sparsity_loss` | λ_t · mean(A) | Regularization term |
| `train/total_loss` | nll + sparsity_loss | What optimizer sees |
| `eval/nll_soft` | NLL with **deterministic** soft gates: A = σ(Z / τ), NO Gumbel noise | Smooth relaxation perf |
| `eval/nll_hard` | NLL with hard gates: A = (Z > 0).float(), NO Gumbel noise | Inference-mode perf |
| `eval/nll_baseline` | NLL with A = all-ones | Should be constant |
| `topology/mean_A` | mean(A) | Overall gate activation |
| `topology/seq_gate_frac` | Fraction of sequential gates > 0.5 | |
| `topology/hyp_gate_frac` | Fraction of hyperconnection gates > 0.5 | |
| `topology/jaccard_var` | Variance of pairwise Jaccard across batch | Context-dependency |
| `schedule/tau` | Current temperature | |
| `schedule/lambda` | Current sparsity coefficient | |
| `grad/predictor_norm` | Total L2 norm of predictor gradients | |

**Collapse alarm:** If `topology/mean_A` < 0.01 or > 0.99 for 100
consecutive steps, log a WARNING. The predictor has degenerated.

---

## 8. Config Format

Use YAML. Example (`configs/sanity_check.yaml`):

```yaml
# Model
olmo_model_id: "allenai/OLMo-2-0425-1B"
qwen_model_id: "Qwen/Qwen3-Embedding-0.6B"

# Predictor
predictor_hidden_dim: 1024
predictor_rank: 32
cascading_gate_k: 5.0
input_norm: "none"  # one of: none, gate_mean, rms_post, ln_post, rms_pre
                    # use "none" for sanity check, "gate_mean" for training

# Data
dataset: "allenai/dolma"
dataset_name: "v1_7"        # Dolma version / subset
seq_len: 1024               # OLMo token count per packed sequence
batch_size: 4
micro_batch_size: 4         # per-GPU micro batch; if < batch_size, use gradient accumulation
qwen_input_prefix: ""       # use raw text directly (see §3.1.1)

# Eval
eval_skip: 1000000          # skip this many examples to reach held-out region
eval_size: 1000             # number of eval sequences (cached in memory)

# Training
total_steps: 1000
lr: 3e-4
weight_decay: 0.01
optimizer: "adamw"  # only adamw supported

# Schedules
tau_init: 5.0
tau_final: 0.2
tau_schedule: "cosine"
lambda_max: 0.0    # no sparsity for sanity check
lambda_warmup_frac: 0.2

# Logging
wandb_project: "dagformer"
wandb_run_name: "sanity-check"
log_every: 10
eval_every: 100

# Checkpointing
save_every: 500
save_dir: "checkpoints/"

# Hardware
num_gpus: 1
```

Parse into a `@dataclass` with validation. Crash on unknown keys.

---

## 9. Key Invariants (ALWAYS enforce)

1. **Baseline reproduction**: A=1 (all 30,720 entries) with `input_norm:
   "none"` → NLL matches vanilla OLMo within 0.01. This validates the
   entire gate injection and per-head splitting logic. Test BEFORE and
   AFTER any architectural change.

2. **DAG constraint**: A is block-upper-triangular based on LAYER indices.
   A[i,j] = 0 whenever `j//16 <= i//16`. Enforced by multiplicative mask,
   never by loss or gradient clipping. Do NOT use `torch.triu()`.

3. **Gradient flow**: After every forward-backward, assert that all
   predictor parameters have non-None, non-zero gradients.

4. **Memory budget**: Must fit on 4×A40 for seq_len=1024. If OOM, reduce
   batch size. Do NOT change the architecture to fix memory.

5. **Frozen models stay frozen**: OLMo and Qwen must have
   `requires_grad=False` on ALL parameters. Verify this at startup.
   In Phase 1, the only trainable parameters are the MLP decoder.

6. **Deterministic eval**: Eval uses NO Gumbel noise, ever. Two eval modes:
   (a) Soft: A = σ(Z/τ), continuous [0,1]. (b) Hard: A = (Z>0).float(),
   binary {0,1}, with hard cascading gate (g_j = 1 if inc_j>0, else 0).
   Always report both `eval/nll_soft` and `eval/nll_hard`.

---

## 10. Oracle Search Reference Numbers

These are the results from the completed oracle search. Use them to
validate that the learned predictor is heading in the right direction.

| Metric | Value |
|--------|-------|
| Windows evaluated | 50 |
| Window size | 1024 tokens |
| Improvement rate | 100% (all 50 improved) |
| Baseline NLL (median) | 2.58 |
| Oracle NLL (median) | 0.12 |
| NLL delta (median) | +2.38 |
| Oracle sequential gates ON | ~91% |
| Oracle hyperconnection gates ON | ~70% |
| Oracle search steps per window | 500 |
| Oracle search method | Surrogate gradient (STE) |

The learned predictor does NOT need to match oracle performance. The
**decision gate** for Phase 1 success is: predictor NLL ≤ dense baseline
NLL (i.e., the predictor must not make things WORSE).

---

## 11. Dependencies (exact)

```toml
[project]
name = "dagformer"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "transformers>=4.40",
    "datasets",
    "wandb",
    "pyyaml",
    "einops",
]
```

**NOT used (by design):**
- No HuggingFace Accelerate
- No PyTorch Lightning
- No DeepSpeed
- No custom CUDA kernels

Multi-GPU via `torch.nn.parallel.DistributedDataParallel` only.

---

## 12. What NOT to Do

- **Do NOT implement Phase 2** (joint OLMo training). Design the code to
  support it (param groups, differential LR) but do not implement unfreezing.
- **Do NOT implement a diffusion-based predictor.** The MLP decoder is
  the current design. Diffusion is future work.
- **Do NOT write custom CUDA kernels.** Use dense matmuls with masking.
- **Do NOT support other base models.** Hardcode OLMo2-1B for now.
- **Do NOT use Accelerate or Lightning.** Pure PyTorch.
- **Do NOT run hyperparameter search.** Manual ablations only.
- **Do NOT fork OLMo2's source code.** Load from HF and modify via
  hooks, monkey-patching, or subclassing.
- **Do NOT use `nn.DataParallel`.** Use `DistributedDataParallel` only.

---

## 13. Code Style

- Type hints on all function signatures
- Docstrings on all public functions and classes
- Config as `@dataclass`, not raw dicts
- `assert` for shape checks in every forward pass (e.g.,
  `assert A.shape == (batch, 256, 256)`)
- No silent failures — crash loudly with informative messages
- Prefer explicit loops over clever one-liners when clarity matters
- One class per file is fine; don't over-split
- Use `einops.rearrange` for complex tensor reshaping (clearer than
  chains of `.view().permute().contiguous()`)

---

## 14. Quick Reference — Model IDs and Shapes

| Thing | Value |
|-------|-------|
| OLMo model ID | `allenai/OLMo-2-0425-1B` |
| Qwen model ID | `Qwen/Qwen3-Embedding-0.6B` |
| OLMo layers | 16 |
| OLMo heads per layer | 16 |
| Total nodes | 256 |
| A matrix shape | `[batch, 256, 256]` |
| A constraint | Block-upper-triangular: `A[i,j]=0` when `j//16 <= i//16` |
| A free entries | 30,720 (cross-layer only) |
| Predictor rank r | 32 (default) |
| Predictor hidden dim | 1024 (default) |
| Sequence length | 1024 |
| Gumbel-Sigmoid τ range | 5.0 → 0.2 |
| Cascading gate k | 5.0 |
| Input normalization | `none` (sanity check), `gate_mean` (training default), ablate all 5 |
| Sparsity λ range | 0 → 0.01 |
