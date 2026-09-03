# Methodology: A FLOPs × Loss Pareto Frontier for DAGFormer

**Purpose.** Ground a publishable "quality-per-compute" claim: DAGFormer (OLMo-2 with
per-head, per-token 4-stream Q/K/V/R dynamic routing across layers) achieves lower loss
at equal compute than (a) our own dense OLMo-2 baselines and (b) comparable methods
(MUDDFormer, Hyper-Connections, DenseFormer). This doc gives (1) rigorous FLOPs
accounting that does **not** undercount DAGFormer's own cost, (2) a tokenizer- and
data-agnostic loss metric (bits-per-byte), (3) how prior work draws these frontiers, and
(4) a concrete plot spec.

All claims are grounded in primary sources with arXiv IDs. Key references:
Kaplan et al. 2020 (**2001.08361**), Hoffmann et al. / Chinchilla (**2203.15556**),
The Pile (**2101.00027**), Paloma (**2312.10523**), OLMo (**2402.00838**),
Cerebras-GPT (**2304.03208**), Pythia (**2304.01373**), MUDDFormer (**2502.12170**),
Hyper-Connections (**2409.19606**), DenseFormer (**2402.02622**), Epoch Chinchilla
replication (**2404.10102**), the "MISFITTING" scaling-law-reporting survey (**2502.18969**).

---

## 1. FLOPs accounting

### 1.1 The 6·N·D training approximation — derivation and validity

**Statement.** Total training compute `C ≈ 6·N·D`, where `N` = model parameters,
`D` = training tokens. Per training token the cost is `≈ 6N` FLOPs.

**Derivation (the "6 = 2 × 3").**
- **Factor 2 — multiply-accumulate.** Each parameter participates in one multiply + one
  add per token in a matmul, so a **forward** pass costs `≈ 2N` FLOPs/token
  (Kaplan Eq. 2.2, `C_forward ≈ 2N + context term`; Chinchilla App. F: "a factor of 2 to
  describe the multiply accumulate cost").
- **Factor 3 — backward is ~2× forward.** Both papers assume `backward ≈ 2 × forward`
  (Chinchilla App. F: "we assume that the backward pass has twice the FLOPs of the forward
  pass"). So forward `2N` + backward `4N` = **`6N` per token**; over `D` tokens,
  `C ≈ 6ND`. (The equivalent "three matmul passes" framing: forward `2N`,
  grad-wrt-inputs `2N`, grad-wrt-weights `2N` = `6N`.)

**What N includes.** ⚠ The two founding papers differ:
- **Kaplan** uses **non-embedding** parameters: `N ≈ 12·n_layer·d_model²`
  (with `d_attn = d_ff/4 = d_model`), **excluding** token + positional embeddings
  ("we do not include these ... this produces significantly cleaner scaling laws").
- **Chinchilla** **includes** embeddings in both `N` and the FLOP count ("we include all
  training FLOPs, including those contributed to by the embedding matrices ... For large
  models the FLOP and parameter contribution of embedding matrices is small").

  **→ Our choice: report `N` = non-embedding parameters (Kaplan convention)** and state it
  explicitly. At 75M–600M the embedding matrix (vocab≈50k × d_model) is a *large* fraction
  of params, so mixing conventions across scales would distort the small-model points.
  Using non-embedding `N` consistently keeps the frontier clean. (Cite both; note the
  convention.)

**When 6ND is valid — Chinchilla's own error table (Table A4).** Chinchilla compares the
detailed per-operation count to `6ND`; the ratio (detailed / 6ND) across their models:

| Params | 73M | 305M | 552M | 1.1B | 1.6B | 6.8B |
|--------|-----|------|------|------|------|------|
| detailed / 6ND | 1.03 | 1.10 | 1.08 | 1.04 | 1.03 | 0.99 |

So 6ND is within **±10%** and converges to ≈1.0 as models grow ("the differences ... are
very small and do not impact our analysis"). 6ND slightly **under**-counts at small scale
(embedding + attention overheads are relatively larger there).

### 1.2 The attention (context) term — when it stops being negligible

Kaplan's forward pass keeps a context-dependent term (Table 1, "Attention: Mask" row):
```
attention FLOPs / token = 2 · n_layer · n_ctx · d_attn        (linear in T per token)
sequence-level attention FLOPs ∝ n_layer · n_ctx² · d          (the familiar O(T²))
```
Ratio of this term to the parameter term `2N ≈ 24·n_layer·d_model²`:
```
(2·L·T·d) / (24·L·d²) = T / (12·d_model)
```
So attention is **negligible when `d_model > n_ctx / 12`** (Kaplan's exact stated
condition: "For contexts and models with d_model > n_ctx/12, the context-dependent
computational cost per token is a relatively small fraction of the total compute").

**Our regime check.** `T = seq_len = 1024`, so the threshold is `d_model > 85`. Our
smallest model has `d_model = 1024` (75M–150M) up to `2048` (1B):
`T/(12·d) = 1024/12288 ≈ 8.3%` at the 75M scale, `≈ 4.2%` at 1B. **The attention term is
1–8% of 2N** — non-negligible at the small end, and (crucially) it is the **same** for
DAGFormer and the dense baseline of equal `d_model`. We therefore (a) report `6ND` as the
primary X, and (b) **additionally compute the full per-operation FLOPs (incl. the T² term)
for both DAGFormer and baseline** and confirm the frontier ordering is unchanged — this
forecloses a reviewer objection that "6ND hides the attention cost." (MISFITTING,
2502.18969, found many papers never even state how FLOPs were counted; we state the exact
formula, §1.4.)

### 1.3 Training FLOPs vs inference/forward FLOPs — which goes on the X-axis

- **Training FLOPs/token ≈ 6N** (`C = 6ND`). This is what Chinchilla / Cerebras-GPT put
  on X in "loss vs pre-training FLOPs" frontiers. It answers **"how good is the model I get
  for a training budget"** — the standard scaling-law question.
- **Inference/forward FLOPs/token ≈ 2N** (+ attention term). No backward pass, so exactly
  **one-third** of the training per-token cost. This answers **"how good is the model per
  unit of *serving* compute."**

**Recommendation: use TRAINING FLOPs (`6ND`) as the primary X-axis**, because
(i) it is the axis of every canonical scaling-law frontier (Kaplan Fig 1, Chinchilla Fig 1–2,
Cerebras Fig 1–2), making our figure directly comparable to and overlayable with published
external anchors, and (ii) the training-vs-inference ratio is a fixed 3× for dense models,
so the two axes are affine-related for the baseline and the choice does not change the
*ranking*. **BUT** DAGFormer adds a persistent per-token routing cost that is present at
inference too, so we **also produce an inference-FLOPs/token version** of the figure
(Cerebras-GPT does exactly this in their Fig 6, folding inference FLOPs into X). The
honest headline for an architecture with always-on routing is arguably the total-compute
view; showing both training-FLOPs and inference-FLOPs panels pre-empts the "you only win on
paper training FLOPs" critique.

### 1.4 Counting DAGFormer's EXTRA FLOPs honestly (the core fairness issue)

**Why this matters.** The single most common way "we-beat-the-frontier" architecture papers
mislead is **undercounting their own added compute** (MISFITTING 2502.18969; the MoE FLOP
literature 2501.12370). Comparable methods do this: **MUDDFormer (2502.12170) quotes "0.4%
FLOPs overhead" but its own Table 4 shows ~16% wall-clock slowdown** (a ~40× gap, because
the dynamic dense aggregation is memory-bandwidth-bound, not FLOP-bound). **Hyper-Connections
(2409.19606) quotes no number at all** ("almost negligible") despite an `n×` (typically 4×)
wider residual stream. We must not repeat this. We charge DAGFormer every FLOP.

**DAGFormer's added computation (from `src/model/olmo_graph.py::FourWayDAGFormer`).** Per
token, on top of the dense `2N` forward, DAGFormer adds two things:

**(A) The routing/mixing term — an O(L²·H·T·d_head) contribution.**
At each layer `l` (l = 1 … L−1), the model mixes the `l` prior layer outputs (plus
embedding, so `n_src = l+1`) into each of the `H` head inputs, separately for the Q, K, V
streams, in **head_dim space** (the einsum `'lbthd,bthl->bhtd'`), plus an R (residual)
stream mixed in model_dim. For one token:
```
mix FLOPs(l) ≈ 2 · n_src · H · d_head · 3      (Q,K,V head-space mixing, MACs)
             + 2 · n_src · d_model              (R stream, model-dim mixing)
with  H·d_head = d_model, so ≈ 2·n_src·(3·d_model + d_model) = 8·n_src·d_model
Summing l = 1…L−1  (Σ n_src = Σ(l+1) ≈ L²/2):
```
```
ROUTING_MIX_FLOPs / token  ≈  8 · d_model · Σ_{l} (l+1)  ≈  4 · L² · d_model
```
This is the promised **O(L² · d_model)** per-token term (equivalently O(L²·H·d_head)).
It is quadratic in **depth** (not sequence length): every layer reads all shallower layers.

**(B) The routing-weight predictor network — a small MLP per layer.**
For each layer `l`, a generator MLP maps the current hidden state `[·, d_model]` →
per-token Q/K/V/R weights. In the FourWay implementation the generator trunk is
`Linear(d_model, h_route) → GELU → Linear(h_route, out_dim)` with
`out_dim = 3·H·n_src + n_src` and `h_route = fourway_hidden` (512 in our configs).
Per token:
```
PREDICTOR_FLOPs(l) ≈ 2 · d_model · h_route  +  2 · h_route · (3·H·(l+1) + (l+1))
Summing over l:
```
```
PREDICTOR_FLOPs / token  ≈  2·(L−1)·d_model·h_route  +  2·h_route·(3H+1)·Σ(l+1)
                         ≈  2·L·d_model·h_route  +  h_route·(3H+1)·L²
```
(If a Qwen-encoder predictor is used instead — see `predictor.py::QwenEncoder` — its
forward is a full `≈ 2·N_qwen` per token and must be added; but the deployed FourWay
variant uses the lightweight per-layer generator above, which is far cheaper. **State which
predictor the reported model uses** and charge its true cost.)

**Effective training FLOPs/token for DAGFormer:**
```
C_dag / token ≈ 3 × [ 2·N_dag_params            # forward params (incl. generator params in N_dag)
                    + 4·L²·d_model               # routing mix (A)
                    + attention context term ]   # 2·L·T·d, same form as dense
             (the ×3 = fwd + 2×bwd; note the generator MLP params ARE trainable, so they
              take the full 6× like any trainable param; the frozen-Qwen path, if used,
              takes only 2× forward with no backward.)
```
```
C_dense / token ≈ 3 × [ 2·N_dense + attention context term ]
```
Because DAGFormer's extra params (the per-layer generators) are already **inside**
`N_dag`, the clean way to write it is: **`C_dag = 6·N_dag·D + 6·(4L²·d_model)·D_train`**
for the trainable-generator case, i.e. just apply `6ND` to the *true* parameter count
`N_dag` **and add the routing-mix term** (which is activation FLOPs, not parameter FLOPs,
so it is not captured by `6·N`). Equivalently, define an **effective parameter count**:
```
N_eff(DAGFormer) = N_dag_params + (routing-mix amortized)  ,  and plot 6·N_eff·D.
```
In practice we compute the **full analytical per-token FLOPs** (params + routing-mix +
predictor + attention) with a small script and multiply by `3·D` — no hand-waving.

**Sanity magnitudes (1B config: L=16, d_model=2048, H=16, h_route=512, T=1024).**
- routing-mix `4·L²·d_model = 4·256·2048 ≈ 2.1M FLOPs/token` vs dense forward
  `2N ≈ 2·1.28e9 ≈ 2.6e9` → **~0.08%**. Tiny in *FLOPs*, exactly as MUDDFormer found.
- predictor `≈ 2·L·d_model·h_route + h_route·(3H+1)·L² ≈ 2·16·2048·512 + 512·49·256
  ≈ 3.4e7 + 6.4e6 ≈ 4e7 FLOPs/token` → **~1.5%** of `2N`. The generator MLP dominates the
  overhead, not the mixing.

  ⚠ **The FLOP overhead (~1.5%) understates the real cost** — exactly the MUDDFormer trap.
  We therefore **also report measured wall-clock throughput and, ideally, measured FLOPs**
  (e.g. via `torch.profiler` / DeepSpeed FLOPs profiler) in an appendix, and **use the
  larger of {analytical, measured-effective} on the X-axis** so we never flatter ourselves.
  DenseFormer (2402.02622) is the model to emulate here: it prices its overhead in
  **measured batches-per-second** and includes a **matched-throughput** baseline.

**Summary formula to put in the paper:**
```
FLOPs_per_token(dense)     = 6N + 6·(2·L·T·d)                       [train]  ;  /3 for inference
FLOPs_per_token(DAGFormer) = 6·N_dag + 6·(4·L²·d + P) + 6·(2·L·T·d) [train]  ;  /3 for inference
    where N_dag includes the routing-generator params,
          4·L²·d = per-token routing-mix activation FLOPs,
          P      = per-token predictor-MLP FLOPs (or 2·N_qwen if a frozen encoder is used).
```

---

## 2. Making loss comparable across models (different data, tokenizers)

### 2.1 Why raw perplexity is NOT comparable

A model outputs a distribution over its **token** vocabulary; perplexity is per-token.
Two models that fit the same *text* equally well but tokenize differently report different
per-token loss purely because they cut the text into different numbers of tokens. Paloma
(2312.10523) states it exactly: *"Perplexity per token is not comparable between models with
different vocabularies (Jelinek 1998) or, by extension, different tokenizers (Mielke 2019).
Since models distribute probability over a vocabulary of tokens, models with larger
vocabularies will tend to have higher perplexities than ones with smaller vocabularies."*
Our models, MUDDFormer/Pythia (GPT-NeoX-20B tok), Cerebras (GPT-2 tok) all differ → raw PPL
is unusable across them.

### 2.2 Bits-per-byte (BPB) — the fix

Normalize the **total** negative log-likelihood (nats) by the number of **UTF-8 bytes** of
the underlying text (tokenizer-invariant), converting nats→bits via `ln 2`:
```
BPB = total_NLL_nats / (ln 2 · total_UTF8_bytes)
    = (mean_per_token_loss_nats · n_tokens) / (ln 2 · n_UTF8_bytes)
    = (n_tokens / n_UTF8_bytes) · log2(perplexity)
```
The byte count is a fixed property of the text; the ratio `n_tokens / n_bytes` is the
tokenizer's fertility, which exactly cancels the tokenizer's effect on per-token loss.
This is **the** cross-model LM metric, defined canonically in **The Pile (2101.00027 §3.1)**:
*"Our preferred metric is bits per UTF-8 encoded byte (BPB). Bits per byte is preferred over
bits per character or perplexity ... due to its invariance to different tokenization schemes
and the ambiguity of measuring characters in Unicode ... BPB = (L_T/L_B)·log2(e^ℓ)"* (they
report L_T/L_B = 0.29335 GPT-2-tokens/byte on the Pile). `byte_perplexity = 2^BPB`;
`BPC` (bits-per-character) is the same with characters in the denominator (= BPB for ASCII).

**EleutherAI lm-evaluation-harness** implements this in `lm_eval/api/metrics.py`:
`bits_per_byte = -weighted_mean(items)/ln2` where each task's `process_results` pairs the
document's rolling loglikelihood (summed per-token log-probs, nats) with the document's UTF-8
byte count computed on the **original detokenized text** (e.g.
`_bytes = len(doc["page"].encode("utf-8"))`), summed across documents before the ratio. It
is the default metric bundle for `loglikelihood_rolling` tasks
(`["word_perplexity","byte_perplexity","bits_per_byte"]`). **Use this harness so our numbers
are directly comparable to everyone else's.**

**Alternative (Cerebras-GPT style).** If we must compare raw cross-entropy across vocabs,
rescale nats/token by the tokens/reference-unit ratio to a reference tokenizer
("we correct all cross-entropy results for different vocabularies to be comparable to the
GPT-2 vocabulary", 2304.03208). This is algebraically equivalent to BPB; BPB is cleaner.

### 2.3 What common corpus

The field converges on **evaluate on a shared held-out corpus, report BPB, per-document
(not concatenated), decontaminated**:
- **Primary: Paloma (2312.10523)** — 585 domains / 18 sources, designed for cross-model fit;
  reports **BPB (its Eq. 3)**; recommends fixing the tokenizer when it is not the variable and
  evaluating documents individually. **OLMo (2402.00838) adopts Paloma and reports
  "bits per byte as defined by Gao et al. (2020)"** and decontaminates training data against
  Paloma. Since our base is OLMo-2, Paloma is the natural, defensible primary.
- **Secondary anchors: the Pile validation/test set** in BPB (lets us overlay MUDDFormer,
  Pythia, Cerebras which report on the Pile) and a **held-out C4 slice** (one Paloma source;
  cheap, single-distribution web text).
- **In-distribution: a held-out Dolma / OLMo-mix slice** in BPB for the cleanest
  *within-study* DAGFormer-vs-our-dense comparison (identical data + tokenizer → BPB and
  nats/token agree, and the comparison is confounder-free).

**Decontamination.** Following Paloma/OLMo, remove any training document overlapping the eval
sets (paragraph-level leak check). State this.

---

## 3. How prior work plots these frontiers

**Kaplan 2020 (2001.08361) Fig 1** — 3 panels, **all log-log**, shared Y = test loss (nats):
X = compute (PF-days) / dataset (tokens) / non-embedding params. Power laws are straight
lines; `L(C_min) = (3.1e8 / C_min)^0.050` on the compute panel (⚠ the constant is
3.1×10⁸ PF-days, not the often-misquoted 2.3×10⁸). Framing: "smooth power law." Note Kaplan's
`N ∝ C^0.73` compute-optimal scaling was later **overturned by Chinchilla** — cite Kaplan for
the FLOP accounting and the log-log frontier idea, not for the exponent.

**Chinchilla 2022 (2203.15556)** — three constructions of the compute-optimal frontier that
all agree (`N ∝ C^0.5`, `D ∝ C^0.5`, ~20 tokens/param):
- **Fig 1 "overlaid predictions":** loss vs training FLOPs (log-log), predicted frontier +
  real model points (Gopher, GPT-3, MT-NLG).
- **Fig 2 Approach 1 — training-curve envelope (the literal Pareto construction we want):**
  each model 70M–10B traces its own **loss-vs-FLOPs curve as it trains**; the frontier is the
  **lower envelope = min loss over all model curves at each FLOP budget**. Read off which
  model achieves each envelope point → optimal `N(C)`, `D(C)`.
- **Fig 3 Approach 2 — IsoFLOP:** loss vs **parameters** (log-x) at fixed FLOP budgets → a
  **U-shaped valley** per budget whose minimum = compute-optimal `N`; fit a parabola, connect
  minima.
- **Fig 4 Approach 3 — parametric fit** `L(N,D)=E+A/N^α+B/D^β` (E=1.69, A=406.4, B=410.7,
  α=0.34, β=0.28); frontier = line through each iso-loss contour "at the point with the fewest
  FLOPs."

**Cerebras-GPT (2304.03208) Fig 1–2 (the template most like ours):** X = pre-training FLOPs
(log), Y = **Pile test loss, nats/token, vocab-corrected to GPT-2** (log). Fitted offset power
law `L(f) = (f/5.984e22)^-0.0737 + 0.5066`. Their models (111M–13B, all at ~20 tok/param) form
the **lower-left frontier**; over-trained Pythia/OPT sit **above** it. They also show
percent-loss-from-frontier plots, downstream accuracy vs FLOPs, and (Fig 6) **loss vs
train+inference FLOPs** — the inference-inclusive view we should mirror.

**Pythia (2304.01373):** ⚠ **does NOT present a loss-vs-FLOPs compute-optimal frontier** — it
is a controlled suite for training-dynamics analysis; its scaling-type plots are downstream
**accuracy vs parameters** (vs OPT/BLOOM). **Cite Pythia as an external model *suite* to
anchor points, not as a frontier method** (miscitation risk).

**What makes a convincing "below the frontier" figure** (synthesizing the above +
Epoch replication 2404.10102 + MISFITTING 2502.18969):
1. **Strong, compute-optimal baseline** (the #1 failure mode: beating an *undertrained/untuned*
   dense baseline is worthless — the Kaplan→Chinchilla lesson). Each dense run's LR must decay
   to end **for its own token budget**; tune batch/wd/warmup per scale; state it.
2. **Log-log axes**, Y in **BPB** (portable across tokenizers), same eval corpus for all points.
3. **A real frontier line, not two dots** — ≥5–8 well-spread points; draw it as a **fitted
   offset power law** `L = E + A·C^-α` and/or a **training-curve lower envelope**; show a
   **bootstrap uncertainty band** and **do not extrapolate** past the largest fitted point.
4. **Charge your method its true FLOPs on X** (§1.4).
5. **Prove dominance as a frontier *shift*, not one crossing:** overlay DAGFormer's own fitted
   frontier below the dense frontier with **non-overlapping bands**, and show both
   **iso-compute** (lower Y at fixed X, at several budgets) and **iso-loss** (fewer FLOPs at
   fixed Y). "Pareto-dominant" = below-and-left: lower-or-equal loss at lower-or-equal compute,
   strictly better on one. If dominance holds only on a sub-range, say so and mark the crossing.

---

## 4. Concrete recipe for OUR figure

### 4.1 Axes
- **Main figure (Fig A) — X = training FLOPs `C = 6·N_eff·D` (log), Y = BPB on common corpus
  (log or linear-in-BPB; BPB is already log-domain so linear-Y is standard).** `N_eff`
  includes DAGFormer's routing params; the routing-mix + predictor activation FLOPs are added
  analytically (§1.4). This mirrors Cerebras Fig 1–2 → overlayable with published anchors.
- **Companion (Fig B) — X = inference FLOPs/token `≈ 2·N + routing + predictor` (log), same Y.**
  Shows the serving-compute view (Cerebras Fig 6 style); the honest view for always-on routing.
- Optionally a **capacity panel** X = non-embedding params — but **label it "capacity," not
  compute** (params is a misleading compute axis for a routed model; always keep FLOPs as the
  compute axis).

### 4.2 Series to plot
1. **Our dense OLMo-2 baseline** at 75M / 150M / 300M / 600M / 1B → **fit these into the
   reference frontier** (offset power law `L = E + A·C^-α`; draw line + bootstrap band). These
   five well-spread points across ~1.5 orders of magnitude of FLOPs make a legitimate frontier.
2. **Our DAGFormer** at the same 75M / 150M / 300M / 600M / 1B, X-position using the **true**
   DAGFormer FLOPs (§1.4). Fit its **own** frontier line + band. The headline = this line lies
   below the dense line with non-overlapping bands.
3. **External open anchors** (distinct markers, "as reported / re-evaluated"): **Pythia**
   70M–1.4B, **Cerebras-GPT** 111M–590M, **MUDDFormer / MUDDPythia**, **DenseFormer**. Plot at
   their true FLOPs, Y in BPB on the **same** corpus (re-evaluate their public checkpoints with
   lm-eval-harness where possible — Cerebras-GPT and OLMo both re-ran evals themselves rather
   than trusting published numbers; we should too). These calibrate that our dense frontier is
   competitive and that DAGFormer beats MUDDFormer/DenseFormer specifically.

### 4.3 Drawing the frontier line
- Fit `L(C) = E + A·C^-α` (offset power law, Cerebras form) separately to the dense points and
  the DAGFormer points via robust (Huber) least squares in log-space; report E, A, α with CIs.
- **And/or** build the **Chinchilla Approach-1 training-curve envelope**: log each run's
  BPB-vs-FLOPs curve during training, take the lower envelope. This is the most
  assumption-light frontier and lets us show the DAGFormer envelope sitting under the dense
  envelope throughout.
- **Bootstrap** the fit (resample runs/seeds); shade the 95% band; **clip at the largest
  measured FLOP** — no extrapolation into a region only DAGFormer occupies.

### 4.4 Fairness caveats to disclose (every one)
1. **FLOPs accounting stated explicitly** with the exact formula (§1.4), including DAGFormer's
   routing-mix `O(L²·d)` + predictor-MLP terms. **Report measured wall-clock / measured FLOPs
   too**, and use the larger of analytical vs measured on X (don't repeat MUDDFormer's ~40×
   FLOPs-vs-wallclock gap silently). `N` convention (non-embedding) stated.
2. **Loss comparability:** BPB on a shared, decontaminated corpus, per-document, same
   inference code (lm-eval-harness). Note that external anchors were trained on different
   data/tokenizers — BPB controls the tokenizer but **not** the data; hence:
3. **The clean within-study comparison is DAGFormer vs our dense at matched data + tokenizer +
   token budget** (both on the same Dolma/OLMo-mix, same OLMo-2 tokenizer). Present this as the
   *primary, confounder-free* result; external anchors are *context*, not the core claim.
4. **Baseline strength:** state that each dense baseline is compute-optimal / tuned (LR decays
   to end for its own budget), so we are not beating a crippled baseline.
5. **Seeds / error bars** on both Y (BPB) and the frontier fit; ≥2–3 seeds where feasible.
6. **Attention term:** confirm the `T²` term (excluded by 6ND) does not change ordering
   (it is identical for DAGFormer and dense of equal d_model; §1.2).
7. **Range of dominance:** state the FLOP range over which DAGFormer's frontier is
   provably below the dense frontier (non-overlapping bands); mark any crossing.

---

## Appendix — matplotlib plot spec

```
FIGURE A (primary):  BPB vs Training FLOPs
------------------------------------------------------------------
ax.set_xscale('log'); ax.set_yscale('log')     # log-log (Chinchilla/Cerebras convention)
ax.set_xlabel('Training FLOPs  (C = 6·N_eff·D)')
ax.set_ylabel('Bits per byte  (Paloma / common corpus, ↓ better)')

SERIES:
 1. dense_points     : scatter, filled circles, 5 pts (75M,150M,300M,600M,1B)
                        x = 6·N_nonembed·D ; y = BPB ; yerr = seed std
 2. dense_frontier   : line, fit L=E+A·C^-α to dense_points ; +/- shaded 95% bootstrap band
 3. dagformer_points : scatter, filled squares, 5 pts, SAME scales
                        x = TRUE dagformer FLOPs (params+routing_mix+predictor) ; y = BPB ; yerr
 4. dag_frontier     : line (distinct color), fit to dagformer_points ; shaded band
 5. anchors_ext      : scatter, distinct hollow markers per family
                        {Pythia, Cerebras-GPT, MUDDFormer, DenseFormer}, re-evaluated to same BPB
GUIDES:
 - one vertical dashed line at a chosen iso-compute C0 -> annotate ΔBPB (dense vs DAGFormer)
 - one horizontal dashed line at a chosen iso-BPB      -> annotate ΔFLOPs (× compute saving)
 - clip both frontier lines at max measured FLOP (no extrapolation)
 - legend distinguishes "ours (fit frontier)" vs "external anchors (re-evaluated)"

FIGURE B (companion):  BPB vs Inference FLOPs/token   (x = 2N + routing + predictor; else identical)
FIGURE C (appendix, optional):  training-curve lower-envelope (BPB vs FLOPs, per-run curves + envelope)
FIGURE D (appendix):  measured wall-clock throughput (tokens/s) vs scale, DAGFormer vs dense
                       + a matched-throughput dense baseline (DenseFormer-style honesty check)
```

**Fit function to reuse for both frontiers** (offset power law, Cerebras form
`L(f)=(f/f0)^-α + E`): robust (Huber) least-squares in log-space on `(C_i, BPB_i)`; bootstrap
over runs×seeds for the band; return `(E, f0, α)` + CIs; clip drawn line to `[min C_i, max C_i]`.

---

### One-line takeaways
- **X-axis:** training FLOPs `6·N_eff·D` (primary) + inference FLOPs/token (companion), log.
  Charge DAGFormer its **true** cost: `6·N_dag + 6·(4L²·d + predictor) + attention`; report
  **measured** cost too and use the larger. (Avoid MUDDFormer's 0.4%-quoted / 16%-real gap.)
- **Y-axis:** **bits-per-byte** on a shared, decontaminated corpus (**Paloma**, + Pile/C4/Dolma
  slices), per-document, via lm-eval-harness. Raw PPL is not cross-tokenizer comparable.
- **Frontier:** fit `L=E+A·C^-α` (+ Chinchilla Approach-1 envelope) to our **5-point dense**
  baseline; overlay our **5-point DAGFormer**; prove a **frontier shift** (non-overlapping
  bands, iso-compute AND iso-loss). External Pythia/Cerebras/MUDDFormer/DenseFormer as anchors.
- **Cleanest claim:** DAGFormer vs our dense at **matched data + tokenizer + tokens** — the
  confounder-free within-study comparison; everything else is context.
