# Behavioural circuits in the routing graph

The [September 17 audit](../PROJECT_SYNC_2026-09-17.md) reconciles these results
with the author's direct head-edit experiments. This suite tests
instruction-derived editing directions on hyperconnections; its negative
results do not cover every editing direction, sequential path or behavior.

Finding the set of cross-layer hyperconnections that *causes* a behaviour, by
contrasting the routing weights the structure predictor emits under opposed
instructions and then intervening on the edges that differ.

The premise is that DAGFormer exposes something a deployed transformer does
not: an explicit, editable topology. If a behaviour lives in a small set of
routing edges, you can read it off the predictor and steer it there — before
any hidden state exists.

```
extract_contrast.py   run the same content under pos / neg / neutral
                      instructions, record alpha over the aligned span
        |
discover_circuit.py   Delta = alpha_pos - alpha_neg, decide which coordinates
                      are really non-zero, assemble the graph
        |
verify_circuit.py     intervene on those coordinates and see whether the
                      behaviour moves — against controls that say what a
                      non-circuit would have done
        |
plot_circuit.py       optional figures
```

## The measurement

### Coordinates

A routing edge is `(stream, target layer l, target head h, source s)`:

* `stream` ∈ `{q, k, v, r}` — the per-head query/key/value mixes and the shared
  residual mix.
* `s = 0` is the embedding, `s ≥ 1` is the output of layer `s-1`.
* `s == l` is the **sequential** path (bias-initialised to 1.0).
* `s < l` is a **hyperconnection** — a skip. Pass `--hyper-only` to restrict
  discovery to these.

Everything is addressed in one flat `[D]` vector laid out as
`for l in 1..L-1: [q(H·n_src), k(H·n_src), v(H·n_src), r(n_src)]`, which is
both `scripts/interp_common.py`'s canonical order and the output layout of
`FourWayDAGFormer.correction_mlps[l-1]`. For the 300M model (L=12, H=16),
D = 3773, of which 3234 are hyperconnections and 539 are sequential paths.

### Two editable channels

| channel | what it is | why you would use it |
|---|---|---|
| `pred` | `FourWayPredictor(input_ids)` — an external 2-layer causal encoder | the interpretability target: a pure function of the token context, readable and editable *before* the model runs |
| `corr` | per-layer `correction_mlps` inside the model, reading hidden states | where content-dependent routing adjustments live |
| `eff`  | `pred + corr`, discovery only | the weights the model actually mixes with |

`alpha` is **not** softmaxed — it is a raw linear mixing coefficient
(`einsum('lbthd,bthl->bhtd')`). So `alpha *= (1+λ)` really does enlarge a
connection, which is what makes the amplification test meaningful.

One consequence to keep in mind when reading results: `q` and `k` pass through
`q_norm` / `k_norm` after mixing, so uniform magnitude changes there are partly
renormalised away. `v` and `r` are not normalised in this config
(`use_v_norm` defaults off), so they carry raw magnitude effects.
`verify_circuit.py` reports a per-stream breakdown for exactly this reason.

### Position alignment

The predictor is position-aware, so two instructions of different token length
would put the content at different absolute positions and the contrast would be
confounded by a position shift. `build_prompt_set` therefore

* tokenises the instruction and the content **separately** and concatenates, so
  the content token ids are bit-identical across polarities, and
* front-pads every instruction prefix with newline filler to a common length.

Every prompt in a set shares one content span. Right-padding is safe with no
attention mask because the predictor (`predictor_causal: true`) and the model
(`is_causal=True`) are both strictly causal.

## Deciding what counts as non-zero

The proposal is that `Delta` is sparse. That is a hypothesis, not a licence to
pick a threshold — so `discover_circuit.py` measures it.

**The default rule is `--select null`, and significance testing is not it.**
On this model, `Delta` has almost no item-level variance: the predictor is a
small causal encoder and its routing output is very nearly a deterministic
function of the instruction, so `corr(Delta_train, Delta_test)` across disjoint
halves of the items comes out at 1.000 and sign agreement is 1.000 even on
*randomly chosen* edges. Under that variance structure a per-item significance
test answers "is this edge's difference non-zero?" with "yes" for essentially
every edge — measured: BH-FDR at q=0.05 selected 98.6% of eligible edges, and
selected just as many from a null contrast. The question worth asking is the
other one:

> is this edge's difference bigger than the difference the same averaging
> produces when there is no behavioural contrast at all?

1. **Balanced-split null, used as the calibration set** (`--select null`). The
   pos and neg paraphrases are re-partitioned into two halves that each hold
   the same polarity mix, then differenced. The behavioural contrast cancels
   *exactly* while averaging depth, paraphrase diversity and item count all
   match the real contrast. Three disjoint sets of draws are generated: one
   estimates a per-coordinate null scale `sigma` (shrunk toward the bulk so a
   lucky near-zero cannot dominate), one calibrates the threshold on
   `S = |Delta| / sigma` at the `1-q` quantile, and one is held back to report
   how many edges the finished rule picks from a signal-free contrast. That
   third number, and the **signal / null selection ratio**, is the headline: a
   ratio near 1 means the circuit is not distinguishable from paraphrase noise,
   whatever the p-values say.

2. **Sign-flip permutation test** (`--select fdr`, `--select fwer`), kept
   because it is the right rule whenever item-level variance *is* substantial.
   Under the null that the two instructions are interchangeable, each item's
   paired difference is sign-symmetric; sign flips leave the per-coordinate sum
   of squares invariant, so the permuted `t` has a closed form and the test is
   exact and cheap (no scipy in this environment). Per-coordinate p-values go
   through Benjamini–Hochberg; the max-|t| null gives a family-wise alternative
   that assumes nothing about independence between edges.

3. **A held-out item split.** The circuit is selected on half the items; the
   other half only checks sign agreement and `corr(Delta_train, Delta_test)`,
   against a random-edge baseline. When the selected and random baselines are
   equal, that is the diagnostic telling you to use `--select null`.

`--select topk` and `--select abs` are also available; note that top-k selects
`k` edges from the null by construction, so its null count carries no
information.

Reported alongside: Gini of `|Delta|`, and how many edges carry 50% / 90% of
the mass. Those numbers are the direct test of "the difference matrix is
sparse". If 600 of 3773 edges carry half the mass, it is not.

## Verifying causality

Correlational evidence that an edge differs is not evidence it matters.
`verify_circuit.py` intervenes and scores, on the **held-out item half** by
default, with the steering direction taken from the train half.

**Sufficiency** — on neutral prompts, push the circuit toward the pos
instruction: `add` (`alpha[M] += λ·Delta[M]`), `scale` (`alpha[M] *= (1+λ)` —
the literal "enlarge the connection weights"), `signed_scale`
(`alpha[M] += λ·|alpha[M]|·sign(Delta[M])`).

The additive λ grid has to be read against `Delta/alpha`, which the report
prints along with the λ at which `λ·|Delta|` equals `|alpha|`. On the `pred`
channel of the 300M checkpoint that parity λ is ~80: a sweep stopping at λ=8
perturbs the routing weights by 10% of their own size, and a flat result there
says the perturbation was small, not that the circuit is inert.

**Activation patching** — run the *neg* prompt but copy the circuit's routing
values from a rerun of the same tokens under the pos instruction. This is the
donor-defined arm with no hand-chosen magnitude. A full-mask patch is a
comparison, not an upper bound on subsets: coordinate effects can cancel.
The patch is legitimate because the prompts are position-aligned, so donor and
recipient tensors are index-comparable. Donor recomputation is supported on the
`pred` channel.

**Necessity** — zero the circuit while the pos instruction is given.

**Controls**, same prompts, same metric, and by default renormalised to the
circuit's `‖Delta[M]‖₂` so "the circuit wins" cannot just mean "the circuit has
bigger numbers in it":

| arm | rules out |
|---|---|
| `dense_*` | the effect is whatever a bulk shift of all routing weights does |
| `random_*` | any size-matched edit would do it |
| `matched_*` | edits at deep layers / the v stream just do more |
| `complement_*` | the edges the rule rejected work equally well |
| `cross_*` | another behaviour's circuit does the same thing (`--cross-circuit`) |
| per-stream | which streams actually carry the effect |

The `*_scale` controls exist separately from the `*_add` ones because a
multiplicative edit cannot be norm-matched: its size is set by whatever `alpha`
already holds, so "would scaling *any* n edges do this?" needs its own arms.

**Damage gate.** Every arm's held-out NLL is compared with the unedited
reference, and an arm that raises it by more than `--max-nll-rise` (default
0.05 nats) is marked `damaged`. A damaged arm can neither win the verdict nor
be the control that declares a circuit non-specific — otherwise the loudest
"control" is invariably `dense_scale` at the top λ, which moves the metric a
long way because it has destroyed the model. When *every* steering arm is
damaged the verdict says so explicitly: that is a channel with causal leverage
but no usable steering, which is a different result from a channel that does
nothing.

Verdicts compare *toward-target* shifts (`expect · sign(headroom) · shift`),
not `|shift|`, so an arm that moves the behaviour hard in the wrong direction
is never mistaken for the strongest effect.

`dense_*` is not optional. A previous experiment on this model
(`experiments/results/interp/steering2_step9000.json`) found a dense
mean-difference injection on the `pred` channel to be a **literal no-op**
(code_mass 0.009557 → 0.009558 at λ=1). Without a dense arm you cannot tell
"this circuit is not causal" from "nothing on the α side does anything here".

Every non-patch arm also reports NLL on held-out text. A behaviour shift bought
by damaging the language model is not steering.

## Before any of this means anything

Two prerequisites, either of which invalidates everything downstream.

**Does the instruction move the behaviour?** `extract_contrast.py` scores the
behaviour under each instruction with **no** intervention and prints a paired
`t`. That gap is the headroom. A ~300M base LM with no instruction tuning may
simply not follow "answer incorrectly" — and if the instruction does not move
the behaviour, a circuit found downstream is a circuit for *reading* the
instruction, not for doing the thing. The script prints `USABLE` or `WEAK`;
treat `WEAK` as disqualifying for a causal claim, and fall back to
`domain_code`, which is a positive control this repo already has evidence for.

**Can editing alpha move anything?** `routing_leverage.py` measures the
intervention channel itself, independently of any behaviour. Alpha does not
inject signal; it re-weights a mixture the model already computed, so
`dy/dalpha_s = x_s` exactly and the reachable set is the span of the sources.
The script reports, per edge, the rotation and the rescale of a head's input per
unit alpha, plus two sanity checks on the channel: what share of the alpha mass
sits on hyperconnections rather than on the bias-initialised sequential path,
and how much alpha varies across tokens at all. A predictor emitting
near-constant weights has learned a static architecture, and "the routing this
token got" is not then a meaningful object. Given `--delta` it pushes a measured
behavioural difference through the same algebra to report how far the
instruction actually rotates each head's input — the number that decides whether
a flat steering result is an architectural ceiling or a bug in the intervention.

It rebuilds the model's internal `layer_outputs` from module hooks and asserts
that the rebuilt stack reproduces the model's own logits before reporting
anything, because every number in it is wrong if that reconstruction drifts.

## Is the difference a circuit or a steering vector?

A dense `Delta` added to alpha at inference *is* classic activation steering,
just in the routing basis rather than the residual stream — the advantage this
directory is chasing only exists if the causal object has a short description
in a basis whose elements have names. Edge sparsity is the cheapest form of
that property, not the only one, so `analyze_structure.py` walks a ladder of
compressions and reports the smallest description that still reproduces `Delta`:

| rung | compression | what it would mean |
|---|---|---|
| 1 | top-k edges | a circuit in the usual sense |
| 2 | top-m heads, full source profile | group-sparse; a head is the unit circuit work uses anyway |
| 3 | rank-r over (slot x source) | a few routing *modes*, each nameable |
| 4 | mean per (stream, layer, source-distance) cell | structured-dense: dense, but still a sentence |
| 5 | one signed scalar | a vector; concede the point |

Fidelity is cosine over eligible coordinates, because `verify_circuit.py`
rescales every edit to a matched norm and so is indifferent to magnitude.

The controls are the whole exercise. A shuffled `Delta` sets the floor for what
concentration looks like by chance, but it is far too weak on its own: routing
differences are low-rank *whether or not a behaviour caused them*, because alpha
itself is low-rank across tokens, and shuffling destroys that shared structure
and scores it as signal. So `--contrast` re-runs the ladder on **balanced-split
null draws** — same items, same paraphrases, same averaging depth, pos-vs-neg
cancelled — and a structural claim counts only if the real `Delta` compresses
better than those. It also runs the selected set through a connectivity test
(do edges chain into paths, or are they independent nudges?) against random
subsets of the same size.

This stage needs no model and runs on the saved `*_stats.npz`.

## Does the pipeline have the power to find anything?

A null result is only a result if the instrument could have seen a positive
one. `planted_circuit.py` plants effects of *known* size and finds where each
stage stops working, in the same induced-rotation unit `routing_leverage.py`
reports, so the floors are directly comparable to what the real instruction
does.

`--part select` adds a sparse signal of known support and magnitude on top of
real balanced-split noise from the checkpoint, then runs the actual
`null_context` + `select` rules and scores precision and recall against the
planted support. This isolates the discovery stage: the noise is real, only the
signal is synthetic.

`--part sensitivity` applies random sparse edits to alpha and measures the
behaviour shift and the held-out NLL rise. This is the power curve of the
verification stage, and it is read under the same damage gate the arms use — a
row that moves the score by 3 nats of NLL has not steered the model, it has
broken it. The planted directions are random, so the floor it reports bounds
the stage from above; a genuinely behavioural direction of the same magnitude
should do better.

## What does alpha_pred encode, if not behaviour?

`probe_alpha.py` decodes candidate variables out of alpha_pred with linear
probes — position, current token, previous token, corpus frequency rank, and
instruction polarity — each against a null that retrains on shuffled labels,
with held-out groups (windows, or items for polarity) that do not straddle the
split. It also reports how much alpha_pred variance survives conditioning on
the current token id, both raw and after the per-position mean is removed:
if nothing survives, the "predicted topology" is a lookup table on (position,
token) and there is no per-context structure to interpret.

No POS tagger or parser is installed here, so `prev_token` and `log_freq_rank`
stand in for the syntactic probes — they test whether alpha_pred sees context
and lexical class at all, which is what those probes were for. Only the
predictor is loaded, so this does not need the base model.

## Behaviours

Registered in `behaviors.py`:

| name | contrast | scorer |
|---|---|---|
| `honesty` | "answer incorrectly" vs "answer truthfully", 52 high-frequency facts | `candidate_margin`: logp(false answer) − logp(true answer) |
| `honesty_fewshot` | the same 52 facts, but lying vs truthful **demonstrations** instead of an imperative | same |
| `domain_code` | code preamble vs prose preamble (positive control) | `candidate_margin` on a code vs a prose continuation |
| `sentiment` | positive vs negative continuation instruction | `token_set_margin` |

`honesty_fewshot` exists because a 300M base LM does not follow "answer
incorrectly" — the instructed gap is inside noise. Few-shot demonstration is
the elicitation method such a model does respond to, and the two specs share
items and scorer so the difference between their step-1 gaps isolates
elicitation from behaviour.

Each needs ≥2 paraphrases per polarity (the balanced-split null needs them) and
a `neutral` pool (the steering arms need somewhere uncommitted to steer from).
Add your own by registering a `BehaviorSpec` or passing a JSON file with the
same field names to `--behavior`.

## Usage

```bash
MODELS=/work/hdd/bfqt/shared/dagformer-models/300m-dagformer

python experiments/interp/extract_contrast.py \
    --config $MODELS/config.yaml --ckpt $MODELS/checkpoint.pt \
    --behavior honesty --channel both

python experiments/interp/discover_circuit.py \
    --behavior honesty --channel pred --select null --q 0.005 --hyper-only

python experiments/interp/verify_circuit.py \
    --config $MODELS/config.yaml --ckpt $MODELS/checkpoint.pt \
    --behavior honesty --channel pred
```

Once two behaviours have circuits, add the specificity arm:

```bash
R=experiments/results/interp/circuits
python experiments/interp/verify_circuit.py \
    --config $MODELS/config.yaml --ckpt $MODELS/checkpoint.pt \
    --behavior honesty --channel pred \
    --cross-circuit $R/circuit_domain_code_pred_stats.npz
```

Or `bash experiments/interp/run_pipeline.sh honesty` / `sbatch
scripts/slurm/interp_behavior_circuit.slurm`.

The power analysis and the probes, which do not depend on a behaviour having
been found:

```bash
R=experiments/results/interp/circuits

python experiments/interp/planted_circuit.py --part select \
    --stats $R/circuit_domain_code_pred_stats.npz \
    --leverage $R/leverage_domain_code_pred.npz \
    --contrast $R/contrast_domain_code.pt --out $R/planted_select_domain_code_pred

python experiments/interp/planted_circuit.py --part sensitivity \
    --config $MODELS/config.yaml --ckpt $MODELS/checkpoint.pt \
    --stats $R/circuit_domain_code_pred_stats.npz \
    --leverage $R/leverage_domain_code_pred.npz \
    --behavior domain_code --k 16 256 3234 --mags 0.01 0.05 0.1 0.2 0.5 1.0 5.0 \
    --out $R/planted_sensitivity_domain_code_pred

python experiments/interp/probe_alpha.py \
    --config $MODELS/config.yaml --ckpt $MODELS/checkpoint.pt \
    --behavior domain_code --eligible $R/circuit_domain_code_pred_stats.npz \
    --out $R/probe_alpha_domain_code
```

Only the `pred` channel is needed for steps 1–2, and
`--channel pred --no-behavior-score` skips the base-model load entirely, which
makes discovery runnable on CPU.

## Outputs

Written to `experiments/results/interp/circuits/`:

```
contrast_<behavior>.pt              alpha per prompt + ids + behaviour scores
contrast_<behavior>_summary.json    the instruction-sensitivity verdict
circuit_<behavior>_<channel>.json   the graph, selection rule, sparsity, replication
circuit_<behavior>_<channel>.md     human-readable report
circuit_<behavior>_<channel>.dot    Graphviz (dot -Tpdf)
circuit_<behavior>_<channel>_stats.npz   Delta, t, p, masks, item split
verify_<behavior>_<channel>.json    every arm
verify_<behavior>_<channel>.md      dose-response table and verdict
leverage_<behavior>_<channel>.{json,npz}   per-edge rotation and rescale per unit alpha
structure[_<behavior>].{json,md}    the compression ladder against its null
planted_{select,sensitivity}_<behavior>_<channel>.{json,md}   power floors
probe_alpha_<behavior>.{json,md}    what alpha_pred decodes to
```

## Results on the shared 300M checkpoint

`300m-dagformer`, L=12, H=16, D=3773, 3234 eligible hyperconnections, `--select
null --q 0.005 --hyper-only`. These are the numbers the pipeline currently
produces; they are a result about this checkpoint, not about the method.

**Step 1 — can the instruction move the behaviour at all?**

| behaviour | gap (pos − neg) | paired t | verdict |
|---|---|---|---|
| `honesty` (instructed) | −0.024 | −1.33 | WEAK, and the wrong way |
| `honesty_fewshot` (2-shot) | +0.004 | +0.16 | WEAK |
| `honesty_fewshot` (4-shot) | +0.013 | +0.83 | WEAK |
| `domain_code` | **+3.333** | **+16.02** | USABLE |

The model cannot be made to lie — not by instruction, and not by
demonstration. The shift does grow monotonically with the number of shots, so
the elicitation is working in the right direction; it is the 300M model that
has almost no instructed-lying behaviour to find. Everything downstream of a
WEAK row is a circuit for *reading* the instruction.

**Step 2 — how much does the instruction move `alpha`?** (as a fraction of mean
`|alpha|`)

| behaviour / channel | instruction | content | paraphrase | instr/content | edges | null ratio |
|---|---|---|---|---|---|---|
| `domain_code` / `pred` | 0.0082 | 0.0075 | 0.0119 | 1.09 | 513 | 12.0× |
| `domain_code` / `corr` | 0.2514 | 0.2051 | 0.0988 | 1.23 | 829 | 130.9× |
| `honesty` / `pred` | 0.0045 | 0.0055 | 0.0101 | 0.81 | 2 | 0.40× |
| `honesty` / `corr` | 0.0229 | 0.1194 | 0.0542 | 0.19 | 13 | 0.16× |
| `honesty_fewshot` / `pred` | 0.0018 | 0.0053 | 0.0067 | 0.34 | 0 | 0.00× |
| `honesty_fewshot` / `corr` | 0.0101 | 0.1165 | 0.0336 | 0.09 | 0 | 0.00× |

Two things to read here. On `pred`, the instruction moves the routing weights
by well under 1% of their own magnitude for every behaviour — and for the
honesty contrasts by *less than the spread between two paraphrases of the same
instruction*, which is why the null rule selects nothing and the ratio sits
below 1. On `corr` with a behaviour that actually moves, the instruction effect
is 25% of `|alpha|` and exceeds the content effect: a large, behaviour-specific
routing response, selected at a 131× signal/null ratio.

`Delta` is **not** sparse in any of these runs: 421–610 of 3234 edges carry
half the `|Delta|` mass, Gini 0.53–0.61. The premise that a behaviour maps to a
small set of hyperconnections is not supported here.

**Step 3 — does intervening move the behaviour?** (`domain_code`, headroom
+3.35)

* `pred`: patching **every** eligible coordinate with the value the other
  instruction produces moves the behaviour `+0.000991`, **0.0296% of headroom**
  in the committed verification JSON. Steering and knockout arms
  show no effect in the intended direction. The predictor reads the instruction
  without re-routing on it.
* `corr`: `Delta/alpha` is 0.32–0.45, and editing does move the behaviour hard
  (up to −5.5) — but every arm trips the damage gate, up to +6.7 nats of
  held-out NLL. Tightening to 162 edges at `--q 0.0005` and dropping λ to 0.05
  gets under the gate, and there the shift is `+0.033` (1% of headroom, inside
  noise). The tested grid did not produce a supported improvement under this
  corpus's NLL gate.

So for this checkpoint: the structure predictor is a content-driven router that
the instruction barely perturbs, and the correction MLPs carry a large
instruction signal. The tested instruction-contrast directions did not provide
useful domain-code steering under the reported damage criterion. Direct
head-specific edits, other behaviors and other checkpoints require separate
tests; the author's context-fidelity experiments address one such setting.

### Step 0 — is the channel usable? (`routing_leverage.py`, `domain_code`)

| | `pred` | `corr` | `eff` |
|---|---|---|---|
| hyperconnection share of \|alpha\| mass | 0.708 | 0.912 | 0.833 |
| mean \|alpha\|, sequential path | 0.942 | 0.311 | 1.050 |
| mean \|alpha\|, hyperconnections | 0.380 | 0.537 | 0.873 |
| alpha variation across tokens (/ mean \|alpha\|) | **0.042** | **0.515** | 0.352 |
| alpha variation across prompts | 0.066 | 0.430 | 0.282 |
| head-input rotation induced by the behavioural `Delta` | **0.0033** | **0.0838** | — |

Rotation per unit alpha is `0.456` and rescale is `0.389`, so the sources are
*not* collinear and alpha has plenty of leverage in principle. The bottleneck is
upstream of the leverage: the predictor barely moves. Two things follow.

- Hyperconnections carry 71–91% of the routing mass, so the DAG is genuinely
  used — this checkpoint is not a vanilla transformer with decorations.
- The per-token topology lives almost entirely in `corr`, which varies by 52% of
  its own magnitude across tokens, against 4% for `pred`. The module the
  interpretability story is built around emits a near-static architecture, and
  the dynamic router is the one that reads hidden states. That single fact
  predicts the whole results table above: `pred`'s behavioural `Delta` rotates a
  head's input by 0.33% and moves the behaviour 0.1% of headroom, while `corr`'s
  rotates it by 8.4% and moves the behaviour hard enough to break the model.

### Step 3.5 — circuit or steering vector? (`analyze_structure.py`)

For `domain_code`, the behaviour with usable headroom, **no compression beats
the behaviour-free null** on either channel:

| compression | real (`pred`) | null p95 | real (`corr`) | null p95 |
|---|---|---|---|---|
| top-64 edges | 0.758 | 0.797 | 0.894 | 0.923 |
| top-8 heads | 0.486 | 0.561 | 0.779 | 0.824 |
| rank-1 | 0.859 | 0.883 | 0.944 | 0.957 |
| rank-2 | 0.924 | 0.933 | 0.962 | 0.971 |

The real `Delta` sits above the null *mean* almost everywhere and below its p95
everywhere: ordinary sampling variation, not behavioural structure. Connectivity
of the selected set matches random subsets of the same size exactly (0.67 vs
0.68 of edges having a selected predecessor, longest chain 6 vs 6).

The weaker shuffled-`Delta` null is what makes this worth stating. Against it,
`honesty_fewshot` — where *zero* edges survive selection, i.e. pure noise —
scores rank-1 cosine `0.944`, **better** than `domain_code`'s `0.859`. Routing
differences are low-rank whether or not a behaviour caused them. Any structural
claim measured against a shuffle is measuring alpha's own geometry.

### Power — could any of this have found a circuit? (`planted_circuit.py`)

The two stages come apart, and this is the most useful thing the directory has
measured.

**Discovery has ample power.** Planting a 16-edge circuit into real
balanced-split noise, the `--select null` rule at `q=0.005` recovers it at an
induced rotation of **0.00013**, twenty-five times smaller than the 0.0033 the
real instruction produces, at precision 0.90. At the real instruction's own
rotation budget spread over 256 edges (0.0040), recall is 0.90 and precision
0.97. The rule admits 9 edges from pure noise, so the false-positive floor is
low enough not to swamp any of this.

| planted | induced rotation | recall | precision |
|---|---|---|---|
| 16 edges @ 0.005 | 0.00006 | 0.46 | 0.89 |
| 16 edges @ 0.01 | 0.00013 | 0.73 | 0.90 |
| 64 edges @ 0.02 | 0.00113 | 0.90 | 0.94 |
| 256 edges @ 0.02 | 0.00402 | 0.90 | 0.97 |

So "we found no circuit" is not "we could not have found one". If a sparse
circuit of the strength the instruction actually carries existed in `pred`, the
selection rule would have recovered most of it.

**Verification has none.** Sweeping random sparse edits from 16 to all 3234
eligible edges and magnitudes from 0.01 to 5.0, **not one cell moves the
behaviour by 2 se while keeping the NLL rise under the 0.05 gate.** Every arm
that moves the score at all is already several nats into breaking the model:
`k=3234 @ 0.5` shifts 8.2 se at `+4.70` nats, `k=256 @ 0.5` shifts 2.5 se at
`+0.92`. Below the gate the shifts are 0.0-0.4 se regardless of `k`. The
directions are random, so they are sensitivity checks rather than known-causal
positive controls or upper bounds. `verify_circuit.py` also tested the aligned
instruction direction, and its full patch moves 0.0296% of headroom.

The conclusion is limited to the tested behavior, masks, directions and doses.
These results do not establish that all edits to `alpha_pred` are causally inert.

### What alpha_pred does encode (`probe_alpha.py`, 32x256 held-out tokens)

| probe | metric | score | shuffled-label null | trivial baseline |
|---|---|---|---|---|
| position | R² | **0.979** | -0.036 | 0.000 |
| current token id | acc | **0.998** | 0.038 | 0.092 |
| log frequency rank | R² | 0.592 | -0.050 | 0.000 |
| previous token id | acc | 0.275 | 0.039 | 0.083 |
| instruction polarity | acc | **1.000** | 0.495 | 0.500 |

Share of alpha_pred variance left unexplained by absolute position: **0.047**.
By current token id, after the per-position mean is removed: **0.119 of that
residual variance**. The two numbers have different denominators; 0.119 is
not the unexplained fraction of the original total variance.
`prev_token` at 0.275 against a 0.083 baseline says the remainder is not
nothing — there is real but weak context sensitivity — and that residual is
where any train-time story would have to live.

The last row is the one to sit with. **Instruction polarity is decodable from
alpha_pred at 100% on held-out items**, from the content span alone, where the
tokens are identical across polarities. The predictor knows exactly which
instruction it was given. It writes that knowledge into the routing weights.
And moving the routing weights along that same direction does essentially
nothing to the scored domain-code output in the tested intervention. This
distinguishes decodability from a causal effect of that direction. The probe
holds out content items while sharing instruction templates, so it also does
not establish generalization to unseen instructions.

## What a negative result looks like, and why to publish it

The design deliberately makes it possible to fail loudly:

* `WEAK` instruction sensitivity → the behaviour was never there to find.
* `|Delta|` mass spread over hundreds of edges, Gini near a random baseline →
  the difference matrix is not sparse, so the "local circuit" premise fails for
  this behaviour.
* Balanced-split null selects as many edges as the real contrast → the
  selection rule is picking noise.
* `dense_add` flat at large λ → the `pred` channel has no causal leverage in
  this checkpoint, and no circuit story is testable through it. Try `corr`.
* Circuit and `matched_random` move the behaviour equally → the effect is
  positional, not topological.

Any of these is a real result about how much behavioural structure the routing
graph carries. What would not be a real result is a circuit picked with a
hand-tuned threshold, amplified until something moved, with no control arm.

Nor is it a real result until `planted_circuit.py` has been run. "We found
nothing" and "we could not have seen anything this small" are different claims,
and only the power analysis separates them. On this checkpoint it separates them
cleanly: discovery would have found a circuit twenty-five times weaker than the
one the instruction carries, and no alpha edit of any size steers the behaviour
without breaking the model. The negative is about the checkpoint, not the tools.
