# Experiment design: V-runaway stabilization (analyzable, non-softmax)

**Status:** designed, not yet run. Phase 3 (small-scale, cheap) — runs after the
1B + 600M campaigns free disk/GPU. Owner prefers analyzable, non-softmax fixes.

## Motivation
The FourWay model's **V stream is the runaway** (its routing magnitude grows
uncontrolled; the main training instability). The current mitigation `use_v_norm`
(global 2048-dim RMSNorm after the V mix) has two weaknesses the model review
surfaced:
1. It controls only the *global* RMS of the 2048-vector — a single runaway head
   or source layer can still dominate as long as overall RMS is renormalized.
2. It is **not identity-preserving** at init (vanilla OLMo-2 has no V-norm), so it
   slightly shifts step-0 away from the exact baseline.

`R` is even more load-bearing (it is the entire residual carrier, doubling as
router + mandatory skip), so the same reparam is worth testing on R.

## The idea (identity-anchored residual Δ)
Reparameterize the per-head V mix as **deviation from the vanilla value**:

```
V_head = V_last  +  Δ,     Δ = Σ_{l'<L} α_v[l'] · (V_{l'} − V_last)
```
where `V_last` is the current layer's own projected V (the vanilla value) and the
identity slot is excluded from the Δ sum. Then:
- Runaway becomes **directly measurable** as `‖Δ‖` (log it per layer/step).
- The regularizer becomes *correct*: penalize `Δ.pow(2).mean()` (a true
  "stay-near-vanilla" pressure), unlike the current `routing_l2` which — bug —
  penalizes the identity slot's `1.0` and pulls α toward all-zeros.
- **Identity init is exact** (α at identity ⇒ Δ=0 ⇒ V_head=V_last = vanilla V).

Optional add-on: **per-head soft RMS clamp** instead of / on top of global v_norm:
`V ← V · min(1, c·rms_ref / rms_head(V))` — identity-preserving in-range, only
bites the outlier heads the global norm misses. One scalar `c`.

## Arms (75M first, promote winner to 150M) — Dolma v1.7, matched tokens
| arm | config | tests |
|-----|--------|-------|
| A `dense` | baseline (no routing) | floor reference |
| B `vnorm` | current: `use_v_norm=true` | the status quo |
| C `resid_delta_v` | V = V_last + Δ, penalty `λ·‖Δ‖²`, no v_norm | the proposal |
| D `resid_delta_v + clamp` | C + per-head soft clamp | belt-and-suspenders |
| E `resid_delta_vr` | C applied to **both** V and R | R also anchored |

All arms: identity init verified at step 0 (assert NLL(step0) == dense NLL within
0.01). Same seed, tokens, LR schedule.

## Metrics / decision gate
- `train/nll` trajectory + final (primary).
- **`routing/delta_v_norm`** and `routing/delta_r_norm` per layer (the runaway
  signal — expect C/D/E to keep these bounded vs B blowing up ~step 5k).
- Downstream (lambada/hellaswag/piqa/arc_e/sciq/wtppl) at the end.
- **Gate:** an arm "wins" if final train NLL ≤ vnorm (B) AND its Δ-norm stays
  bounded (no >±10 spikes) AND downstream ≥ B. Promote the winner to 150M, then
  fold into the next 1B config.

## Implementation notes (do at Phase 3, with a GPU sanity check)
- `src/model/olmo_graph.py` `FourWayDAGFormer.forward`: change the V-mix (and
  optionally R-mix) to the `V_last + Δ` form; add `routing/delta_*_norm` logging.
- New config flags: `v_parameterization: "raw"|"resid_delta"` (default raw =
  current), `delta_l2_lambda`, `v_head_clamp_c` (0 = off), `r_parameterization`.
- Fix the existing `routing_l2/l1` to penalize `(α − identity)` (or deprecate in
  favor of `delta_l2_lambda`), and guard the identity-destroying knobs
  (`softmax_rv`/`softmax`/`temperature`/`sinkhorn`) with a startup assert that
  step-0 NLL still matches dense (catches silent identity breakage).
- **Baseline-reproduction sanity check must pass** before any real run
  (CLAUDE.md §4.3 / §9 invariant).

## Cost
5 arms × 75M × (Dolma, ~Chinchilla tokens) on A40 ≈ small. Promote 1 winner to
150M. Total a fraction of one 1B run. Uses the Dolma mmap shards built in Phase 2.
