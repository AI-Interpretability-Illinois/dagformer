## What this is

A five-stage pipeline for asking whether behaviour-specific circuits exist in
the DAGFormer routing graph and can be steered through it, plus the two stages
that decide whether a negative answer is reportable. Run on the shared 300M
checkpoint, the answer is **no** — and the point of the extra stages is that
"no" here is a measurement rather than a shrug.

## The pipeline

| stage | question | control |
|---|---|---|
| `extract_contrast.py` | does the instruction move the behaviour at all? | paired `t` gate; `WEAK` disqualifies everything downstream |
| `discover_circuit.py` | which edges differ? | balanced-split null — same items, same paraphrases, same averaging depth, pos-vs-neg cancelled |
| `verify_circuit.py` | is the difference causal? | matched-norm random / complement / cross-behaviour arms, all under a held-out NLL damage gate |
| `routing_leverage.py` | is the channel usable at all? | rebuilds `layer_outputs` from module hooks and asserts the rebuild reproduces the model's own logits |
| `analyze_structure.py` | circuit, or steering vector? | compression ladder (top-k, head groups, low-rank, structured cells) scored against behaviour-free null draws |
| `planted_circuit.py` | could any of this have found a circuit? | plants effects of known size into real noise; reports floors in induced-rotation units |
| `probe_alpha.py` | what does `alpha_pred` encode instead? | shuffled-label nulls, held-out groups that don't straddle the split |

## What it finds on `300m-dagformer`

**Discovery has power to spare.** A planted 16-edge circuit is recovered at an
induced rotation of **0.00013** at precision 0.90 — twenty-five times below the
0.0033 the real instruction produces. At the instruction's own rotation budget
spread over 256 edges, recall 0.90 / precision 0.97. So "no circuit found" is
not "could not have found one".

**Verification has none.** No sparse alpha edit, from 16 to all 3234 eligible
edges and magnitude 0.01 to 5.0, moves the behaviour by 2 se while keeping the
NLL rise under the 0.05 nat gate. Everything that moves the score is already
several nats into breaking the model. The aligned direction agrees: the full
patch moves 0.1% of headroom.

**`alpha_pred` is readable but not writable.** Linear probes over 32x256
held-out tokens:

| probe | metric | score | shuffled null | baseline |
|---|---|---|---|---|
| position | R² | 0.979 | -0.036 | 0.000 |
| current token id | acc | 0.998 | 0.038 | 0.092 |
| log frequency rank | R² | 0.592 | -0.050 | 0.000 |
| previous token id | acc | 0.275 | 0.039 | 0.083 |
| **instruction polarity** | acc | **1.000** | 0.495 | 0.500 |

Only 4.7% of the variance survives conditioning on absolute position, and 11.9%
on token id once position is removed — the predicted topology is ~88% a lookup
table on (position, token). Instruction polarity nonetheless decodes at 100% on
held-out items, off a content span whose tokens are identical across
polarities. The predictor knows exactly which instruction it was given, writes
that into the routing weights, and nothing downstream reads it.

**No compression of `Delta` beats the behaviour-free null** on either channel,
and the weaker shuffled-`Delta` null is actively misleading: `honesty_fewshot`,
where zero edges survive selection, scores rank-1 cosine 0.944 against
`domain_code`'s 0.859. Routing differences are low-rank whether or not a
behaviour caused them.

## The conclusion

The structure predictor is **load-bearing as a parameter and vestigial as a
predictor**: training kept its average and ignored its variation. The per-token
routing that does vary lives in the correction MLPs, which read hidden states
and are therefore as entangled as any ordinary activation — which is exactly
what the architecture's separation of structure from content was supposed to
avoid.

This is a result about this checkpoint's training, not about the architecture.
Alpha has real leverage (rotation of a head's input per unit alpha is 0.456,
the sources are not collinear) and hyperconnections carry 71–91% of the mass,
so the DAG is load-bearing for the model's computation. Nothing in the loss
requires the *predictor* to be the thing that carries it.

## Scope and limits

- One behaviour (`domain_code`) carries the causal claims, because it is the
  only one with usable headroom on a 300M base LM — `honesty` shows a gap of
  the wrong sign (t = -1.33) and the few-shot variants are flat.
- The sensitivity sweep plants random directions, so its floor bounds the
  verification stage from above rather than pinning it. The aligned direction
  was tested separately in `verify_circuit.py` and agrees.
- No POS tagger or parser is installed in the analysis environment, so
  `prev_token` and `log_freq_rank` stand in for the syntactic probes.
- Committed results under `experiments/results/interp/circuits/` are the
  evidence; the `*.pt` and `*.npz` they came from are gitignored and regenerate
  from the commands in the README.

## Suggested next step

The cheapest measurement that would sharpen this is a channel knockout: zero
`corr` alone (alpha_eff = alpha_pred), zero `pred` alone (alpha_eff =
Delta_corr), compare the NLL damage. That turns "the dynamics migrated to
`corr`" from an inference about variance ratios into a measurement about
dependence.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
