# Causal verification — `domain_code` / channel `pred`

Circuit: 513 edges, rule `{"rule": "null", "q": 0.005, "s_cutoff": 1.868829515971337, "expected_false_positives": 16.17, "n_null_draws_est": 8, "n_null_draws_cal": 8}`  
Intervening on the `pred` channel, scored on the **test** item half (8 items)  
Metric: mean logp(code) - logp(prose) per token  
Controls norm-matched; edits applied from the content span

## Verdict

- **steering: NULL — the best arm `circuit_signed_scale@1` shifts +0.0578, inside its own noise**
- **patch pos->neg: NO EFFECT IN THE RIGHT DIRECTION — the best undamaged arm `patch_pos_into_neg` shifts -0.0044, which is the opposite way from what the intervention should do**
- **patch neg->pos: NULL — the best arm `patch_neg_into_pos` shifts -0.0006, inside its own noise**
- **patch ceiling: `patch_pos_into_neg_all` patches every eligible coordinate and moves +0.0010 = 0.0% of headroom — this bounds every circuit on this channel from above**
- **necessity (knockout): NULL — the best arm `zero_circuit_on_pos` shifts -0.0356, inside its own noise**

- reference: pos `-2.3472`, neutral `-4.8204`, neg `-5.6971` → headroom `+3.3499`
- step-1 instruction sensitivity: `USABLE` (paired t = 15.99)
- circuit composition: 513 hyperconnections, 0 sequential
- on the circuit: mean `|alpha|` = `0.4546`, mean `|Delta|` = `0.005314` → `Delta/alpha` = `0.0117`. If `|alpha|` is tiny the multiplicative arms have almost nothing to enlarge, and a flat `circuit_scale` row says that rather than saying the circuit is inert.
- the additive arms reach parity with `|alpha|` at `lam = 85.6`; this sweep went to `64`
- over the whole channel `Delta/alpha` = `0.00808` — that is the entire effect the instruction has on the routing weights, and so the size of the perturbation the full patch arm applies
- arms are marked **damaged** when held-out NLL rises more than `0.05` nats; they are excluded from the verdict on both sides, since an arm that breaks the model neither steers it nor proves a circuit unspecific

`% headroom` is the shift as a fraction of the pos−neg gap, signed by intent: 100% means the edit reproduced the whole effect of changing the instruction, and a knockout reads positive when it does suppress the behaviour. `t` is a paired test over prompts against the arm's own baseline.

## Arms

| arm | group | edges | lam | score | shift vs base | t | % headroom | NLL |
|---|---|---|---|---|---|---|---|---|
| `ref_neutral` | reference | 0 | 0 | -4.8204 ± 0.4370 | +0.0000 | +0.00 | +0.0% | 5.3060 (+0.0000) |
| `ref_pos` | reference | 0 | 0 | -2.3472 ± 0.2572 | +2.4732 | +nan | +73.8% | 5.3060 (+0.0000) |
| `ref_neg` | reference | 0 | 0 | -5.6971 ± 0.3019 | -0.8767 | +nan | -26.2% | 5.3060 (+0.0000) |
| `circuit_add@1` | sufficiency | 513 | 1 | -4.8177 ± 0.4376 | +0.0027 | +0.93 | +0.1% | 5.3060 (-0.0001) |
| `dense_add@1` | control | 3234 | 1 | -4.8244 ± 0.4385 | -0.0040 | -1.33 | -0.1% | 5.3056 (-0.0004) |
| `random_add@1` | control | 513 | 1 | -4.8210 ± 0.4390 | -0.0006 | -0.16 | -0.0% | 5.3066 (+0.0006) |
| `matched_add@1` | control | 513 | 1 | -4.8245 ± 0.4372 | -0.0042 | -1.69 | -0.1% | 5.3058 (-0.0002) |
| `complement_add@1` | control | 513 | 1 | -4.8229 ± 0.4377 | -0.0025 | -0.74 | -0.1% | 5.3063 (+0.0003) |
| `circuit_add@4` | sufficiency | 513 | 4 | -4.8223 ± 0.4372 | -0.0019 | -0.76 | -0.1% | 5.3051 (-0.0009) |
| `dense_add@4` | control | 3234 | 4 | -4.8234 ± 0.4383 | -0.0031 | -0.87 | -0.1% | 5.3063 (+0.0003) |
| `random_add@4` | control | 513 | 4 | -4.8112 ± 0.4383 | +0.0092 | +2.72 | +0.3% | 5.3062 (+0.0002) |
| `matched_add@4` | control | 513 | 4 | -4.8202 ± 0.4369 | +0.0001 | +0.04 | +0.0% | 5.3071 (+0.0011) |
| `complement_add@4` | control | 513 | 4 | -4.8180 ± 0.4378 | +0.0024 | +0.72 | +0.1% | 5.3067 (+0.0007) |
| `circuit_add@16` | sufficiency | 513 | 16 | -4.8267 ± 0.4383 | -0.0063 | -1.30 | -0.2% | 5.3059 (-0.0001) |
| `dense_add@16` | control | 3234 | 16 | -4.8334 ± 0.4360 | -0.0130 | -1.78 | -0.4% | 5.3094 (+0.0033) |
| `random_add@16` | control | 513 | 16 | -4.8451 ± 0.4350 | -0.0247 | -4.73 | -0.7% | 5.3068 (+0.0008) |
| `matched_add@16` | control | 513 | 16 | -4.8135 ± 0.4375 | +0.0069 | +1.97 | +0.2% | 5.3056 (-0.0005) |
| `complement_add@16` | control | 513 | 16 | -4.8067 ± 0.4357 | +0.0137 | +3.02 | +0.4% | 5.3075 (+0.0015) |
| `circuit_add@64` | sufficiency | 513 | 64 | -4.8508 ± 0.4370 | -0.0304 | -1.98 | -0.9% | 5.3173 (+0.0113) |
| `dense_add@64` | control | 3234 | 64 | -4.8177 ± 0.4228 | +0.0027 | +0.10 | +0.1% | 5.3577 (+0.0517) **damaged** |
| `random_add@64` | control | 513 | 64 | -4.8299 ± 0.4325 | -0.0095 | -1.06 | -0.3% | 5.3411 (+0.0351) |
| `matched_add@64` | control | 513 | 64 | -4.7175 ± 0.4328 | +0.1029 | +6.99 | +3.1% | 5.3152 (+0.0092) |
| `complement_add@64` | control | 513 | 64 | -4.7917 ± 0.4296 | +0.0286 | +1.91 | +0.9% | 5.3146 (+0.0086) |
| `circuit_scale@1` | amplify | 513 | 1 | -4.7694 ± 0.4406 | +0.0510 | +2.32 | +1.5% | 5.3372 (+0.0312) |
| `circuit_signed_scale@1` | amplify | 513 | 1 | -4.7626 ± 0.4441 | +0.0578 | +2.25 | +1.7% | 5.3376 (+0.0316) |
| `dense_scale@1` | control | 3234 | 1 | -4.9713 ± 0.4376 | -0.1509 | -3.46 | -4.5% | 5.4250 (+0.1189) **damaged** |
| `random_scale@1` | control | 513 | 1 | -4.8503 ± 0.4357 | -0.0299 | -3.31 | -0.9% | 5.3166 (+0.0106) |
| `matched_scale@1` | control | 513 | 1 | -4.8948 ± 0.4320 | -0.0744 | -3.69 | -2.2% | 5.3309 (+0.0249) |
| `complement_scale@1` | control | 513 | 1 | -4.7296 ± 0.4274 | +0.0908 | +4.11 | +2.7% | 5.3412 (+0.0352) |
| `circuit_scale_on_pos@1` | amplify | 513 | 1 | -2.3171 ± 0.2520 | +0.0301 | +1.77 | +0.9% | 5.3372 (+0.0312) |
| `circuit_scale@3` | amplify | 513 | 3 | -4.6439 ± 0.4416 | +0.1764 | +3.37 | +5.3% | 5.4526 (+0.1466) **damaged** |
| `circuit_signed_scale@3` | amplify | 513 | 3 | -4.6731 ± 0.4506 | +0.1473 | +2.24 | +4.4% | 5.4415 (+0.1355) **damaged** |
| `dense_scale@3` | control | 3234 | 3 | -4.9343 ± 0.4683 | -0.1139 | -0.98 | -3.4% | 6.0354 (+0.7294) **damaged** |
| `random_scale@3` | control | 513 | 3 | -4.5338 ± 0.4481 | +0.2865 | +6.04 | +8.6% | 5.5208 (+0.2148) **damaged** |
| `matched_scale@3` | control | 513 | 3 | -4.8491 ± 0.4293 | -0.0287 | -0.49 | -0.9% | 5.4467 (+0.1407) **damaged** |
| `complement_scale@3` | control | 513 | 3 | -4.5544 ± 0.4071 | +0.2660 | +4.43 | +7.9% | 5.4510 (+0.1450) **damaged** |
| `circuit_scale_on_pos@3` | amplify | 513 | 3 | -2.2615 ± 0.2497 | +0.0857 | +1.79 | +2.6% | 5.4526 (+0.1466) **damaged** |
| `circuit_scale@9` | amplify | 513 | 9 | -4.2780 ± 0.4318 | +0.5424 | +4.49 | +16.2% | 6.0300 (+0.7240) **damaged** |
| `circuit_signed_scale@9` | amplify | 513 | 9 | -4.1350 ± 0.4249 | +0.6853 | +5.62 | +20.5% | 6.0804 (+0.7743) **damaged** |
| `dense_scale@9` | control | 3234 | 9 | -3.1715 ± 0.2670 | +1.6489 | +6.99 | +49.2% | 8.8775 (+3.5715) **damaged** |
| `random_scale@9` | control | 513 | 9 | -3.0942 ± 0.3644 | +1.7262 | +5.87 | +51.5% | 10.0033 (+4.6973) **damaged** |
| `matched_scale@9` | control | 513 | 9 | -4.1276 ± 0.4203 | +0.6928 | +5.70 | +20.7% | 6.1662 (+0.8602) **damaged** |
| `complement_scale@9` | control | 513 | 9 | -4.0282 ± 0.3534 | +0.7921 | +4.92 | +23.6% | 6.0335 (+0.7275) **damaged** |
| `circuit_scale_on_pos@9` | amplify | 513 | 9 | -1.9846 ± 0.2722 | +0.3626 | +2.99 | +10.8% | 6.0300 (+0.7240) **damaged** |
| `circuit_add_q@64` | stream | 144 | 64 | -4.8425 ± 0.4372 | -0.0221 | -3.82 | -0.7% | — |
| `circuit_add_k@64` | stream | 159 | 64 | -4.7820 ± 0.4358 | +0.0384 | +3.79 | +1.1% | — |
| `circuit_add_v@64` | stream | 208 | 64 | -4.8675 ± 0.4415 | -0.0471 | -3.87 | -1.4% | — |
| `circuit_add_r@64` | stream | 2 | 64 | -4.8133 ± 0.4366 | +0.0071 | +1.76 | +0.2% | — |
| `zero_circuit_on_pos` | necessity | 513 | 0 | -2.3828 ± 0.2620 | -0.0356 | -1.71 | +1.1% | 5.3512 (+0.0452) |
| `zero_random_on_pos` | control | 513 | 0 | -2.3285 ± 0.2531 | +0.0187 | +1.21 | -0.6% | 5.3165 (+0.0105) |
| `patch_pos_into_neg` | patch | 513 | 0 | -5.7015 ± 0.3021 | -0.0044 | -2.07 | -0.1% | — |
| `patch_pos_into_neg_random` | control | 513 | 0 | -5.6986 ± 0.3027 | -0.0015 | -0.71 | -0.0% | — |
| `patch_pos_into_neg_all` | patch | 3234 | 0 | -5.6961 ± 0.3022 | +0.0010 | +0.43 | +0.0% | — |
| `patch_neg_into_pos` | patch | 513 | 0 | -2.3478 ± 0.2569 | -0.0006 | -0.37 | +0.0% | — |
| `patch_neg_into_pos_random` | control | 513 | 0 | -2.3468 ± 0.2567 | +0.0004 | +0.20 | -0.0% | — |
| `patch_neg_into_pos_all` | patch | 3234 | 0 | -2.3480 ± 0.2570 | -0.0008 | -0.29 | +0.0% | — |

## How to read this

- If `dense_add` moves the behaviour as much as `circuit_add`, the circuit is not localised — the effect is whatever a bulk shift of the routing weights does.
- If `dense_add` and every other arm are flat at large λ, the α channel has no causal leverage at all in this checkpoint and the discovery result is uninterpretable as a causal claim (this is what `steering2` found for a dense α-side injection).
- `patch_pos_into_neg` reaching `ref_pos` while `patch_pos_into_neg_random` does not is the cleanest possible positive result: no chosen magnitude, and the ceiling arm bounds it.
- q/k stream arms being flat while v/r move is expected, not a bug: q_norm/k_norm renormalise after mixing, so magnitude changes on those streams are partly undone.
- Any arm whose NLL rises sharply bought its behaviour shift by damaging the model; it is not steering.
