# Causal verification — `domain_code` / channel `corr`

Circuit: 829 edges, rule `{"rule": "null", "q": 0.005, "s_cutoff": 5.1315412039172745, "expected_false_positives": 16.17, "n_null_draws_est": 6, "n_null_draws_cal": 6}`  
Intervening on the `corr` channel, scored on the **test** item half (8 items)  
Metric: mean logp(code) - logp(prose) per token  
Controls norm-matched; edits applied from the content span

## Verdict

- **steering: ALL ARMS DAMAGED — every one of the 10 arms raised held-out NLL by more than 0.05 nats (largest shift +4.0383 by `circuit_signed_scale@9` at +4.74 nats); this channel moves the behaviour only by breaking the model**
- **necessity (knockout): DAMAGED — `zero_circuit_on_pos` shifts -0.2026 but raises held-out NLL by +0.354 nats; that is model damage, not steering**

- reference: pos `-2.3485`, neutral `-4.8126`, neg `-5.7002` → headroom `+3.3517`
- step-1 instruction sensitivity: `USABLE` (paired t = 16.02)
- circuit composition: 829 hyperconnections, 0 sequential
- on the circuit: mean `|alpha|` = `0.4658`, mean `|Delta|` = `0.2108` → `Delta/alpha` = `0.453`. If `|alpha|` is tiny the multiplicative arms have almost nothing to enlarge, and a flat `circuit_scale` row says that rather than saying the circuit is inert.
- the additive arms reach parity with `|alpha|` at `lam = 2.21`; this sweep went to `64`
- over the whole channel `Delta/alpha` = `0.317` — that is the entire effect the instruction has on the routing weights, and so the size of the perturbation the full patch arm applies
- arms are marked **damaged** when held-out NLL rises more than `0.05` nats; they are excluded from the verdict on both sides, since an arm that breaks the model neither steers it nor proves a circuit unspecific

`% headroom` is the shift as a fraction of the pos−neg gap, signed by intent: 100% means the edit reproduced the whole effect of changing the instruction, and a knockout reads positive when it does suppress the behaviour. `t` is a paired test over prompts against the arm's own baseline.

## Arms

| arm | group | edges | lam | score | shift vs base | t | % headroom | NLL |
|---|---|---|---|---|---|---|---|---|
| `ref_neutral` | reference | 0 | 0 | -4.8126 ± 0.4388 | +0.0000 | +0.00 | +0.0% | 5.3062 (+0.0000) |
| `ref_pos` | reference | 0 | 0 | -2.3485 ± 0.2564 | +2.4641 | +nan | +73.5% | 5.3062 (+0.0000) |
| `ref_neg` | reference | 0 | 0 | -5.7002 ± 0.3019 | -0.8876 | +nan | -26.5% | 5.3062 (+0.0000) |
| `circuit_add@1` | sufficiency | 829 | 1 | -4.8629 ± 0.4362 | -0.0502 | -2.48 | -1.5% | 5.3676 (+0.0614) **damaged** |
| `dense_add@1` | control | 3234 | 1 | -4.7859 ± 0.4362 | +0.0267 | +0.94 | +0.8% | 5.4461 (+0.1400) **damaged** |
| `random_add@1` | control | 829 | 1 | -4.8182 ± 0.4428 | -0.0056 | -0.34 | -0.2% | 5.3701 (+0.0640) **damaged** |
| `matched_add@1` | control | 829 | 1 | -4.7531 ± 0.4428 | +0.0595 | +2.04 | +1.8% | 5.4430 (+0.1368) **damaged** |
| `complement_add@1` | control | 829 | 1 | -4.8037 ± 0.4362 | +0.0089 | +1.03 | +0.3% | 5.3187 (+0.0125) |
| `circuit_add@4` | sufficiency | 829 | 4 | -4.9648 ± 0.4601 | -0.1522 | -2.36 | -4.5% | 5.9784 (+0.6722) **damaged** |
| `dense_add@4` | control | 3234 | 4 | -4.7533 ± 0.4390 | +0.0594 | +0.47 | +1.8% | 7.6295 (+2.3233) **damaged** |
| `random_add@4` | control | 829 | 4 | -4.5807 ± 0.4187 | +0.2320 | +4.32 | +6.9% | 5.6181 (+0.3120) **damaged** |
| `matched_add@4` | control | 829 | 4 | -4.6101 ± 0.3839 | +0.2026 | +2.91 | +6.0% | 5.7837 (+0.4775) **damaged** |
| `complement_add@4` | control | 829 | 4 | -4.8058 ± 0.4335 | +0.0068 | +0.20 | +0.2% | 5.4645 (+0.1583) **damaged** |
| `circuit_add@16` | sufficiency | 829 | 16 | -4.1948 ± 0.4498 | +0.6179 | +2.27 | +18.4% | 9.0786 (+3.7724) **damaged** |
| `dense_add@16` | control | 3234 | 16 | +0.2784 ± 0.2552 | +5.0910 | +7.87 | +151.9% | 15.6775 (+10.3713) **damaged** |
| `random_add@16` | control | 829 | 16 | -0.0539 ± 0.2903 | +4.7587 | +14.92 | +142.0% | 13.5542 (+8.2480) **damaged** |
| `matched_add@16` | control | 829 | 16 | -4.1921 ± 0.3872 | +0.6206 | +5.90 | +18.5% | 6.1933 (+0.8871) **damaged** |
| `complement_add@16` | control | 829 | 16 | -5.1453 ± 0.4378 | -0.3326 | -2.61 | -9.9% | 7.8688 (+2.5626) **damaged** |
| `circuit_add@64` | sufficiency | 829 | 64 | -3.7866 ± 0.2316 | +1.0260 | +2.91 | +30.6% | 10.8743 (+5.5681) **damaged** |
| `dense_add@64` | control | 3234 | 64 | -3.2542 ± 0.3301 | +1.5584 | +2.24 | +46.5% | 12.3467 (+7.0405) **damaged** |
| `random_add@64` | control | 829 | 64 | -2.0493 ± 0.4337 | +2.7634 | +8.18 | +82.4% | 9.7950 (+4.4888) **damaged** |
| `matched_add@64` | control | 829 | 64 | +2.8824 ± 0.2733 | +7.6951 | +15.79 | +229.6% | 12.4731 (+7.1669) **damaged** |
| `complement_add@64` | control | 829 | 64 | -6.2384 ± 0.5386 | -1.4258 | -3.75 | -42.5% | 15.9545 (+10.6484) **damaged** |
| `circuit_scale@1` | amplify | 829 | 1 | -4.8631 ± 0.4160 | -0.0505 | -1.12 | -1.5% | 5.3671 (+0.0609) **damaged** |
| `circuit_signed_scale@1` | amplify | 829 | 1 | -4.8398 ± 0.4492 | -0.0272 | -0.62 | -0.8% | 5.4586 (+0.1524) **damaged** |
| `dense_scale@1` | control | 3234 | 1 | -4.7156 ± 0.4503 | +0.0970 | +0.76 | +2.9% | 6.4585 (+1.1524) **damaged** |
| `random_scale@1` | control | 829 | 1 | -4.6345 ± 0.4251 | +0.1781 | +4.55 | +5.3% | 5.3707 (+0.0646) **damaged** |
| `matched_scale@1` | control | 829 | 1 | -4.6003 ± 0.4272 | +0.2123 | +4.91 | +6.3% | 5.4202 (+0.1140) **damaged** |
| `complement_scale@1` | control | 829 | 1 | -4.8692 ± 0.4375 | -0.0565 | -0.90 | -1.7% | 5.8579 (+0.5517) **damaged** |
| `circuit_scale_on_pos@1` | amplify | 829 | 1 | -2.2201 ± 0.2688 | +0.1284 | +3.88 | +3.8% | 5.3671 (+0.0609) **damaged** |
| `circuit_scale@3` | amplify | 829 | 3 | -4.8632 ± 0.3630 | -0.0506 | -0.40 | -1.5% | 6.0139 (+0.7077) **damaged** |
| `circuit_signed_scale@3` | amplify | 829 | 3 | -3.8339 ± 0.6230 | +0.9787 | +4.90 | +29.2% | 6.6047 (+1.2986) **damaged** |
| `dense_scale@3` | control | 3234 | 3 | -7.5183 ± 1.0249 | -2.7057 | -3.23 | -80.7% | 12.7903 (+7.4842) **damaged** |
| `random_scale@3` | control | 829 | 3 | -4.6590 ± 0.4480 | +0.1536 | +1.34 | +4.6% | 5.6641 (+0.3579) **damaged** |
| `matched_scale@3` | control | 829 | 3 | -4.4767 ± 0.5004 | +0.3359 | +2.61 | +10.0% | 5.9772 (+0.6710) **damaged** |
| `complement_scale@3` | control | 829 | 3 | -9.7056 ± 0.4716 | -4.8930 | -7.04 | -146.0% | 10.6127 (+5.3065) **damaged** |
| `circuit_scale_on_pos@3` | amplify | 829 | 3 | -2.0905 ± 0.3049 | +0.2580 | +2.90 | +7.7% | 6.0139 (+0.7077) **damaged** |
| `circuit_scale@9` | amplify | 829 | 9 | -3.6694 ± 0.5248 | +1.1432 | +3.06 | +34.1% | 8.3404 (+3.0342) **damaged** |
| `circuit_signed_scale@9` | amplify | 829 | 9 | -0.7744 ± 0.4837 | +4.0383 | +9.59 | +120.5% | 10.0486 (+4.7424) **damaged** |
| `dense_scale@9` | control | 3234 | 9 | -8.0018 ± 1.6731 | -3.1892 | -1.74 | -95.1% | 18.2849 (+12.9787) **damaged** |
| `random_scale@9` | control | 829 | 9 | -4.0350 ± 0.4708 | +0.7777 | +2.34 | +23.2% | 9.2028 (+3.8967) **damaged** |
| `matched_scale@9` | control | 829 | 9 | -3.0465 ± 0.4929 | +1.7661 | +3.01 | +52.7% | 9.0899 (+3.7837) **damaged** |
| `complement_scale@9` | control | 829 | 9 | -5.8710 ± 1.0629 | -1.0584 | -1.20 | -31.6% | 15.4875 (+10.1813) **damaged** |
| `circuit_scale_on_pos@9` | amplify | 829 | 9 | -1.6722 ± 0.3687 | +0.6763 | +3.35 | +20.2% | 8.3404 (+3.0342) **damaged** |
| `circuit_add_q@64` | stream | 269 | 64 | -3.4612 ± 0.4673 | +1.3514 | +9.53 | +40.3% | — |
| `circuit_add_k@64` | stream | 299 | 64 | -3.5822 ± 0.4525 | +1.2304 | +10.62 | +36.7% | — |
| `circuit_add_v@64` | stream | 245 | 64 | -2.7615 ± 0.2416 | +2.0511 | +6.97 | +61.2% | — |
| `circuit_add_r@64` | stream | 16 | 64 | -4.4492 ± 0.2988 | +0.3634 | +1.04 | +10.8% | — |
| `zero_circuit_on_pos` | necessity | 829 | 0 | -2.5511 ± 0.2691 | -0.2026 | -5.48 | +6.0% | 5.6604 (+0.3542) **damaged** |
| `zero_random_on_pos` | control | 829 | 0 | -2.2460 ± 0.2699 | +0.1025 | +2.45 | -3.1% | 5.5502 (+0.2440) **damaged** |

## How to read this

- If `dense_add` moves the behaviour as much as `circuit_add`, the circuit is not localised — the effect is whatever a bulk shift of the routing weights does.
- If `dense_add` and every other arm are flat at large λ, the α channel has no causal leverage at all in this checkpoint and the discovery result is uninterpretable as a causal claim (this is what `steering2` found for a dense α-side injection).
- `patch_pos_into_neg` reaching `ref_pos` while `patch_pos_into_neg_random` does not is the cleanest possible positive result: no chosen magnitude, and the ceiling arm bounds it.
- q/k stream arms being flat while v/r move is expected, not a bug: q_norm/k_norm renormalise after mixing, so magnitude changes on those streams are partly undone.
- Any arm whose NLL rises sharply bought its behaviour shift by damaging the model; it is not steering.
