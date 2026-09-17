# Causal verification — `domain_code` / channel `corr`

Circuit: 1598 edges, rule `{"rule": "null", "q": 0.005, "s_cutoff": 2.671120999332342, "expected_false_positives": 16.17, "n_null_draws_est": 8, "n_null_draws_cal": 8}`  
Intervening on the `corr` channel, scored on the **test** item half (8 items)  
Metric: mean logp(code) - logp(prose) per token  
Controls norm-matched; edits applied from the content span

## Verdict

- **steering: NULL — the best arm `circuit_scale@0.5` shifts +0.0282, inside its own noise**
- **necessity (knockout): DAMAGED — `zero_circuit_on_pos` shifts -0.3202 but raises held-out NLL by +0.253 nats; that is model damage, not steering**

- reference: pos `-2.3460`, neutral `-4.8171`, neg `-5.7007` → headroom `+3.3548`
- step-1 instruction sensitivity: `USABLE` (paired t = 15.98)
- circuit composition: 1598 hyperconnections, 0 sequential
- on the circuit: mean `|alpha|` = `0.4507`, mean `|Delta|` = `0.1979` → `Delta/alpha` = `0.439`. If `|alpha|` is tiny the multiplicative arms have almost nothing to enlarge, and a flat `circuit_scale` row says that rather than saying the circuit is inert.
- the additive arms reach parity with `|alpha|` at `lam = 2.28`; this sweep went to `64`
- over the whole channel `Delta/alpha` = `0.317` — that is the entire effect the instruction has on the routing weights, and so the size of the perturbation the full patch arm applies
- arms are marked **damaged** when held-out NLL rises more than `0.05` nats; they are excluded from the verdict on both sides, since an arm that breaks the model neither steers it nor proves a circuit unspecific

`% headroom` is the shift as a fraction of the pos−neg gap, signed by intent: 100% means the edit reproduced the whole effect of changing the instruction, and a knockout reads positive when it does suppress the behaviour. `t` is a paired test over prompts against the arm's own baseline.

## Arms

| arm | group | edges | lam | score | shift vs base | t | % headroom | NLL |
|---|---|---|---|---|---|---|---|---|
| `ref_neutral` | reference | 0 | 0 | -4.8171 ± 0.4380 | +0.0000 | +0.00 | +0.0% | 3.3490 (+0.0000) |
| `ref_pos` | reference | 0 | 0 | -2.3460 ± 0.2571 | +2.4712 | +nan | +73.7% | 3.3490 (+0.0000) |
| `ref_neg` | reference | 0 | 0 | -5.7007 ± 0.3028 | -0.8836 | +nan | -26.3% | 3.3490 (+0.0000) |
| `circuit_add@0.05` | sufficiency | 1598 | 0.05 | -4.8116 ± 0.4386 | +0.0056 | +2.43 | +0.2% | 3.3490 (+0.0000) |
| `dense_add@0.05` | control | 3234 | 0.05 | -4.8179 ± 0.4377 | -0.0007 | -0.20 | -0.0% | 3.3490 (+0.0000) |
| `random_add@0.05` | control | 1598 | 0.05 | -4.8265 ± 0.4383 | -0.0094 | -2.36 | -0.3% | 3.3489 (-0.0000) |
| `matched_add@0.05` | control | 1598 | 0.05 | -4.8210 ± 0.4382 | -0.0039 | -1.31 | -0.1% | 3.3491 (+0.0001) |
| `complement_add@0.05` | control | 1598 | 0.05 | -4.8077 ± 0.4385 | +0.0095 | +3.18 | +0.3% | 3.3493 (+0.0003) |
| `circuit_add@0.1` | sufficiency | 1598 | 0.1 | -4.8193 ± 0.4376 | -0.0022 | -0.47 | -0.1% | 3.3492 (+0.0002) |
| `dense_add@0.1` | control | 3234 | 0.1 | -4.8168 ± 0.4368 | +0.0003 | +0.08 | +0.0% | 3.3495 (+0.0005) |
| `random_add@0.1` | control | 1598 | 0.1 | -4.8183 ± 0.4363 | -0.0011 | -0.28 | -0.0% | 3.3494 (+0.0005) |
| `matched_add@0.1` | control | 1598 | 0.1 | -4.8162 ± 0.4382 | +0.0009 | +0.17 | +0.0% | 3.3492 (+0.0002) |
| `complement_add@0.1` | control | 1598 | 0.1 | -4.7878 ± 0.4401 | +0.0293 | +7.01 | +0.9% | 3.3504 (+0.0014) |
| `circuit_add@0.25` | sufficiency | 1598 | 0.25 | -4.8275 ± 0.4356 | -0.0103 | -1.10 | -0.3% | 3.3523 (+0.0034) |
| `dense_add@0.25` | control | 3234 | 0.25 | -4.8147 ± 0.4368 | +0.0025 | +0.30 | +0.1% | 3.3535 (+0.0045) |
| `random_add@0.25` | control | 1598 | 0.25 | -4.8255 ± 0.4409 | -0.0083 | -0.85 | -0.2% | 3.3506 (+0.0016) |
| `matched_add@0.25` | control | 1598 | 0.25 | -4.8139 ± 0.4335 | +0.0032 | +0.31 | +0.1% | 3.3529 (+0.0039) |
| `complement_add@0.25` | control | 1598 | 0.25 | -4.7382 ± 0.4424 | +0.0790 | +9.32 | +2.4% | 3.3574 (+0.0084) |
| `circuit_add@0.5` | sufficiency | 1598 | 0.5 | -4.8312 ± 0.4352 | -0.0141 | -0.90 | -0.4% | 3.3644 (+0.0154) |
| `dense_add@0.5` | control | 3234 | 0.5 | -4.8038 ± 0.4362 | +0.0133 | +0.91 | +0.4% | 3.3684 (+0.0195) |
| `random_add@0.5` | control | 1598 | 0.5 | -4.8294 ± 0.4350 | -0.0122 | -0.94 | -0.4% | 3.3639 (+0.0149) |
| `matched_add@0.5` | control | 1598 | 0.5 | -4.8117 ± 0.4469 | +0.0055 | +0.29 | +0.2% | 3.3683 (+0.0193) |
| `complement_add@0.5` | control | 1598 | 0.5 | -4.6565 ± 0.4463 | +0.1606 | +10.44 | +4.8% | 3.3832 (+0.0343) |
| `circuit_add@1` | sufficiency | 1598 | 1 | -4.8465 ± 0.4331 | -0.0294 | -1.01 | -0.9% | 3.4345 (+0.0856) **damaged** |
| `dense_add@1` | control | 3234 | 1 | -4.7845 ± 0.4373 | +0.0326 | +1.17 | +1.0% | 3.4468 (+0.0979) **damaged** |
| `random_add@1` | control | 1598 | 1 | -4.8535 ± 0.4309 | -0.0363 | -1.78 | -1.1% | 3.3700 (+0.0211) |
| `matched_add@1` | control | 1598 | 1 | -4.8153 ± 0.4347 | +0.0018 | +0.05 | +0.1% | 3.4569 (+0.1079) **damaged** |
| `complement_add@1` | control | 1598 | 1 | -4.4568 ± 0.4561 | +0.3603 | +10.07 | +10.7% | 3.5042 (+0.1552) **damaged** |
| `circuit_add@4` | sufficiency | 1598 | 4 | -4.9980 ± 0.4070 | -0.1808 | -1.57 | -5.4% | 5.3351 (+1.9861) **damaged** |
| `dense_add@4` | control | 3234 | 4 | -4.7458 ± 0.4397 | +0.0714 | +0.57 | +2.1% | 5.5517 (+2.2027) **damaged** |
| `random_add@4` | control | 1598 | 4 | -4.7596 ± 0.4165 | +0.0576 | +0.66 | +1.7% | 4.4513 (+1.1023) **damaged** |
| `matched_add@4` | control | 1598 | 4 | -5.5926 ± 0.4535 | -0.7755 | -5.41 | -23.1% | 5.5604 (+2.2114) **damaged** |
| `complement_add@4` | control | 1598 | 4 | -2.1320 ± 0.3072 | +2.6851 | +10.12 | +80.0% | 8.8878 (+5.5389) **damaged** |
| `circuit_add@16` | sufficiency | 1598 | 16 | +0.8491 ± 0.2994 | +5.6663 | +8.18 | +168.9% | 15.7932 (+12.4443) **damaged** |
| `dense_add@16` | control | 3234 | 16 | +0.2621 ± 0.2543 | +5.0792 | +7.88 | +151.4% | 15.8336 (+12.4846) **damaged** |
| `random_add@16` | control | 1598 | 16 | -2.8548 ± 0.2391 | +1.9624 | +6.37 | +58.5% | 8.4191 (+5.0702) **damaged** |
| `matched_add@16` | control | 1598 | 16 | -2.6708 ± 0.2764 | +2.1464 | +6.99 | +64.0% | 7.3310 (+3.9821) **damaged** |
| `complement_add@16` | control | 1598 | 16 | -0.8333 ± 0.2973 | +3.9838 | +5.87 | +118.8% | 13.9327 (+10.5838) **damaged** |
| `circuit_add@64` | sufficiency | 1598 | 64 | +0.6423 ± 0.2370 | +5.4594 | +9.52 | +162.7% | 12.1401 (+8.7911) **damaged** |
| `dense_add@64` | control | 3234 | 64 | -3.2706 ± 0.3303 | +1.5465 | +2.23 | +46.1% | 10.4011 (+7.0522) **damaged** |
| `random_add@64` | control | 1598 | 64 | -1.1591 ± 0.3689 | +3.6580 | +5.77 | +109.0% | 13.7585 (+10.4095) **damaged** |
| `matched_add@64` | control | 1598 | 64 | +1.5019 ± 0.2509 | +6.3190 | +9.92 | +188.4% | 14.3556 (+11.0066) **damaged** |
| `complement_add@64` | control | 1598 | 64 | -3.9757 ± 0.3715 | +0.8414 | +1.63 | +25.1% | 11.7556 (+8.4066) **damaged** |
| `circuit_scale@0.05` | amplify | 1598 | 0.05 | -4.8161 ± 0.4368 | +0.0010 | +0.23 | +0.0% | 3.3491 (+0.0002) |
| `circuit_signed_scale@0.05` | amplify | 1598 | 0.05 | -4.8208 ± 0.4380 | -0.0037 | -1.07 | -0.1% | 3.3492 (+0.0003) |
| `dense_scale@0.05` | control | 3234 | 0.05 | -4.7996 ± 0.4386 | +0.0175 | +2.29 | +0.5% | 3.3495 (+0.0005) |
| `random_scale@0.05` | control | 1598 | 0.05 | -4.7987 ± 0.4371 | +0.0185 | +2.20 | +0.6% | 3.3492 (+0.0002) |
| `matched_scale@0.05` | control | 1598 | 0.05 | -4.8098 ± 0.4392 | +0.0073 | +1.39 | +0.2% | 3.3491 (+0.0001) |
| `complement_scale@0.05` | control | 1598 | 0.05 | -4.8049 ± 0.4400 | +0.0122 | +1.61 | +0.4% | 3.3492 (+0.0002) |
| `circuit_scale_on_pos@0.05` | amplify | 1598 | 0.05 | -2.3382 ± 0.2575 | +0.0078 | +2.85 | +0.2% | 3.3491 (+0.0002) |
| `circuit_scale@0.1` | amplify | 1598 | 0.1 | -4.8099 ± 0.4366 | +0.0072 | +1.08 | +0.2% | 3.3499 (+0.0010) |
| `circuit_signed_scale@0.1` | amplify | 1598 | 0.1 | -4.8151 ± 0.4375 | +0.0020 | +0.33 | +0.1% | 3.3500 (+0.0011) |
| `dense_scale@0.1` | control | 3234 | 0.1 | -4.7810 ± 0.4387 | +0.0361 | +2.40 | +1.1% | 3.3514 (+0.0025) |
| `random_scale@0.1` | control | 1598 | 0.1 | -4.8051 ± 0.4386 | +0.0121 | +1.31 | +0.4% | 3.3503 (+0.0014) |
| `matched_scale@0.1` | control | 1598 | 0.1 | -4.8142 ± 0.4397 | +0.0030 | +0.45 | +0.1% | 3.3499 (+0.0010) |
| `complement_scale@0.1` | control | 1598 | 0.1 | -4.7901 ± 0.4409 | +0.0270 | +1.91 | +0.8% | 3.3503 (+0.0013) |
| `circuit_scale_on_pos@0.1` | amplify | 1598 | 0.1 | -2.3276 ± 0.2574 | +0.0183 | +3.30 | +0.5% | 3.3499 (+0.0010) |
| `circuit_scale@0.25` | amplify | 1598 | 0.25 | -4.8024 ± 0.4338 | +0.0148 | +0.93 | +0.4% | 3.3552 (+0.0063) |
| `circuit_signed_scale@0.25` | amplify | 1598 | 0.25 | -4.8144 ± 0.4370 | +0.0027 | +0.21 | +0.1% | 3.3572 (+0.0082) |
| `dense_scale@0.25` | control | 3234 | 0.25 | -4.7270 ± 0.4368 | +0.0901 | +2.47 | +2.7% | 3.3645 (+0.0155) |
| `random_scale@0.25` | control | 1598 | 0.25 | -4.7801 ± 0.4405 | +0.0371 | +2.59 | +1.1% | 3.3536 (+0.0046) |
| `matched_scale@0.25` | control | 1598 | 0.25 | -4.7657 ± 0.4485 | +0.0514 | +2.83 | +1.5% | 3.3542 (+0.0052) |
| `complement_scale@0.25` | control | 1598 | 0.25 | -4.7546 ± 0.4429 | +0.0625 | +1.92 | +1.9% | 3.3582 (+0.0092) |
| `circuit_scale_on_pos@0.25` | amplify | 1598 | 0.25 | -2.3084 ± 0.2596 | +0.0375 | +2.83 | +1.1% | 3.3552 (+0.0063) |
| `circuit_scale@0.5` | amplify | 1598 | 0.5 | -4.7889 ± 0.4311 | +0.0282 | +0.92 | +0.8% | 3.3736 (+0.0247) |
| `circuit_signed_scale@0.5` | amplify | 1598 | 0.5 | -4.7962 ± 0.4393 | +0.0209 | +0.79 | +0.6% | 3.3863 (+0.0373) |
| `dense_scale@0.5` | control | 3234 | 0.5 | -4.6489 ± 0.4343 | +0.1682 | +2.57 | +5.0% | 3.4147 (+0.0658) **damaged** |
| `random_scale@0.5` | control | 1598 | 0.5 | -4.8182 ± 0.4521 | -0.0011 | -0.04 | -0.0% | 3.3741 (+0.0252) |
| `matched_scale@0.5` | control | 1598 | 0.5 | -4.7372 ± 0.4375 | +0.0799 | +2.59 | +2.4% | 3.3758 (+0.0269) |
| `complement_scale@0.5` | control | 1598 | 0.5 | -4.7131 ± 0.4465 | +0.1040 | +1.93 | +3.1% | 3.3897 (+0.0408) |
| `circuit_scale_on_pos@0.5` | amplify | 1598 | 0.5 | -2.2727 ± 0.2642 | +0.0733 | +2.77 | +2.2% | 3.3736 (+0.0247) |
| `circuit_scale@1` | amplify | 1598 | 1 | -4.7584 ± 0.4277 | +0.0588 | +1.03 | +1.8% | 3.4480 (+0.0991) **damaged** |
| `circuit_signed_scale@1` | amplify | 1598 | 1 | -4.7026 ± 0.4480 | +0.1145 | +2.21 | +3.4% | 3.5329 (+0.1839) **damaged** |
| `dense_scale@1` | control | 3234 | 1 | -4.7155 ± 0.4505 | +0.1017 | +0.80 | +3.0% | 3.7046 (+0.3557) **damaged** |
| `random_scale@1` | control | 1598 | 1 | -4.6186 ± 0.4513 | +0.1986 | +3.39 | +5.9% | 3.4180 (+0.0691) **damaged** |
| `matched_scale@1` | control | 1598 | 1 | -4.6278 ± 0.4548 | +0.1893 | +2.78 | +5.6% | 3.4578 (+0.1088) **damaged** |
| `complement_scale@1` | control | 1598 | 1 | -4.8656 ± 0.4641 | -0.0485 | -0.41 | -1.4% | 3.5871 (+0.2382) **damaged** |
| `circuit_scale_on_pos@1` | amplify | 1598 | 1 | -2.2243 ± 0.2772 | +0.1216 | +2.30 | +3.6% | 3.4480 (+0.0991) **damaged** |
| `circuit_scale@3` | amplify | 1598 | 3 | -4.5907 ± 0.4367 | +0.2264 | +1.87 | +6.7% | 4.3579 (+1.0090) **damaged** |
| `circuit_signed_scale@3` | amplify | 1598 | 3 | -2.8875 ± 0.4566 | +1.9297 | +10.38 | +57.5% | 5.6624 (+2.3134) **damaged** |
| `dense_scale@3` | control | 3234 | 3 | -7.5452 ± 1.0314 | -2.7281 | -3.22 | -81.3% | 8.4380 (+5.0891) **damaged** |
| `random_scale@3` | control | 1598 | 3 | -10.6159 ± 1.3131 | -5.7988 | -4.26 | -172.9% | 6.2592 (+2.9102) **damaged** |
| `matched_scale@3` | control | 1598 | 3 | -3.9142 ± 0.4256 | +0.9029 | +8.67 | +26.9% | 4.4922 (+1.1432) **damaged** |
| `complement_scale@3` | control | 1598 | 3 | -7.5409 ± 0.6085 | -2.7238 | -4.98 | -81.2% | 6.6319 (+3.2829) **damaged** |
| `circuit_scale_on_pos@3` | amplify | 1598 | 3 | -2.2362 ± 0.3290 | +0.1098 | +0.79 | +3.3% | 4.3579 (+1.0090) **damaged** |
| `circuit_scale@9` | amplify | 1598 | 9 | -3.5899 ± 0.3492 | +1.2272 | +5.20 | +36.6% | 7.2618 (+3.9129) **damaged** |
| `circuit_signed_scale@9` | amplify | 1598 | 9 | -0.2369 ± 0.5129 | +4.5802 | +10.95 | +136.5% | 11.5206 (+8.1716) **damaged** |
| `dense_scale@9` | control | 3234 | 9 | -8.0264 ± 1.6810 | -3.2092 | -1.74 | -95.7% | 14.5629 (+11.2139) **damaged** |
| `random_scale@9` | control | 1598 | 9 | -8.5042 ± 0.9801 | -3.6871 | -2.96 | -109.9% | 13.7076 (+10.3587) **damaged** |
| `matched_scale@9` | control | 1598 | 9 | -2.2883 ± 0.3856 | +2.5288 | +11.88 | +75.4% | 8.3411 (+4.9921) **damaged** |
| `complement_scale@9` | control | 1598 | 9 | -2.9123 ± 0.7665 | +1.9048 | +2.84 | +56.8% | 13.5805 (+10.2316) **damaged** |
| `circuit_scale_on_pos@9` | amplify | 1598 | 9 | -2.3246 ± 0.4231 | +0.0214 | +0.07 | +0.6% | 7.2618 (+3.9129) **damaged** |
| `circuit_add_q@64` | stream | 481 | 64 | -3.6031 ± 0.4366 | +1.2141 | +9.43 | +36.2% | — |
| `circuit_add_k@64` | stream | 588 | 64 | -3.4713 ± 0.4618 | +1.3459 | +14.23 | +40.1% | — |
| `circuit_add_v@64` | stream | 500 | 64 | -3.2810 ± 0.2693 | +1.5362 | +4.68 | +45.8% | — |
| `circuit_add_r@64` | stream | 29 | 64 | +3.3202 ± 0.3462 | +8.1373 | +11.40 | +242.6% | — |
| `zero_circuit_on_pos` | necessity | 1598 | 0 | -2.6662 ± 0.2960 | -0.3202 | -4.89 | +9.5% | 3.6015 (+0.2526) **damaged** |
| `zero_random_on_pos` | control | 1598 | 0 | -2.6680 ± 0.2956 | -0.3220 | -4.08 | +9.6% | 3.7640 (+0.4151) **damaged** |

## How to read this

- If `dense_add` moves the behaviour as much as `circuit_add`, the circuit is not localised — the effect is whatever a bulk shift of the routing weights does.
- If `dense_add` and every other tested arm are flat at large λ, these directions do not establish useful steering for this behavior. Other directions, masks and tasks remain untested.
- `patch_pos_into_neg` reaching `ref_pos` while `patch_pos_into_neg_random` does not is the cleanest possible positive result using an observed donor magnitude. The full patch is a comparison intervention, not a bound on subset effects.
- q/k stream arms being flat while v/r move is expected, not a bug: q_norm/k_norm renormalise after mixing, so magnitude changes on those streams are partly undone.
- Any arm whose NLL rises sharply bought its behaviour shift by damaging the model; it is not steering.
