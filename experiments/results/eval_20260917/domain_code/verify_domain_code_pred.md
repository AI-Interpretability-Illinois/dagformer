# Causal verification — `domain_code` / channel `pred`

Circuit: 513 edges, rule `{"rule": "null", "q": 0.005, "s_cutoff": 1.8688310600682798, "expected_false_positives": 16.17, "n_null_draws_est": 8, "n_null_draws_cal": 8}`  
Intervening on the `pred` channel, scored on the **test** item half (8 items)  
Metric: mean logp(code) - logp(prose) per token  
Controls norm-matched; edits applied from the content span

## Verdict

- **steering: NULL — the best arm `circuit_signed_scale@1` shifts +0.0526, inside its own noise**
- **patch pos->neg: NULL — the best arm `patch_pos_into_neg` shifts +0.0021, inside its own noise**
- **patch neg->pos: NULL — the best arm `patch_neg_into_pos` shifts -0.0021, inside its own noise**
- **full patch: `patch_pos_into_neg_all` patches every eligible coordinate and moves +0.0032 = 0.1% of headroom; subset interventions can differ because effects may cancel**
- **necessity (knockout): NULL — the best arm `zero_circuit_on_pos` shifts -0.0413, inside its own noise**

- reference: pos `-2.3460`, neutral `-4.8171`, neg `-5.7007` → headroom `+3.3548`
- step-1 instruction sensitivity: `USABLE` (paired t = 15.98)
- circuit composition: 513 hyperconnections, 0 sequential
- on the circuit: mean `|alpha|` = `0.4546`, mean `|Delta|` = `0.005314` → `Delta/alpha` = `0.0117`. If `|alpha|` is tiny the multiplicative arms have almost nothing to enlarge, and a flat `circuit_scale` row says that rather than saying the circuit is inert.
- the additive arms reach parity with `|alpha|` at `lam = 85.6`; this sweep went to `64`
- over the whole channel `Delta/alpha` = `0.00808` — that is the entire effect the instruction has on the routing weights, and so the size of the perturbation the full patch arm applies
- arms are marked **damaged** when held-out NLL rises more than `0.05` nats; they are excluded from the verdict on both sides, since an arm that breaks the model neither steers it nor proves a circuit unspecific

`% headroom` is the shift as a fraction of the pos−neg gap, signed by intent: 100% means the edit reproduced the whole effect of changing the instruction, and a knockout reads positive when it does suppress the behaviour. `t` is a paired test over prompts against the arm's own baseline.

## Arms

| arm | group | edges | lam | score | shift vs base | t | % headroom | NLL |
|---|---|---|---|---|---|---|---|---|
| `ref_neutral` | reference | 0 | 0 | -4.8171 ± 0.4380 | +0.0000 | +0.00 | +0.0% | 3.3490 (+0.0000) |
| `ref_pos` | reference | 0 | 0 | -2.3460 ± 0.2571 | +2.4712 | +nan | +73.7% | 3.3490 (+0.0000) |
| `ref_neg` | reference | 0 | 0 | -5.7007 ± 0.3028 | -0.8836 | +nan | -26.3% | 3.3490 (+0.0000) |
| `circuit_add@0.05` | sufficiency | 513 | 0.05 | -4.8180 ± 0.4381 | -0.0009 | -0.43 | -0.0% | 3.3490 (+0.0000) |
| `dense_add@0.05` | control | 3234 | 0.05 | -4.8169 ± 0.4379 | +0.0003 | +0.10 | +0.0% | 3.3488 (-0.0001) |
| `random_add@0.05` | control | 513 | 0.05 | -4.8178 ± 0.4383 | -0.0006 | -0.25 | -0.0% | 3.3489 (-0.0001) |
| `matched_add@0.05` | control | 513 | 0.05 | -4.8164 ± 0.4391 | +0.0008 | +0.24 | +0.0% | 3.3489 (-0.0001) |
| `complement_add@0.05` | control | 513 | 0.05 | -4.8144 ± 0.4379 | +0.0027 | +1.07 | +0.1% | 3.3490 (+0.0001) |
| `circuit_add@0.1` | sufficiency | 513 | 0.1 | -4.8179 ± 0.4382 | -0.0007 | -0.29 | -0.0% | 3.3489 (-0.0000) |
| `dense_add@0.1` | control | 3234 | 0.1 | -4.8193 ± 0.4387 | -0.0021 | -0.70 | -0.1% | 3.3490 (+0.0000) |
| `random_add@0.1` | control | 513 | 0.1 | -4.8129 ± 0.4389 | +0.0042 | +1.77 | +0.1% | 3.3490 (+0.0000) |
| `matched_add@0.1` | control | 513 | 0.1 | -4.8158 ± 0.4382 | +0.0013 | +0.57 | +0.0% | 3.3489 (-0.0001) |
| `complement_add@0.1` | control | 513 | 0.1 | -4.8190 ± 0.4379 | -0.0018 | -0.73 | -0.1% | 3.3490 (+0.0001) |
| `circuit_add@0.25` | sufficiency | 513 | 0.25 | -4.8194 ± 0.4376 | -0.0023 | -0.96 | -0.1% | 3.3489 (-0.0001) |
| `dense_add@0.25` | control | 3234 | 0.25 | -4.8201 ± 0.4388 | -0.0029 | -0.89 | -0.1% | 3.3488 (-0.0002) |
| `random_add@0.25` | control | 513 | 0.25 | -4.8175 ± 0.4384 | -0.0004 | -0.16 | -0.0% | 3.3489 (-0.0000) |
| `matched_add@0.25` | control | 513 | 0.25 | -4.8196 ± 0.4381 | -0.0024 | -0.66 | -0.1% | 3.3489 (-0.0001) |
| `complement_add@0.25` | control | 513 | 0.25 | -4.8172 ± 0.4371 | -0.0001 | -0.02 | -0.0% | 3.3490 (+0.0000) |
| `circuit_add@0.5` | sufficiency | 513 | 0.5 | -4.8202 ± 0.4379 | -0.0030 | -1.33 | -0.1% | 3.3488 (-0.0001) |
| `dense_add@0.5` | control | 3234 | 0.5 | -4.8210 ± 0.4369 | -0.0039 | -1.12 | -0.1% | 3.3489 (-0.0000) |
| `random_add@0.5` | control | 513 | 0.5 | -4.8216 ± 0.4386 | -0.0044 | -1.69 | -0.1% | 3.3489 (-0.0001) |
| `matched_add@0.5` | control | 513 | 0.5 | -4.8169 ± 0.4380 | +0.0002 | +0.08 | +0.0% | 3.3489 (-0.0001) |
| `complement_add@0.5` | control | 513 | 0.5 | -4.8189 ± 0.4375 | -0.0017 | -0.48 | -0.1% | 3.3490 (+0.0001) |
| `circuit_add@1` | sufficiency | 513 | 1 | -4.8216 ± 0.4382 | -0.0045 | -2.13 | -0.1% | 3.3488 (-0.0001) |
| `dense_add@1` | control | 3234 | 1 | -4.8218 ± 0.4370 | -0.0046 | -1.45 | -0.1% | 3.3489 (-0.0001) |
| `random_add@1` | control | 513 | 1 | -4.8151 ± 0.4372 | +0.0020 | +0.68 | +0.1% | 3.3489 (-0.0001) |
| `matched_add@1` | control | 513 | 1 | -4.8225 ± 0.4379 | -0.0054 | -1.87 | -0.2% | 3.3489 (-0.0001) |
| `complement_add@1` | control | 513 | 1 | -4.8155 ± 0.4387 | +0.0016 | +0.73 | +0.0% | 3.3489 (-0.0000) |
| `circuit_add@4` | sufficiency | 513 | 4 | -4.8194 ± 0.4381 | -0.0023 | -0.91 | -0.1% | 3.3489 (-0.0001) |
| `dense_add@4` | control | 3234 | 4 | -4.8188 ± 0.4375 | -0.0017 | -0.46 | -0.0% | 3.3491 (+0.0002) |
| `random_add@4` | control | 513 | 4 | -4.8214 ± 0.4377 | -0.0042 | -1.24 | -0.1% | 3.3489 (-0.0000) |
| `matched_add@4` | control | 513 | 4 | -4.8195 ± 0.4377 | -0.0024 | -0.85 | -0.1% | 3.3490 (+0.0001) |
| `complement_add@4` | control | 513 | 4 | -4.8162 ± 0.4372 | +0.0009 | +0.28 | +0.0% | 3.3490 (+0.0000) |
| `circuit_add@16` | sufficiency | 513 | 16 | -4.8304 ± 0.4375 | -0.0132 | -2.70 | -0.4% | 3.3496 (+0.0007) |
| `dense_add@16` | control | 3234 | 16 | -4.8326 ± 0.4355 | -0.0154 | -1.76 | -0.5% | 3.3515 (+0.0026) |
| `random_add@16` | control | 513 | 16 | -4.8474 ± 0.4360 | -0.0303 | -4.08 | -0.9% | 3.3501 (+0.0012) |
| `matched_add@16` | control | 513 | 16 | -4.8203 ± 0.4377 | -0.0032 | -0.75 | -0.1% | 3.3496 (+0.0007) |
| `complement_add@16` | control | 513 | 16 | -4.8089 ± 0.4361 | +0.0083 | +1.83 | +0.2% | 3.3494 (+0.0004) |
| `circuit_add@64` | sufficiency | 513 | 64 | -4.8486 ± 0.4366 | -0.0314 | -1.83 | -0.9% | 3.3620 (+0.0131) |
| `dense_add@64` | control | 3234 | 64 | -4.8145 ± 0.4235 | +0.0027 | +0.09 | +0.1% | 3.3863 (+0.0373) |
| `random_add@64` | control | 513 | 64 | -4.7679 ± 0.4247 | +0.0493 | +2.06 | +1.5% | 3.3537 (+0.0047) |
| `matched_add@64` | control | 513 | 64 | -4.8322 ± 0.4267 | -0.0151 | -0.90 | -0.5% | 3.3596 (+0.0107) |
| `complement_add@64` | control | 513 | 64 | -4.7900 ± 0.4293 | +0.0272 | +1.65 | +0.8% | 3.3546 (+0.0056) |
| `circuit_scale@0.05` | amplify | 513 | 0.05 | -4.8192 ± 0.4381 | -0.0020 | -0.78 | -0.1% | 3.3488 (-0.0002) |
| `circuit_signed_scale@0.05` | amplify | 513 | 0.05 | -4.8139 ± 0.4377 | +0.0032 | +1.33 | +0.1% | 3.3489 (-0.0001) |
| `dense_scale@0.05` | control | 3234 | 0.05 | -4.8311 ± 0.4361 | -0.0140 | -3.07 | -0.4% | 3.3496 (+0.0006) |
| `random_scale@0.05` | control | 513 | 0.05 | -4.8201 ± 0.4376 | -0.0030 | -0.88 | -0.1% | 3.3488 (-0.0002) |
| `matched_scale@0.05` | control | 513 | 0.05 | -4.8213 ± 0.4382 | -0.0042 | -1.29 | -0.1% | 3.3489 (-0.0001) |
| `complement_scale@0.05` | control | 513 | 0.05 | -4.8143 ± 0.4376 | +0.0028 | +0.85 | +0.1% | 3.3489 (-0.0000) |
| `circuit_scale_on_pos@0.05` | amplify | 513 | 0.05 | -2.3447 ± 0.2572 | +0.0013 | +0.60 | +0.0% | 3.3488 (-0.0002) |
| `circuit_scale@0.1` | amplify | 513 | 0.1 | -4.8132 ± 0.4387 | +0.0039 | +1.29 | +0.1% | 3.3489 (-0.0000) |
| `circuit_signed_scale@0.1` | amplify | 513 | 0.1 | -4.8156 ± 0.4384 | +0.0015 | +0.45 | +0.0% | 3.3489 (-0.0001) |
| `dense_scale@0.1` | control | 3234 | 0.1 | -4.8414 ± 0.4367 | -0.0243 | -3.35 | -0.7% | 3.3507 (+0.0018) |
| `random_scale@0.1` | control | 513 | 0.1 | -4.8253 ± 0.4365 | -0.0082 | -2.10 | -0.2% | 3.3494 (+0.0004) |
| `matched_scale@0.1` | control | 513 | 0.1 | -4.8140 ± 0.4377 | +0.0032 | +0.78 | +0.1% | 3.3490 (+0.0000) |
| `complement_scale@0.1` | control | 513 | 0.1 | -4.8038 ± 0.4376 | +0.0133 | +2.85 | +0.4% | 3.3492 (+0.0002) |
| `circuit_scale_on_pos@0.1` | amplify | 513 | 0.1 | -2.3445 ± 0.2569 | +0.0015 | +0.71 | +0.0% | 3.3489 (-0.0000) |
| `circuit_scale@0.25` | amplify | 513 | 0.25 | -4.8067 ± 0.4379 | +0.0104 | +1.82 | +0.3% | 3.3499 (+0.0010) |
| `circuit_signed_scale@0.25` | amplify | 513 | 0.25 | -4.8045 ± 0.4393 | +0.0126 | +2.12 | +0.4% | 3.3495 (+0.0006) |
| `dense_scale@0.25` | control | 3234 | 0.25 | -4.8663 ± 0.4367 | -0.0492 | -3.68 | -1.5% | 3.3583 (+0.0093) |
| `random_scale@0.25` | control | 513 | 0.25 | -4.8092 ± 0.4389 | +0.0080 | +1.61 | +0.2% | 3.3496 (+0.0007) |
| `matched_scale@0.25` | control | 513 | 0.25 | -4.8204 ± 0.4371 | -0.0033 | -0.73 | -0.1% | 3.3500 (+0.0010) |
| `complement_scale@0.25` | control | 513 | 0.25 | -4.7922 ± 0.4368 | +0.0249 | +4.04 | +0.7% | 3.3499 (+0.0009) |
| `circuit_scale_on_pos@0.25` | amplify | 513 | 0.25 | -2.3381 ± 0.2556 | +0.0079 | +1.68 | +0.2% | 3.3499 (+0.0010) |
| `circuit_scale@0.5` | amplify | 513 | 0.5 | -4.7909 ± 0.4390 | +0.0262 | +2.33 | +0.8% | 3.3529 (+0.0040) |
| `circuit_signed_scale@0.5` | amplify | 513 | 0.5 | -4.7912 ± 0.4414 | +0.0259 | +1.98 | +0.8% | 3.3524 (+0.0035) |
| `dense_scale@0.5` | control | 3234 | 0.5 | -4.9040 ± 0.4361 | -0.0869 | -3.44 | -2.6% | 3.3814 (+0.0324) |
| `random_scale@0.5` | control | 513 | 0.5 | -4.7759 ± 0.4353 | +0.0413 | +3.87 | +1.2% | 3.3527 (+0.0037) |
| `matched_scale@0.5` | control | 513 | 0.5 | -4.8385 ± 0.4435 | -0.0214 | -2.76 | -0.6% | 3.3509 (+0.0020) |
| `complement_scale@0.5` | control | 513 | 0.5 | -4.7718 ± 0.4325 | +0.0454 | +3.57 | +1.4% | 3.3526 (+0.0036) |
| `circuit_scale_on_pos@0.5` | amplify | 513 | 0.5 | -2.3307 ± 0.2543 | +0.0153 | +1.82 | +0.5% | 3.3529 (+0.0040) |
| `circuit_scale@1` | amplify | 513 | 1 | -4.7671 ± 0.4402 | +0.0500 | +2.40 | +1.5% | 3.3647 (+0.0158) |
| `circuit_signed_scale@1` | amplify | 513 | 1 | -4.7646 ± 0.4441 | +0.0526 | +2.07 | +1.6% | 3.3638 (+0.0148) |
| `dense_scale@1` | control | 3234 | 1 | -4.9696 ± 0.4387 | -0.1524 | -3.41 | -4.5% | 3.4629 (+0.1140) **damaged** |
| `random_scale@1` | control | 513 | 1 | -4.9410 ± 0.4432 | -0.1239 | -5.62 | -3.7% | 3.3771 (+0.0281) |
| `matched_scale@1` | control | 513 | 1 | -4.7912 ± 0.4376 | +0.0259 | +1.67 | +0.8% | 3.3576 (+0.0086) |
| `complement_scale@1` | control | 513 | 1 | -4.7329 ± 0.4265 | +0.0842 | +3.58 | +2.5% | 3.3609 (+0.0119) |
| `circuit_scale_on_pos@1` | amplify | 513 | 1 | -2.3184 ± 0.2524 | +0.0276 | +1.68 | +0.8% | 3.3647 (+0.0158) |
| `circuit_scale@3` | amplify | 513 | 3 | -4.6485 ± 0.4404 | +0.1686 | +3.16 | +5.0% | 3.4658 (+0.1168) **damaged** |
| `circuit_signed_scale@3` | amplify | 513 | 3 | -4.6678 ± 0.4500 | +0.1493 | +2.30 | +4.5% | 3.4805 (+0.1315) **damaged** |
| `dense_scale@3` | control | 3234 | 3 | -4.9315 ± 0.4668 | -0.1143 | -0.98 | -3.4% | 4.1643 (+0.8153) **damaged** |
| `random_scale@3` | control | 513 | 3 | -4.8412 ± 0.4527 | -0.0241 | -0.62 | -0.7% | 3.4682 (+0.1193) **damaged** |
| `matched_scale@3` | control | 513 | 3 | -4.7804 ± 0.4467 | +0.0367 | +0.95 | +1.1% | 3.4272 (+0.0783) **damaged** |
| `complement_scale@3` | control | 513 | 3 | -4.5547 ± 0.4072 | +0.2625 | +4.26 | +7.8% | 3.4210 (+0.0720) **damaged** |
| `circuit_scale_on_pos@3` | amplify | 513 | 3 | -2.2629 ± 0.2500 | +0.0831 | +1.75 | +2.5% | 3.4658 (+0.1168) **damaged** |
| `circuit_scale@9` | amplify | 513 | 9 | -4.2777 ± 0.4321 | +0.5394 | +4.41 | +16.1% | 4.2752 (+0.9262) **damaged** |
| `circuit_signed_scale@9` | amplify | 513 | 9 | -4.1384 ± 0.4246 | +0.6787 | +5.55 | +20.2% | 4.5887 (+1.2397) **damaged** |
| `dense_scale@9` | control | 3234 | 9 | -3.1735 ± 0.2670 | +1.6436 | +6.92 | +49.0% | 6.9418 (+3.5928) **damaged** |
| `random_scale@9` | control | 513 | 9 | -1.2093 ± 0.3023 | +3.6078 | +9.43 | +107.5% | 9.1744 (+5.8255) **damaged** |
| `matched_scale@9` | control | 513 | 9 | -4.3581 ± 0.4090 | +0.4591 | +5.52 | +13.7% | 3.8721 (+0.5231) **damaged** |
| `complement_scale@9` | control | 513 | 9 | -4.0256 ± 0.3550 | +0.7915 | +4.89 | +23.6% | 3.8728 (+0.5238) **damaged** |
| `circuit_scale_on_pos@9` | amplify | 513 | 9 | -1.9870 ± 0.2719 | +0.3590 | +2.98 | +10.7% | 4.2752 (+0.9262) **damaged** |
| `circuit_add_q@64` | stream | 144 | 64 | -4.8419 ± 0.4373 | -0.0248 | -3.71 | -0.7% | — |
| `circuit_add_k@64` | stream | 159 | 64 | -4.7800 ± 0.4366 | +0.0371 | +3.85 | +1.1% | — |
| `circuit_add_v@64` | stream | 208 | 64 | -4.8672 ± 0.4409 | -0.0501 | -4.09 | -1.5% | — |
| `circuit_add_r@64` | stream | 2 | 64 | -4.8077 ± 0.4376 | +0.0095 | +1.62 | +0.3% | — |
| `zero_circuit_on_pos` | necessity | 513 | 0 | -2.3873 ± 0.2620 | -0.0413 | -1.95 | +1.2% | 3.3819 (+0.0330) |
| `zero_random_on_pos` | control | 513 | 0 | -2.2243 ± 0.2538 | +0.1216 | +5.20 | -3.6% | 3.3761 (+0.0271) |
| `patch_pos_into_neg` | patch | 513 | 0 | -5.6987 ± 0.3021 | +0.0021 | +0.84 | +0.1% | — |
| `patch_pos_into_neg_random` | control | 513 | 0 | -5.6975 ± 0.3027 | +0.0032 | +1.56 | +0.1% | — |
| `patch_pos_into_neg_all` | patch | 3234 | 0 | -5.6975 ± 0.3032 | +0.0032 | +1.56 | +0.1% | — |
| `patch_neg_into_pos` | patch | 513 | 0 | -2.3481 ± 0.2569 | -0.0021 | -1.09 | +0.1% | — |
| `patch_neg_into_pos_random` | control | 513 | 0 | -2.3475 ± 0.2571 | -0.0015 | -0.76 | +0.0% | — |
| `patch_neg_into_pos_all` | patch | 3234 | 0 | -2.3465 ± 0.2573 | -0.0005 | -0.19 | +0.0% | — |

## How to read this

- If `dense_add` moves the behaviour as much as `circuit_add`, the circuit is not localised — the effect is whatever a bulk shift of the routing weights does.
- If `dense_add` and every other tested arm are flat at large λ, these directions do not establish useful steering for this behavior. Other directions, masks and tasks remain untested.
- `patch_pos_into_neg` reaching `ref_pos` while `patch_pos_into_neg_random` does not is the cleanest possible positive result using an observed donor magnitude. The full patch is a comparison intervention, not a bound on subset effects.
- q/k stream arms being flat while v/r move is expected, not a bug: q_norm/k_norm renormalise after mixing, so magnitude changes on those streams are partly undone.
- Any arm whose NLL rises sharply bought its behaviour shift by damaging the model; it is not steering.
