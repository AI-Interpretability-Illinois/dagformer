# DAGFormer vs baseline — lm-evaluation-harness

- suite: `reasoning` · lm-eval 0.4.13 · context 1024 tokens
- generative tasks limited to 200 docs at 3-shot; log-likelihood tasks use the full split (limit None)
- `*` = |Δ| exceeds twice the quadrature-combined standard error
- for `bits_per_byte` lower is better, so Δ is sign-corrected: + always favours DAGFormer

| model | params | train steps | eval seconds | peak GPU GB |
|---|---|---|---|---|
| 150m-baseline | 152,593,152 | 6000 | 550 | 6.55 |
| 150m-dagformer | 182,561,679 | 6000 | 756 | 6.66 |
| 300m-baseline | 304,137,216 | 12000 | 744 | 6.83 |
| 300m-dagformer | 336,447,805 | 9000 | 1019 | 6.95 |
| 600m-baseline | 682,710,528 | 22900 | 878 | 7.53 |
| 75m-baseline | 76,558,848 | 3000 | 571 | 6.4 |
| 75m-dagformer | 105,657,332 | 3000 | 606 | 6.51 |

> **Not step-matched:** 300m: baseline 12000 steps, dagformer 9000 steps. Token counts also depend on the effective batch size. This comparison does not establish the effect at a matched training budget.

### 75m

| task | metric | baseline | dagformer | Δ (dagformer better = +) |
|---|---|---|---|---|
| gsm8k_bpb | `bits_per_byte,none` | 1.8396 | 1.7591 | +0.0805 |
| arc_challenge | `acc_norm,none` | 0.2167 ± 0.0120 | 0.2048 ± 0.0118 | -0.0119 |
| arc_easy | `acc_norm,none` | 0.3018 ± 0.0094 | 0.3018 ± 0.0094 | +0.0000 |
| mathqa | `acc_norm,none` | 0.2020 ± 0.0073 | 0.2090 ± 0.0074 | +0.0070 |
| commonsense_qa | `acc,none` | 0.1957 ± 0.0114 | 0.1957 ± 0.0114 | +0.0000 |
| social_iqa | `acc,none` | 0.3403 ± 0.0107 | 0.3414 ± 0.0107 | +0.0010 |
| openbookqa | `acc_norm,none` | 0.2440 ± 0.0192 | 0.2500 ± 0.0194 | +0.0060 |
| winogrande | `acc,none` | 0.5036 ± 0.0141 | 0.5107 ± 0.0140 | +0.0071 |
| gsm8k | `exact_match,flexible-extract` | 0.0100 ± 0.0071 | 0.0250 ± 0.0111 | +0.0150 |

### 150m

| task | metric | baseline | dagformer | Δ (dagformer better = +) |
|---|---|---|---|---|
| gsm8k_bpb | `bits_per_byte,none` | 1.5927 | 1.5186 | +0.0741 |
| arc_challenge | `acc_norm,none` | 0.2142 ± 0.0120 | 0.2244 ± 0.0122 | +0.0102 |
| arc_easy | `acc_norm,none` | 0.3359 ± 0.0097 | 0.3472 ± 0.0098 | +0.0114 |
| mathqa | `acc_norm,none` | 0.2010 ± 0.0073 | 0.2221 ± 0.0076 | +0.0211 |
| commonsense_qa | `acc,none` | 0.1974 ± 0.0114 | 0.1966 ± 0.0114 | -0.0008 |
| social_iqa | `acc,none` | 0.3429 ± 0.0107 | 0.3639 ± 0.0109 | +0.0210 |
| openbookqa | `acc_norm,none` | 0.2760 ± 0.0200 | 0.2560 ± 0.0195 | -0.0200 |
| winogrande | `acc,none` | 0.5028 ± 0.0141 | 0.5280 ± 0.0140 | +0.0253 |
| gsm8k | `exact_match,flexible-extract` | 0.0100 ± 0.0071 | 0.0050 ± 0.0050 | -0.0050 |

### 300m

| task | metric | baseline | dagformer | Δ (dagformer better = +) |
|---|---|---|---|---|
| gsm8k_bpb | `bits_per_byte,none` | 1.4019 | 1.3315 | +0.0705 |
| arc_challenge | `acc_norm,none` | 0.2167 ± 0.0120 | 0.2329 ± 0.0124 | +0.0162 |
| arc_easy | `acc_norm,none` | 0.3678 ± 0.0099 | 0.3826 ± 0.0100 | +0.0147 |
| mathqa | `acc_norm,none` | 0.2161 ± 0.0075 | 0.2265 ± 0.0077 | +0.0104 |
| commonsense_qa | `acc,none` | 0.1990 ± 0.0114 | 0.1957 ± 0.0114 | -0.0033 |
| social_iqa | `acc,none` | 0.3623 ± 0.0109 | 0.3613 ± 0.0109 | -0.0010 |
| openbookqa | `acc_norm,none` | 0.2740 ± 0.0200 | 0.2760 ± 0.0200 | +0.0020 |
| winogrande | `acc,none` | 0.5130 ± 0.0140 | 0.5217 ± 0.0140 | +0.0087 |
| gsm8k | `exact_match,flexible-extract` | 0.0250 ± 0.0111 | 0.0150 ± 0.0086 | -0.0100 |

### 600m

_only `600m-baseline` evaluated at this size_

| task | metric | baseline |
|---|---|---|
| gsm8k_bpb | `bits_per_byte,none` | 1.2491 |
| arc_challenge | `acc_norm,none` | 0.2372 ± 0.0124 |
| arc_easy | `acc_norm,none` | 0.4057 ± 0.0101 |
| mathqa | `acc_norm,none` | 0.2214 ± 0.0076 |
| commonsense_qa | `acc,none` | 0.1966 ± 0.0114 |
| social_iqa | `acc,none` | 0.3889 ± 0.0110 |
| openbookqa | `acc_norm,none` | 0.2800 ± 0.0201 |
| winogrande | `acc,none` | 0.5114 ± 0.0140 |
| gsm8k | `exact_match,flexible-extract` | 0.0200 ± 0.0099 |

**All task types combined:** DAGFormer better on 18 of 27 paired task-size cells, baseline on 7, 2 tied (sign test over the 25 non-tied cells, descriptive p = 0.043). This includes BPB and generation tasks.

Task-size cells reuse model pairs and are not independent; the sign test is a descriptive summary, not a calibrated significance claim.

**Reasoning multiple choice only:** 14 wins, 5 losses, 2 ties (descriptive sign-test p = 0.064); excludes GSM8K exact-match and BPB.
