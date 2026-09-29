# Capability-gated routing rule interchange

Model: checkpoints/pr_sync_20260917/300m-dagformer. This is a frozen-model inference experiment; no adapter or backbone training.

## Capability screen

The selected task and rule pair must have at least 80% full-vocabulary accuracy under both rules and 70% joint success on an independent screen. Candidate-only accuracy does not bypass this gate. All screened templates are retained.

| Template | Rule 0 accuracy | Rule 1 accuracy | Rule 2 accuracy | Eligible pairs |
|---|---:|---:|---:|---|
| arrows_0shot | 0.00% | 0.00% | 0.00% | [] |
| arrows_3shot | 6.25% | 0.00% | 12.50% | [] |
| arrows_6shot | 12.50% | 6.25% | 25.00% | [] |
| function_0shot | 0.00% | 0.00% | 0.00% | [] |
| function_3shot | 6.25% | 0.00% | 6.25% | [] |
| function_6shot | 0.00% | 12.50% | 25.00% | [] |
| chain_0shot | 0.00% | 0.00% | 0.00% | [] |
| chain_3shot | 18.75% | 0.00% | 6.25% | [] |
| chain_6shot | 93.75% | 0.00% | 0.00% | [] |
| index_0shot | 0.00% | 0.00% | 0.00% | [] |
| index_3shot | 6.25% | 0.00% | 6.25% | [] |
| index_6shot | 6.25% | 0.00% | 12.50% | [] |
| copy_query_0shot | 100.00% | 100.00% | 93.75% | ['0-1', '1-2', '0-2'] |

Selected: {'variant': 'copy_query_0shot', 'rules': [0, 1], 'family': 'copy_query', 'shots': 0}.

None of the screened transformation templates met the two-rule capability gate. The fallback changes which repeated key is queried. It tests address-conditioned routing transfer, not switching a learned multi-step algorithm; failure here does not establish that routing cannot implement algorithm control.

## Desired versus wrong donor rule

The central contrast holds donor content fixed and changes only its rule. An increase in desired-versus-original margin by itself can reflect reduced confidence in the original answer; the wrong-rule and reversed-difference controls distinguish that from directed operation transfer.

| Split | Query/both: correct minus wrong donor margin [95% CI] | Query/both: forward minus reversed rule-difference margin [95% CI] | Desired-answer accuracy, correct / wrong |
|---|---:|---:|---:|
| discovery | +0.203 [-0.251, +0.722] | +0.534 [-0.257, +1.312] | 0.00% / 1.56% |
| heldout | -0.215 [-0.602, +0.133] | -0.254 [-0.720, +0.195] | 0.00% / 0.00% |
| transfer | -0.304 [-0.727, +0.080] | +0.163 [-0.364, +0.669] | 0.00% / 1.56% |

The experiment uses direct donor values or a donor rule difference at scale 1; it does not search amplified gains or train a new routing controller.

## discovery

Original-rule full-vocabulary accuracy 98.44%; desired-rule reference 98.44%; both rules correct 96.88%. Identity max logit difference 0.0.

| Arm | Desired-answer full-vocab accuracy | Desired−original margin change [95% CI] | Margin recovery | Cross-content desired-rule donor token accuracy |
|---|---:|---:|---:|---:|
| query/pred/same | 0.00% | -0.003 [-0.019, +0.014] | -0.0001 | 0.00% |
| query/pred/cross | 0.00% | -0.010 [-0.030, +0.008] | -0.0004 | 0.00% |
| query/pred/wrong | 0.00% | +0.001 [-0.013, +0.015] | 0.0001 | 0.00% |
| query/pred/random | 0.00% | -0.013 [-0.028, +0.003] | -0.0005 | 0.00% |
| query/pred/cross_delta | 0.00% | -0.008 [-0.016, -0.000] | -0.0003 | 0.00% |
| query/pred/reverse_delta | 0.00% | +0.005 [-0.007, +0.017] | 0.0002 | 0.00% |
| test/pred/same | 0.00% | -0.003 [-0.019, +0.014] | -0.0001 | 0.00% |
| test/pred/cross | 0.00% | +0.010 [-0.011, +0.031] | 0.0004 | 0.00% |
| test/pred/wrong | 0.00% | +0.032 [+0.010, +0.055] | 0.0013 | 0.00% |
| test/pred/random | 0.00% | +0.002 [-0.028, +0.033] | 0.0001 | 0.00% |
| test/pred/cross_delta | 0.00% | -0.008 [-0.016, -0.000] | -0.0003 | 0.00% |
| test/pred/reverse_delta | 0.00% | +0.005 [-0.007, +0.017] | 0.0002 | 0.00% |
| query/corr/same | 0.00% | +1.303 [+0.613, +2.122] | 0.0542 | 0.00% |
| query/corr/cross | 0.00% | +1.527 [+1.028, +2.093] | 0.0635 | 0.00% |
| query/corr/wrong | 0.00% | +1.344 [+0.740, +2.000] | 0.0559 | 0.00% |
| query/corr/random | 1.56% | +1.559 [+0.836, +2.309] | 0.0648 | 0.00% |
| query/corr/cross_delta | 0.00% | +2.802 [+1.975, +3.706] | 0.1165 | 0.00% |
| query/corr/reverse_delta | 0.00% | +2.281 [+1.645, +2.981] | 0.0948 | 0.00% |
| test/corr/same | 0.00% | +1.303 [+0.613, +2.122] | 0.0542 | 0.00% |
| test/corr/cross | 0.00% | +2.182 [+1.589, +2.840] | 0.0907 | 0.00% |
| test/corr/wrong | 0.00% | +1.006 [+0.508, +1.492] | 0.0418 | 0.00% |
| test/corr/random | 0.00% | +1.956 [+1.132, +2.764] | 0.0813 | 0.00% |
| test/corr/cross_delta | 0.00% | +2.802 [+1.975, +3.706] | 0.1165 | 0.00% |
| test/corr/reverse_delta | 0.00% | +2.281 [+1.645, +2.981] | 0.0948 | 0.00% |
| query/both/same | 0.00% | +1.328 [+0.632, +2.163] | 0.0552 | 0.00% |
| query/both/cross | 0.00% | +1.580 [+1.067, +2.166] | 0.0657 | 0.00% |
| query/both/wrong | 1.56% | +1.377 [+0.775, +2.032] | 0.0572 | 0.00% |
| query/both/random | 1.56% | +1.616 [+0.880, +2.387] | 0.0671 | 0.00% |
| query/both/cross_delta | 0.00% | +2.842 [+1.994, +3.778] | 0.1181 | 0.00% |
| query/both/reverse_delta | 0.00% | +2.307 [+1.676, +2.998] | 0.0959 | 0.00% |
| test/both/same | 0.00% | +1.328 [+0.632, +2.163] | 0.0552 | 0.00% |
| test/both/cross | 0.00% | +2.238 [+1.650, +2.889] | 0.0930 | 0.00% |
| test/both/wrong | 0.00% | +1.062 [+0.565, +1.554] | 0.0442 | 0.00% |
| test/both/random | 0.00% | +1.991 [+1.186, +2.783] | 0.0827 | 0.00% |
| test/both/cross_delta | 0.00% | +2.842 [+1.994, +3.778] | 0.1181 | 0.00% |
| test/both/reverse_delta | 0.00% | +2.307 [+1.676, +2.998] | 0.0959 | 0.00% |
| reference | 0.00% | +0.000 [+0.000, +0.000] | 0.0000 | 0.00% |
| target_rule_reference | 98.44% | +24.062 [+22.052, +26.028] | 1.0000 | 0.00% |

## heldout

Original-rule full-vocabulary accuracy 97.66%; desired-rule reference 97.66%; both rules correct 95.31%. Identity max logit difference 0.0.

| Arm | Desired-answer full-vocab accuracy | Desired−original margin change [95% CI] | Margin recovery | Cross-content desired-rule donor token accuracy |
|---|---:|---:|---:|---:|
| query/pred/same | 0.00% | -0.015 [-0.028, -0.004] | -0.0006 | 0.00% |
| query/pred/cross | 0.00% | -0.018 [-0.037, -0.000] | -0.0007 | 0.00% |
| query/pred/wrong | 0.00% | -0.018 [-0.039, +0.001] | -0.0007 | 0.00% |
| query/pred/random | 0.00% | -0.017 [-0.034, -0.002] | -0.0007 | 0.00% |
| query/pred/cross_delta | 0.00% | -0.002 [-0.012, +0.009] | -0.0001 | 0.00% |
| query/pred/reverse_delta | 0.00% | +0.003 [-0.006, +0.012] | 0.0001 | 0.00% |
| test/pred/same | 0.00% | -0.015 [-0.028, -0.004] | -0.0006 | 0.00% |
| test/pred/cross | 0.00% | -0.008 [-0.028, +0.011] | -0.0003 | 0.00% |
| test/pred/wrong | 0.00% | -0.002 [-0.024, +0.019] | -0.0001 | 0.00% |
| test/pred/random | 0.00% | -0.004 [-0.031, +0.019] | -0.0002 | 0.00% |
| test/pred/cross_delta | 0.00% | -0.002 [-0.012, +0.009] | -0.0001 | 0.00% |
| test/pred/reverse_delta | 0.00% | +0.003 [-0.006, +0.012] | 0.0001 | 0.00% |
| query/corr/same | 0.00% | +1.566 [+1.149, +2.012] | 0.0661 | 0.00% |
| query/corr/cross | 0.00% | +1.374 [+1.085, +1.671] | 0.0580 | 0.00% |
| query/corr/wrong | 0.00% | +1.611 [+1.229, +2.023] | 0.0680 | 0.00% |
| query/corr/random | 0.78% | +1.600 [+1.095, +2.149] | 0.0675 | 0.00% |
| query/corr/cross_delta | 0.00% | +2.280 [+1.914, +2.659] | 0.0962 | 0.00% |
| query/corr/reverse_delta | 0.00% | +2.532 [+2.038, +3.058] | 0.1068 | 0.00% |
| test/corr/same | 0.00% | +1.566 [+1.149, +2.012] | 0.0661 | 0.00% |
| test/corr/cross | 0.00% | +2.061 [+1.634, +2.489] | 0.0870 | 0.78% |
| test/corr/wrong | 0.78% | +1.118 [+0.778, +1.468] | 0.0472 | 0.00% |
| test/corr/random | 0.78% | +2.166 [+1.660, +2.653] | 0.0914 | 0.00% |
| test/corr/cross_delta | 0.00% | +2.280 [+1.914, +2.659] | 0.0962 | 0.00% |
| test/corr/reverse_delta | 0.00% | +2.532 [+2.038, +3.058] | 0.1068 | 0.00% |
| query/both/same | 0.00% | +1.606 [+1.173, +2.070] | 0.0678 | 0.00% |
| query/both/cross | 0.00% | +1.398 [+1.107, +1.695] | 0.0590 | 0.00% |
| query/both/wrong | 0.00% | +1.613 [+1.224, +2.031] | 0.0681 | 0.00% |
| query/both/random | 0.78% | +1.640 [+1.126, +2.189] | 0.0692 | 0.00% |
| query/both/cross_delta | 0.00% | +2.302 [+1.932, +2.687] | 0.0971 | 0.00% |
| query/both/reverse_delta | 0.00% | +2.556 [+2.065, +3.083] | 0.1079 | 0.00% |
| test/both/same | 0.00% | +1.606 [+1.173, +2.070] | 0.0678 | 0.00% |
| test/both/cross | 0.00% | +2.088 [+1.652, +2.524] | 0.0881 | 0.78% |
| test/both/wrong | 0.78% | +1.146 [+0.793, +1.510] | 0.0484 | 0.00% |
| test/both/random | 0.78% | +2.217 [+1.696, +2.712] | 0.0935 | 0.00% |
| test/both/cross_delta | 0.00% | +2.302 [+1.932, +2.687] | 0.0971 | 0.00% |
| test/both/reverse_delta | 0.00% | +2.556 [+2.065, +3.083] | 0.1079 | 0.00% |
| reference | 0.00% | +0.000 [+0.000, +0.000] | 0.0000 | 0.00% |
| target_rule_reference | 97.66% | +23.695 [+22.008, +25.339] | 1.0000 | 0.00% |

## transfer

Original-rule full-vocabulary accuracy 95.31%; desired-rule reference 95.31%; both rules correct 92.19%. Identity max logit difference 0.0.

| Arm | Desired-answer full-vocab accuracy | Desired−original margin change [95% CI] | Margin recovery | Cross-content desired-rule donor token accuracy |
|---|---:|---:|---:|---:|
| query/pred/same | 0.78% | +0.007 [-0.002, +0.017] | 0.0003 | 0.00% |
| query/pred/cross | 0.78% | +0.008 [-0.017, +0.025] | 0.0004 | 0.00% |
| query/pred/wrong | 0.78% | +0.013 [-0.012, +0.032] | 0.0006 | 0.00% |
| query/pred/random | 0.78% | +0.008 [-0.013, +0.023] | 0.0004 | 0.00% |
| query/pred/cross_delta | 0.00% | +0.003 [-0.006, +0.013] | 0.0001 | 0.00% |
| query/pred/reverse_delta | 0.78% | +0.009 [+0.000, +0.018] | 0.0004 | 0.00% |
| test/pred/same | 0.78% | +0.007 [-0.002, +0.017] | 0.0003 | 0.00% |
| test/pred/cross | 0.78% | +0.013 [-0.030, +0.051] | 0.0006 | 0.00% |
| test/pred/wrong | 0.78% | +0.011 [-0.034, +0.049] | 0.0005 | 0.00% |
| test/pred/random | 0.78% | +0.001 [-0.043, +0.040] | 0.0001 | 0.00% |
| test/pred/cross_delta | 0.00% | +0.003 [-0.006, +0.013] | 0.0001 | 0.00% |
| test/pred/reverse_delta | 0.78% | +0.009 [+0.000, +0.018] | 0.0004 | 0.00% |
| query/corr/same | 0.78% | +2.082 [+1.627, +2.574] | 0.0962 | 0.00% |
| query/corr/cross | 0.00% | +2.068 [+1.613, +2.560] | 0.0955 | 0.00% |
| query/corr/wrong | 1.56% | +2.358 [+1.772, +2.922] | 0.1089 | 0.00% |
| query/corr/random | 1.56% | +2.248 [+1.785, +2.769] | 0.1038 | 0.00% |
| query/corr/cross_delta | 1.56% | +2.837 [+2.456, +3.241] | 0.1310 | 0.00% |
| query/corr/reverse_delta | 2.34% | +2.679 [+2.146, +3.262] | 0.1237 | 0.00% |
| test/corr/same | 0.78% | +2.082 [+1.627, +2.574] | 0.0962 | 0.00% |
| test/corr/cross | 1.56% | +2.980 [+2.449, +3.528] | 0.1376 | 0.00% |
| test/corr/wrong | 1.56% | +1.910 [+1.426, +2.400] | 0.0882 | 0.00% |
| test/corr/random | 0.78% | +2.443 [+1.949, +2.962] | 0.1128 | 0.00% |
| test/corr/cross_delta | 1.56% | +2.837 [+2.456, +3.241] | 0.1310 | 0.00% |
| test/corr/reverse_delta | 2.34% | +2.679 [+2.146, +3.262] | 0.1237 | 0.00% |
| query/both/same | 0.78% | +2.124 [+1.660, +2.625] | 0.0981 | 0.00% |
| query/both/cross | 0.00% | +2.106 [+1.644, +2.606] | 0.0973 | 0.00% |
| query/both/wrong | 1.56% | +2.410 [+1.815, +2.976] | 0.1113 | 0.00% |
| query/both/random | 1.56% | +2.280 [+1.814, +2.802] | 0.1053 | 0.00% |
| query/both/cross_delta | 1.56% | +2.868 [+2.488, +3.273] | 0.1325 | 0.00% |
| query/both/reverse_delta | 2.34% | +2.706 [+2.171, +3.285] | 0.1250 | 0.00% |
| test/both/same | 0.78% | +2.124 [+1.660, +2.625] | 0.0981 | 0.00% |
| test/both/cross | 1.56% | +3.003 [+2.474, +3.549] | 0.1387 | 0.00% |
| test/both/wrong | 1.56% | +1.932 [+1.439, +2.424] | 0.0892 | 0.00% |
| test/both/random | 0.78% | +2.443 [+1.934, +2.975] | 0.1128 | 0.00% |
| test/both/cross_delta | 1.56% | +2.868 [+2.488, +3.273] | 0.1325 | 0.00% |
| test/both/reverse_delta | 2.34% | +2.706 [+2.171, +3.285] | 0.1250 | 0.00% |
| reference | 0.00% | +0.000 [+0.000, +0.000] | 0.0000 | 0.00% |
| target_rule_reference | 95.31% | +21.650 [+20.028, +23.339] | 1.0000 | 0.00% |

The backbone always receives recipient tokens. Only predictor/local-correction coefficients are replaced or offset; downstream recipient computation then proceeds normally. Independent-content donors have a disjoint answer vocabulary, and the desired recipient answer differs from the donor answer. Correct-rule versus wrong-rule donor contrasts hold donor content fixed. They are saved in each stage JSON, alongside all case-level scores.

The donor-token column always tracks the cross-content desired-rule donor token, including in other control arms. It is not the actual donor answer for every wrong/random/same arm. Random donors use separately generated content groups; each bootstrap group includes its own independent random donor. For the repaired initial 300M execution, old neighboring-group random arms are retained only as exploratory audit records.

Copy-query uses disjoint even/odd token-ID vocabularies on the two content sides. This is a special synthetic out-of-distribution control, not a natural semantic rule switch.

The 300M and 1B checkpoints differ in training corpus and budget. Their comparison extends the capability check and is not a controlled estimate of a size-only effect.

Query scope starts at the explicit selector or copy key; test scope also includes the final table/body but excludes demonstrations. No donor final logits or head/source activations are injected. Full route replacement is a broad intervention, not a sparse circuit. CIs resample complete independent table pairs, keeping their four directed cases together; fixed checkpoints and unadjusted intervals.

Reproduce: `CUDA_VISIBLE_DEVICES=3 /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python scripts/circuit_rule_switch.py --phase run --model checkpoints/pr_sync_20260917/300m-dagformer --out experiments/results/circuit_followup_20260929/rule_switch --batch-size 2 --memory-fraction 0.32 --seed 20260929`.
