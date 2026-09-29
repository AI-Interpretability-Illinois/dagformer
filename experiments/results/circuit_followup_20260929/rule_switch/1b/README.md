# Capability-gated routing rule interchange

Model: /scratch/yurenh2/circuit-followup-checkpoints/1b-dagformer. This is a frozen-model inference experiment; no adapter or backbone training.

## Capability screen

The selected task and rule pair must have at least 80% full-vocabulary accuracy under both rules and 70% joint success on an independent screen. Candidate-only accuracy does not bypass this gate. All screened templates are retained.

| Template | Rule 0 accuracy | Rule 1 accuracy | Rule 2 accuracy | Eligible pairs |
|---|---:|---:|---:|---|
| arrows_0shot | 0.00% | 6.25% | 6.25% | [] |
| arrows_3shot | 12.50% | 31.25% | 18.75% | [] |
| arrows_6shot | 12.50% | 6.25% | 25.00% | [] |
| function_0shot | 0.00% | 0.00% | 0.00% | [] |
| function_3shot | 0.00% | 0.00% | 18.75% | [] |
| function_6shot | 0.00% | 0.00% | 18.75% | [] |
| chain_0shot | 25.00% | 31.25% | 12.50% | [] |
| chain_3shot | 0.00% | 12.50% | 68.75% | [] |
| chain_6shot | 0.00% | 18.75% | 62.50% | [] |
| index_0shot | 0.00% | 0.00% | 0.00% | [] |
| index_3shot | 0.00% | 62.50% | 43.75% | [] |
| index_6shot | 62.50% | 68.75% | 18.75% | [] |
| copy_query_0shot | 100.00% | 100.00% | 100.00% | ['0-1', '1-2', '0-2'] |

Selected: {'variant': 'copy_query_0shot', 'rules': [0, 1], 'family': 'copy_query', 'shots': 0}.

None of the screened transformation templates met the two-rule capability gate. The fallback changes which repeated key is queried. It tests address-conditioned routing transfer, not switching a learned multi-step algorithm; failure here does not establish that routing cannot implement algorithm control.

## Desired versus wrong donor rule

The central contrast holds donor content fixed and changes only its rule. An increase in desired-versus-original margin by itself can reflect reduced confidence in the original answer; the wrong-rule and reversed-difference controls distinguish that from directed operation transfer.

| Split | Query/both: correct minus wrong donor margin [95% CI] | Query/both: forward minus reversed rule-difference margin [95% CI] | Desired-answer accuracy, correct / wrong |
|---|---:|---:|---:|
| discovery | -0.010 [-0.377, +0.425] | +0.534 [+0.075, +0.993] | 0.00% / 1.56% |
| heldout | +0.082 [-0.282, +0.446] | +0.019 [-0.429, +0.473] | 0.00% / 0.00% |
| transfer | +0.060 [-0.297, +0.419] | -0.280 [-0.712, +0.127] | 0.00% / 0.00% |

The experiment uses direct donor values or a donor rule difference at scale 1; it does not search amplified gains or train a new routing controller.

## discovery

Original-rule full-vocabulary accuracy 100.00%; desired-rule reference 100.00%; both rules correct 100.00%. Identity max logit difference 0.0.

| Arm | Desired-answer full-vocab accuracy | Desired−original margin change [95% CI] | Margin recovery | Cross-content desired-rule donor token accuracy |
|---|---:|---:|---:|---:|
| query/pred/same | 0.00% | +0.019 [+0.001, +0.039] | 0.0007 | 0.00% |
| query/pred/cross | 0.00% | +0.006 [-0.018, +0.032] | 0.0002 | 0.00% |
| query/pred/wrong | 0.00% | -0.003 [-0.026, +0.023] | -0.0001 | 0.00% |
| query/pred/random | 0.00% | +0.011 [-0.009, +0.035] | 0.0004 | 0.00% |
| query/pred/cross_delta | 0.00% | +0.001 [-0.011, +0.015] | 0.0001 | 0.00% |
| query/pred/reverse_delta | 0.00% | -0.010 [-0.028, +0.011] | -0.0004 | 0.00% |
| test/pred/same | 0.00% | +0.019 [+0.001, +0.039] | 0.0007 | 0.00% |
| test/pred/cross | 0.00% | -0.015 [-0.045, +0.017] | -0.0006 | 0.00% |
| test/pred/wrong | 0.00% | -0.024 [-0.054, +0.009] | -0.0009 | 0.00% |
| test/pred/random | 0.00% | -0.021 [-0.051, +0.010] | -0.0008 | 0.00% |
| test/pred/cross_delta | 0.00% | +0.001 [-0.011, +0.015] | 0.0001 | 0.00% |
| test/pred/reverse_delta | 0.00% | -0.010 [-0.028, +0.011] | -0.0004 | 0.00% |
| query/corr/same | 0.00% | +1.679 [+1.340, +2.034] | 0.0620 | 0.00% |
| query/corr/cross | 0.00% | +1.976 [+1.550, +2.512] | 0.0729 | 0.00% |
| query/corr/wrong | 1.56% | +1.979 [+1.657, +2.322] | 0.0730 | 0.00% |
| query/corr/random | 0.00% | +2.513 [+2.054, +2.962] | 0.0927 | 0.00% |
| query/corr/cross_delta | 0.00% | +4.754 [+4.214, +5.225] | 0.1754 | 0.00% |
| query/corr/reverse_delta | 1.56% | +4.247 [+3.844, +4.626] | 0.1567 | 0.00% |
| test/corr/same | 0.00% | +1.679 [+1.340, +2.034] | 0.0620 | 0.00% |
| test/corr/cross | 0.00% | +1.899 [+1.440, +2.430] | 0.0701 | 0.00% |
| test/corr/wrong | 0.00% | +0.948 [+0.534, +1.446] | 0.0350 | 0.00% |
| test/corr/random | 0.00% | +1.825 [+1.311, +2.377] | 0.0673 | 0.00% |
| test/corr/cross_delta | 0.00% | +4.754 [+4.214, +5.225] | 0.1754 | 0.00% |
| test/corr/reverse_delta | 1.56% | +4.247 [+3.844, +4.626] | 0.1567 | 0.00% |
| query/both/same | 0.00% | +1.756 [+1.388, +2.150] | 0.0648 | 0.00% |
| query/both/cross | 0.00% | +2.029 [+1.584, +2.576] | 0.0749 | 0.00% |
| query/both/wrong | 1.56% | +2.039 [+1.709, +2.392] | 0.0752 | 0.00% |
| query/both/random | 0.00% | +2.559 [+2.089, +3.019] | 0.0944 | 0.00% |
| query/both/cross_delta | 0.00% | +4.813 [+4.274, +5.298] | 0.1776 | 0.00% |
| query/both/reverse_delta | 1.56% | +4.280 [+3.872, +4.672] | 0.1579 | 0.00% |
| test/both/same | 0.00% | +1.756 [+1.388, +2.150] | 0.0648 | 0.00% |
| test/both/cross | 0.00% | +1.944 [+1.475, +2.484] | 0.0718 | 0.00% |
| test/both/wrong | 0.00% | +0.991 [+0.573, +1.494] | 0.0366 | 0.00% |
| test/both/random | 0.00% | +1.881 [+1.359, +2.439] | 0.0694 | 0.00% |
| test/both/cross_delta | 0.00% | +4.813 [+4.274, +5.298] | 0.1776 | 0.00% |
| test/both/reverse_delta | 1.56% | +4.280 [+3.872, +4.672] | 0.1579 | 0.00% |
| reference | 0.00% | +0.000 [+0.000, +0.000] | 0.0000 | 0.00% |
| target_rule_reference | 100.00% | +27.098 [+25.553, +28.598] | 1.0000 | 0.00% |

## heldout

Original-rule full-vocabulary accuracy 99.22%; desired-rule reference 99.22%; both rules correct 98.44%. Identity max logit difference 0.0.

| Arm | Desired-answer full-vocab accuracy | Desired−original margin change [95% CI] | Margin recovery | Cross-content desired-rule donor token accuracy |
|---|---:|---:|---:|---:|
| query/pred/same | 0.00% | +0.012 [+0.001, +0.024] | 0.0004 | 0.00% |
| query/pred/cross | 0.00% | +0.016 [-0.001, +0.036] | 0.0006 | 0.00% |
| query/pred/wrong | 0.00% | +0.023 [+0.009, +0.039] | 0.0009 | 0.00% |
| query/pred/random | 0.00% | +0.020 [+0.006, +0.033] | 0.0007 | 0.00% |
| query/pred/cross_delta | 0.00% | -0.006 [-0.016, +0.004] | -0.0002 | 0.00% |
| query/pred/reverse_delta | 0.00% | +0.006 [-0.006, +0.017] | 0.0002 | 0.00% |
| test/pred/same | 0.00% | +0.012 [+0.001, +0.024] | 0.0004 | 0.00% |
| test/pred/cross | 0.00% | -0.001 [-0.026, +0.027] | -0.0000 | 0.00% |
| test/pred/wrong | 0.00% | +0.003 [-0.023, +0.030] | 0.0001 | 0.00% |
| test/pred/random | 0.00% | -0.012 [-0.038, +0.014] | -0.0004 | 0.00% |
| test/pred/cross_delta | 0.00% | -0.006 [-0.016, +0.004] | -0.0002 | 0.00% |
| test/pred/reverse_delta | 0.00% | +0.006 [-0.006, +0.017] | 0.0002 | 0.00% |
| query/corr/same | 0.00% | +2.207 [+1.827, +2.614] | 0.0824 | 0.00% |
| query/corr/cross | 0.00% | +2.333 [+1.976, +2.693] | 0.0871 | 0.00% |
| query/corr/wrong | 0.00% | +2.224 [+1.851, +2.611] | 0.0831 | 0.00% |
| query/corr/random | 0.00% | +2.466 [+1.928, +3.003] | 0.0921 | 0.00% |
| query/corr/cross_delta | 0.00% | +5.023 [+4.610, +5.454] | 0.1876 | 0.00% |
| query/corr/reverse_delta | 1.56% | +4.999 [+4.500, +5.508] | 0.1867 | 0.00% |
| test/corr/same | 0.00% | +2.207 [+1.827, +2.614] | 0.0824 | 0.00% |
| test/corr/cross | 0.00% | +2.244 [+1.904, +2.580] | 0.0838 | 0.00% |
| test/corr/wrong | 0.00% | +1.004 [+0.646, +1.368] | 0.0375 | 0.00% |
| test/corr/random | 0.00% | +2.086 [+1.494, +2.695] | 0.0779 | 0.00% |
| test/corr/cross_delta | 0.00% | +5.023 [+4.610, +5.454] | 0.1876 | 0.00% |
| test/corr/reverse_delta | 1.56% | +4.999 [+4.500, +5.508] | 0.1867 | 0.00% |
| query/both/same | 0.00% | +2.248 [+1.866, +2.658] | 0.0839 | 0.00% |
| query/both/cross | 0.00% | +2.353 [+1.992, +2.718] | 0.0879 | 0.00% |
| query/both/wrong | 0.00% | +2.270 [+1.895, +2.667] | 0.0848 | 0.00% |
| query/both/random | 0.00% | +2.517 [+1.971, +3.057] | 0.0940 | 0.00% |
| query/both/cross_delta | 0.00% | +5.044 [+4.633, +5.473] | 0.1883 | 0.00% |
| query/both/reverse_delta | 1.56% | +5.025 [+4.526, +5.535] | 0.1876 | 0.00% |
| test/both/same | 0.00% | +2.248 [+1.866, +2.658] | 0.0839 | 0.00% |
| test/both/cross | 0.00% | +2.293 [+1.956, +2.625] | 0.0856 | 0.00% |
| test/both/wrong | 0.00% | +1.045 [+0.696, +1.408] | 0.0390 | 0.00% |
| test/both/random | 0.00% | +2.148 [+1.557, +2.761] | 0.0802 | 0.00% |
| test/both/cross_delta | 0.00% | +5.044 [+4.633, +5.473] | 0.1883 | 0.00% |
| test/both/reverse_delta | 1.56% | +5.025 [+4.526, +5.535] | 0.1876 | 0.00% |
| reference | 0.00% | +0.000 [+0.000, +0.000] | 0.0000 | 0.00% |
| target_rule_reference | 99.22% | +26.781 [+25.025, +28.434] | 1.0000 | 0.00% |

## transfer

Original-rule full-vocabulary accuracy 100.00%; desired-rule reference 100.00%; both rules correct 100.00%. Identity max logit difference 0.0.

| Arm | Desired-answer full-vocab accuracy | Desired−original margin change [95% CI] | Margin recovery | Cross-content desired-rule donor token accuracy |
|---|---:|---:|---:|---:|
| query/pred/same | 0.00% | +0.007 [-0.006, +0.019] | 0.0002 | 0.00% |
| query/pred/cross | 0.00% | +0.011 [-0.002, +0.025] | 0.0004 | 0.00% |
| query/pred/wrong | 0.00% | +0.007 [-0.007, +0.021] | 0.0003 | 0.00% |
| query/pred/random | 0.00% | +0.015 [-0.001, +0.034] | 0.0005 | 0.00% |
| query/pred/cross_delta | 0.00% | +0.000 [-0.015, +0.017] | 0.0000 | 0.00% |
| query/pred/reverse_delta | 0.00% | +0.003 [-0.007, +0.013] | 0.0001 | 0.00% |
| test/pred/same | 0.00% | +0.007 [-0.006, +0.019] | 0.0002 | 0.00% |
| test/pred/cross | 0.00% | -0.009 [-0.037, +0.023] | -0.0003 | 0.00% |
| test/pred/wrong | 0.00% | -0.012 [-0.040, +0.016] | -0.0004 | 0.00% |
| test/pred/random | 0.00% | -0.004 [-0.030, +0.026] | -0.0001 | 0.00% |
| test/pred/cross_delta | 0.00% | +0.000 [-0.015, +0.017] | 0.0000 | 0.00% |
| test/pred/reverse_delta | 0.00% | +0.003 [-0.007, +0.013] | 0.0001 | 0.00% |
| query/corr/same | 0.00% | +2.884 [+2.465, +3.358] | 0.1043 | 0.00% |
| query/corr/cross | 0.00% | +2.668 [+2.341, +3.017] | 0.0964 | 0.00% |
| query/corr/wrong | 0.00% | +2.585 [+2.203, +2.985] | 0.0935 | 0.00% |
| query/corr/random | 0.00% | +2.642 [+2.185, +3.100] | 0.0955 | 0.00% |
| query/corr/cross_delta | 0.78% | +5.367 [+4.976, +5.787] | 0.1940 | 0.00% |
| query/corr/reverse_delta | 0.78% | +5.658 [+5.212, +6.161] | 0.2045 | 0.00% |
| test/corr/same | 0.00% | +2.884 [+2.465, +3.358] | 0.1043 | 0.00% |
| test/corr/cross | 0.78% | +2.845 [+2.494, +3.221] | 0.1028 | 0.00% |
| test/corr/wrong | 0.00% | +1.301 [+1.027, +1.574] | 0.0470 | 0.00% |
| test/corr/random | 0.00% | +2.109 [+1.587, +2.652] | 0.0762 | 0.00% |
| test/corr/cross_delta | 0.78% | +5.367 [+4.976, +5.787] | 0.1940 | 0.00% |
| test/corr/reverse_delta | 0.78% | +5.658 [+5.212, +6.161] | 0.2045 | 0.00% |
| query/both/same | 0.00% | +2.952 [+2.524, +3.435] | 0.1067 | 0.00% |
| query/both/cross | 0.00% | +2.708 [+2.374, +3.059] | 0.0979 | 0.00% |
| query/both/wrong | 0.00% | +2.648 [+2.266, +3.054] | 0.0957 | 0.00% |
| query/both/random | 0.00% | +2.728 [+2.258, +3.190] | 0.0986 | 0.00% |
| query/both/cross_delta | 0.78% | +5.410 [+5.004, +5.846] | 0.1956 | 0.00% |
| query/both/reverse_delta | 0.78% | +5.690 [+5.244, +6.193] | 0.2057 | 0.00% |
| test/both/same | 0.00% | +2.952 [+2.524, +3.435] | 0.1067 | 0.00% |
| test/both/cross | 0.78% | +2.916 [+2.552, +3.306] | 0.1054 | 0.00% |
| test/both/wrong | 0.00% | +1.360 [+1.094, +1.623] | 0.0492 | 0.00% |
| test/both/random | 0.00% | +2.174 [+1.642, +2.718] | 0.0786 | 0.00% |
| test/both/cross_delta | 0.78% | +5.410 [+5.004, +5.846] | 0.1956 | 0.00% |
| test/both/reverse_delta | 0.78% | +5.690 [+5.244, +6.193] | 0.2057 | 0.00% |
| reference | 0.00% | +0.000 [+0.000, +0.000] | 0.0000 | 0.00% |
| target_rule_reference | 100.00% | +27.667 [+25.969, +29.369] | 1.0000 | 0.00% |

The backbone always receives recipient tokens. Only predictor/local-correction coefficients are replaced or offset; downstream recipient computation then proceeds normally. Independent-content donors have a disjoint answer vocabulary, and the desired recipient answer differs from the donor answer. Correct-rule versus wrong-rule donor contrasts hold donor content fixed. They are saved in each stage JSON, alongside all case-level scores.

The donor-token column always tracks the cross-content desired-rule donor token, including in other control arms. It is not the actual donor answer for every wrong/random/same arm. Random donors use separately generated content groups; each bootstrap group includes its own independent random donor. For the repaired initial 300M execution, old neighboring-group random arms are retained only as exploratory audit records.

Copy-query uses disjoint even/odd token-ID vocabularies on the two content sides. This is a special synthetic out-of-distribution control, not a natural semantic rule switch.

The 300M and 1B checkpoints differ in training corpus and budget. Their comparison extends the capability check and is not a controlled estimate of a size-only effect.

Query scope starts at the explicit selector or copy key; test scope also includes the final table/body but excludes demonstrations. No donor final logits or head/source activations are injected. Full route replacement is a broad intervention, not a sparse circuit. CIs resample complete independent table pairs, keeping their four directed cases together; fixed checkpoints and unadjusted intervals.

Reproduce: `CUDA_VISIBLE_DEVICES=3 /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python scripts/circuit_rule_switch.py --phase run --model /scratch/yurenh2/circuit-followup-checkpoints/1b-dagformer --out experiments/results/circuit_followup_20260929/rule_switch/1b --batch-size 1 --memory-fraction 0.32 --seed 20260929`.
