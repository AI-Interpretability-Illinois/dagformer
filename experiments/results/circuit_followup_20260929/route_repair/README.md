# Localization of damage-induced routing compensation

Frozen 300M FourWay + local correction. Zero one attention head output at all token positions. Capture the complete clean and damaged effective coefficients (predictor + correction). Keep the lesion and all clean coefficients, then restore selected layer/stream groups to their damaged-run values. Source activations are always computed by the recipient.

A group is all head/source coefficients of Q, K, V, or R in one layer at all tokens. This is a coarse response-localization experiment, not a sparse-edge or full-circuit claim. The donor is the same input and checkpoint under the lesion; no answer or other-example content is introduced.

Replaying every damaged-run coefficient must reproduce the normal damaged model exactly. Global predictor inputs are unchanged, so coefficient changes arise from local correction. Candidate groups are only downstream of the lesion. Discovery ranks their individual NLL rescue; top-k unions are not asserted to be optimal.

This is offline causal replay of damage-induced coefficients: it tests the sufficiency of recorded changes under other clean-coefficient clamps, rather than the necessity of an online feedback loop. Random controls match layer-group counts; their selected-set overlaps and any identical sets are recorded in final_arms.

Primary head (zero-based): [6, 11]. Selection: {'ordered_groups': [[7, 'q'], [8, 'r'], [10, 'r'], [7, 'v'], [9, 'r'], [11, 'r'], [7, 'r'], [7, 'k'], [10, 'q'], [9, 'v'], [11, 'k'], [11, 'q'], [9, 'q'], [9, 'k'], [10, 'k'], [11, 'v'], [8, 'k'], [8, 'q'], [10, 'v'], [8, 'v']], 'discovery_pairs': 32, 'individual_nll_rescue': [0.04837045782187488, 0.036833172118349466, 0.02087243714777287, 0.01705631163349608, 0.012962333705218043, 0.012184415056253783, 0.005289303939207457, 0.0017216252163052559, 0.0014940786277293228, 0.001163941909908317, 0.00069201485166559, 0.0005271935879136436, 0.00024318273062817752, 4.104532126802951e-05, -2.4664157535880804e-05, -0.0005899178722756915, -0.0006236215704120696, -0.0007049954074318521, -0.0010797580180224031, -0.0026585474552121013]}

## heldout

Pairs: 64. Full replay logit error: 0.0.

| Arm | NLL | Accuracy | NLL rescue [paired 95% CI] |
|---|---:|---:|---:|
| clean | 0.0786 | 100.00% | — |
| A_dynamic | 0.3824 | 99.22% | — |
| A_fixed | 0.5528 | 96.09% | — |
| A_full_replay | 0.3824 | 99.22% | — |
| top1 | 0.4826 | 96.88% | +0.0702 [+0.0408, +0.1036] |
| uninjured_top1 | 0.0720 | 100.00% | +0.0066 [+0.0031, +0.0100] |
| random1_0 | 0.5548 | 96.09% | -0.0019 [-0.0118, +0.0070] |
| random1_1 | 0.5409 | 95.31% | +0.0119 [+0.0017, +0.0231] |
| random1_2 | 0.5409 | 95.31% | +0.0119 [+0.0017, +0.0231] |
| top2 | 0.4406 | 99.22% | +0.1123 [+0.0736, +0.1576] |
| uninjured_top2 | 0.0672 | 100.00% | +0.0113 [+0.0070, +0.0161] |
| random2_0 | 0.5413 | 96.88% | +0.0115 [+0.0012, +0.0223] |
| random2_1 | 0.5411 | 96.09% | +0.0117 [+0.0010, +0.0234] |
| random2_2 | 0.4994 | 96.88% | +0.0534 [+0.0376, +0.0708] |
| top4 | 0.4158 | 98.44% | +0.1370 [+0.0940, +0.1854] |
| uninjured_top4 | 0.0655 | 100.00% | +0.0131 [+0.0080, +0.0188] |
| random4_0 | 0.5427 | 96.88% | +0.0101 [-0.0029, +0.0236] |
| random4_1 | 0.5316 | 96.09% | +0.0212 [+0.0088, +0.0342] |
| random4_2 | 0.4783 | 96.88% | +0.0745 [+0.0449, +0.1075] |
| top8 | 0.3821 | 99.22% | +0.1707 [+0.1187, +0.2311] |
| uninjured_top8 | 0.0686 | 100.00% | +0.0100 [+0.0036, +0.0165] |
| random8_0 | 0.4141 | 98.44% | +0.1387 [+0.0947, +0.1896] |
| random8_1 | 0.4254 | 98.44% | +0.1274 [+0.0862, +0.1745] |
| random8_2 | 0.4442 | 98.44% | +0.1086 [+0.0708, +0.1524] |

Positive rescue is lower NLL than the lesion with clean coefficients held fixed. Uninjured-response controls instead compare against the uninjured model. Paired bootstrap groups both value-swap directions; uncertainty is over input pairs, not model seeds.

| Contrast | NLL benefit [paired 95% CI] |
|---|---:|
| top1_damage_specific_rescue | +0.0637 [+0.0368, +0.0944] |
| top1_minus_random1_0 | +0.0722 [+0.0424, +0.1063] |
| top1_minus_random1_1 | +0.0583 [+0.0278, +0.0900] |
| top1_minus_random1_2 | +0.0583 [+0.0278, +0.0900] |
| top2_damage_specific_rescue | +0.1009 [+0.0649, +0.1430] |
| top2_minus_random2_0 | +0.1007 [+0.0621, +0.1434] |
| top2_minus_random2_1 | +0.1005 [+0.0619, +0.1434] |
| top2_minus_random2_2 | +0.0589 [+0.0293, +0.0945] |
| top4_damage_specific_rescue | +0.1239 [+0.0839, +0.1682] |
| top4_minus_random4_0 | +0.1269 [+0.0841, +0.1750] |
| top4_minus_random4_1 | +0.1158 [+0.0741, +0.1630] |
| top4_minus_random4_2 | +0.0625 [+0.0418, +0.0863] |
| top8_damage_specific_rescue | +0.1607 [+0.1125, +0.2169] |
| top8_minus_random8_0 | +0.0320 [+0.0197, +0.0455] |
| top8_minus_random8_1 | +0.0434 [+0.0284, +0.0605] |
| top8_minus_random8_2 | +0.0621 [+0.0407, +0.0870] |

Recovered fractions divide the mean NLL rescue by mean total routing compensation; they do not describe a fraction of the whole task circuit.

{'top1': {'mean': 0.41188310231801073, 'paired_95ci': [0.306541577780471, 0.4976950259097407]}, 'top2': {'mean': 0.6586621251584802, 'paired_95ci': [0.5722138623262535, 0.7309776661448864]}, 'top4': {'mean': 0.8038521361216254, 'paired_95ci': [0.7220986906629696, 0.8880749531433376]}, 'top8': {'mean': 1.0016650042748738, 'paired_95ci': [0.9701909742145511, 1.036932317224774]}}

## transfer

Pairs: 64. Full replay logit error: 0.0.

| Arm | NLL | Accuracy | NLL rescue [paired 95% CI] |
|---|---:|---:|---:|
| clean | 0.3981 | 92.19% | — |
| A_dynamic | 1.0991 | 82.03% | — |
| A_fixed | 1.3623 | 74.22% | — |
| A_full_replay | 1.0991 | 82.03% | — |
| top1 | 1.2581 | 78.91% | +0.1042 [+0.0608, +0.1530] |
| uninjured_top1 | 0.3830 | 95.31% | +0.0152 [-0.0044, +0.0424] |
| random1_0 | 1.3621 | 74.22% | +0.0002 [-0.0147, +0.0159] |
| random1_1 | 1.3448 | 75.00% | +0.0175 [+0.0004, +0.0340] |
| random1_2 | 1.3448 | 75.00% | +0.0175 [+0.0004, +0.0340] |
| top2 | 1.1981 | 79.69% | +0.1641 [+0.1124, +0.2240] |
| uninjured_top2 | 0.3749 | 95.31% | +0.0233 [+0.0026, +0.0518] |
| random2_0 | 1.3459 | 75.00% | +0.0164 [-0.0003, +0.0328] |
| random2_1 | 1.3433 | 75.00% | +0.0190 [+0.0032, +0.0344] |
| random2_2 | 1.2736 | 75.00% | +0.0887 [+0.0623, +0.1181] |
| top4 | 1.1456 | 80.47% | +0.2167 [+0.1562, +0.2870] |
| uninjured_top4 | 0.3658 | 95.31% | +0.0323 [+0.0114, +0.0599] |
| random4_0 | 1.3442 | 75.00% | +0.0181 [-0.0019, +0.0399] |
| random4_1 | 1.3209 | 75.00% | +0.0414 [+0.0188, +0.0659] |
| random4_2 | 1.2390 | 78.91% | +0.1233 [+0.0779, +0.1767] |
| top8 | 1.0988 | 81.25% | +0.2635 [+0.1846, +0.3564] |
| uninjured_top8 | 0.3793 | 95.31% | +0.0188 [-0.0072, +0.0490] |
| random8_0 | 1.1479 | 80.47% | +0.2144 [+0.1482, +0.2943] |
| random8_1 | 1.1538 | 79.69% | +0.2084 [+0.1417, +0.2896] |
| random8_2 | 1.1784 | 79.69% | +0.1839 [+0.1221, +0.2573] |

Positive rescue is lower NLL than the lesion with clean coefficients held fixed. Uninjured-response controls instead compare against the uninjured model. Paired bootstrap groups both value-swap directions; uncertainty is over input pairs, not model seeds.

| Contrast | NLL benefit [paired 95% CI] |
|---|---:|
| top1_damage_specific_rescue | +0.0890 [+0.0588, +0.1227] |
| top1_minus_random1_0 | +0.1039 [+0.0589, +0.1527] |
| top1_minus_random1_1 | +0.0867 [+0.0449, +0.1335] |
| top1_minus_random1_2 | +0.0867 [+0.0449, +0.1335] |
| top2_damage_specific_rescue | +0.1409 [+0.1015, +0.1857] |
| top2_minus_random2_0 | +0.1478 [+0.0997, +0.2048] |
| top2_minus_random2_1 | +0.1452 [+0.0980, +0.2005] |
| top2_minus_random2_2 | +0.0754 [+0.0253, +0.1245] |
| top4_damage_specific_rescue | +0.1844 [+0.1367, +0.2401] |
| top4_minus_random4_0 | +0.1986 [+0.1423, +0.2634] |
| top4_minus_random4_1 | +0.1753 [+0.1227, +0.2363] |
| top4_minus_random4_2 | +0.0934 [+0.0712, +0.1180] |
| top8_damage_specific_rescue | +0.2447 [+0.1794, +0.3189] |
| top8_minus_random8_0 | +0.0491 [+0.0340, +0.0671] |
| top8_minus_random8_1 | +0.0551 [+0.0366, +0.0751] |
| top8_minus_random8_2 | +0.0797 [+0.0556, +0.1063] |

Recovered fractions divide the mean NLL rescue by mean total routing compensation; they do not describe a fraction of the whole task circuit.

{'top1': {'mean': 0.39581295086243595, 'paired_95ci': [0.2905705044013024, 0.48807381696884067]}, 'top2': {'mean': 0.6237377384892459, 'paired_95ci': [0.5283626546450183, 0.7220795471529485]}, 'top4': {'mean': 0.8234118408787238, 'paired_95ci': [0.7540811906135213, 0.9118133367085195]}, 'top8': {'mean': 1.001355570345109, 'paired_95ci': [0.9799707200932156, 1.0267424650014485]}}

## unchanged_query

Pairs: 64. Full replay logit error: 0.0.

| Arm | NLL | Accuracy | NLL rescue [paired 95% CI] |
|---|---:|---:|---:|
| clean | 0.0581 | 100.00% | — |
| A_dynamic | 0.3871 | 96.09% | — |
| A_fixed | 0.4853 | 96.09% | — |
| A_full_replay | 0.3871 | 96.09% | — |
| top1 | 0.4379 | 96.88% | +0.0474 [+0.0267, +0.0717] |
| uninjured_top1 | 0.0511 | 100.00% | +0.0071 [+0.0026, +0.0133] |
| random1_0 | 0.4883 | 95.31% | -0.0030 [-0.0091, +0.0028] |
| random1_1 | 0.4825 | 95.31% | +0.0028 [-0.0091, +0.0118] |
| random1_2 | 0.4825 | 95.31% | +0.0028 [-0.0091, +0.0118] |
| top2 | 0.4149 | 96.88% | +0.0704 [+0.0420, +0.1026] |
| uninjured_top2 | 0.0487 | 100.00% | +0.0094 [+0.0037, +0.0173] |
| random2_0 | 0.4832 | 94.53% | +0.0021 [-0.0097, +0.0114] |
| random2_1 | 0.4840 | 96.09% | +0.0013 [-0.0112, +0.0105] |
| random2_2 | 0.4575 | 95.31% | +0.0278 [+0.0067, +0.0495] |
| top4 | 0.4006 | 96.09% | +0.0848 [+0.0459, +0.1277] |
| uninjured_top4 | 0.0472 | 100.00% | +0.0110 [+0.0045, +0.0206] |
| random4_0 | 0.4876 | 94.53% | -0.0023 [-0.0147, +0.0082] |
| random4_1 | 0.4799 | 94.53% | +0.0054 [-0.0131, +0.0215] |
| random4_2 | 0.4354 | 96.88% | +0.0500 [+0.0232, +0.0812] |
| top8 | 0.3907 | 96.09% | +0.0946 [+0.0545, +0.1394] |
| uninjured_top8 | 0.0492 | 100.00% | +0.0089 [+0.0021, +0.0176] |
| random8_0 | 0.4056 | 96.09% | +0.0797 [+0.0442, +0.1171] |
| random8_1 | 0.4125 | 96.88% | +0.0728 [+0.0410, +0.1089] |
| random8_2 | 0.4166 | 96.88% | +0.0688 [+0.0390, +0.1015] |

Positive rescue is lower NLL than the lesion with clean coefficients held fixed. Uninjured-response controls instead compare against the uninjured model. Paired bootstrap groups both value-swap directions; uncertainty is over input pairs, not model seeds.

| Contrast | NLL benefit [paired 95% CI] |
|---|---:|
| top1_damage_specific_rescue | +0.0404 [+0.0232, +0.0602] |
| top1_minus_random1_0 | +0.0504 [+0.0272, +0.0782] |
| top1_minus_random1_1 | +0.0446 [+0.0222, +0.0693] |
| top1_minus_random1_2 | +0.0446 [+0.0222, +0.0693] |
| top2_damage_specific_rescue | +0.0610 [+0.0367, +0.0887] |
| top2_minus_random2_0 | +0.0683 [+0.0406, +0.1002] |
| top2_minus_random2_1 | +0.0691 [+0.0413, +0.1006] |
| top2_minus_random2_2 | +0.0427 [+0.0214, +0.0667] |
| top4_damage_specific_rescue | +0.0738 [+0.0404, +0.1104] |
| top4_minus_random4_0 | +0.0870 [+0.0487, +0.1320] |
| top4_minus_random4_1 | +0.0794 [+0.0476, +0.1155] |
| top4_minus_random4_2 | +0.0348 [+0.0167, +0.0532] |
| top8_damage_specific_rescue | +0.0857 [+0.0504, +0.1234] |
| top8_minus_random8_0 | +0.0150 [+0.0056, +0.0248] |
| top8_minus_random8_1 | +0.0218 [+0.0079, +0.0357] |
| top8_minus_random8_2 | +0.0259 [+0.0091, +0.0431] |

Recovered fractions divide the mean NLL rescue by mean total routing compensation; they do not describe a fraction of the whole task circuit.

{'top1': {'mean': 0.483024748226399, 'paired_95ci': [0.36028474483117745, 0.6386249073924745]}, 'top2': {'mean': 0.7174333286910172, 'paired_95ci': [0.6116849923269484, 0.8482247985671031]}, 'top4': {'mean': 0.8632943443407262, 'paired_95ci': [0.7345335942885091, 0.9540753656559721]}, 'top8': {'mean': 0.9637015960284783, 'paired_95ci': [0.8955689110767587, 1.0056059717120727]}}
