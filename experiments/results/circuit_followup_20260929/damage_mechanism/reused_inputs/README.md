# Damage, conditional backups, and routing response

Frozen 300M checkpoints; no finetuning. Head lesions zero attention-head outputs before o_proj at all positions. Native DAG lesions remove selected V source-read contributions only at the three known historical value-token locations.

A is the first non-layer0 V head from yesterday's discovery-only common-message ranking. Candidate backup heads are all 16 heads in the next layer. DAG source-read candidates are the next-layer heads reading sources 0/1/2 (embedding / layer0 output / layer1 output), which precede A. Selection maximizes conditional NLL interaction on discovery only, under clean effective-route clamps in DAG.

Clean clamps preserve the complete per-input effective alpha (global predictor plus local correction), at every layer/stream/position. They preserve input dependence and prevent effective routing from responding to the damage. The external global predictor reads only input IDs, so its coefficients cannot change from an internal lesion with the same input and frozen weights. Attention probabilities can still change.

Interaction = NLL(A+B) - NLL(A) - NLL(B) + NLL(clean). Positive interaction is additional joint damage; it is not sufficient by itself to prove a unique backup computation. Recent-source controls use the same destination head and three most recent sources; random controls preserve destination layer and number of heads/source reads. Controls are not matched for intervention norm.

The DAG native source-read experiments are not compared numerically with dense head counts. Head-pair interventions use the same head-output operator in both models, but model parameters and selected head locations differ. All positions are supplied by task construction. This is synthetic copying, not general semantic binding or pretrained pruning recovery.

## dag

Selection: {'A': (6, 11), 'backup_head': (7, 1), 'head_conditional_NLL_scores': [-0.010897813755946117, 0.29431894691060734, 0.00015151104435062734, 0.022935276616408373, -0.002240026189610944, 0.017137495705355832, 0.0021522498318518046, 0.08004012449782749, -9.873840099317022e-06, 0.21069844327666942, -0.0005645678102155216, 0.19763166619486583, -0.008009730835510709, 0.0013626994941660087, 0.03642128663886979, -0.021835827942595643], 'discovery_pairs': 32, 'selection_under_fixed_effective_alpha': True, 'control_heads': [3, 5, 8], 'early_read_head': (7, 1), 'early_sources': [0, 1, 2], 'early_conditional_NLL_scores': [-0.0006160844677651767, 0.3613989252326064, 0.001043873576236365, 0.025704839959871606, -0.0015607662817274104, 0.001363274661343894, 0.0009435658976144623, 0.12669930332776858, 0.004801961855264381, 0.22676109811527567, 0.0009407345787622035, 0.2514602980818381, 0.0018150211444663, -0.00024237730303866556, 0.046791598610980145, 5.78266099182656e-06], 'early_control_heads': [11, 7, 2]}

### heldout: 64 independent pairs

| Arm | Correct NLL | Correct full-vocabulary accuracy | Donor full-vocabulary accuracy |
|---|---:|---:|---:|
| A_dynamic | 0.5613 | 93.75% | 0.00% |
| head_B_dynamic | 0.2153 | 99.22% | 0.00% |
| head_AB_dynamic | 0.7811 | 92.97% | 0.00% |
| head_random0_B | 0.1427 | 99.22% | 0.00% |
| head_random0_AB | 0.5905 | 94.53% | 0.00% |
| head_random1_B | 0.1434 | 99.22% | 0.00% |
| head_random1_AB | 0.5879 | 93.75% | 0.00% |
| head_random2_B | 0.1322 | 99.22% | 0.00% |
| head_random2_AB | 0.5578 | 93.75% | 0.00% |
| clean_fixed | 0.1350 | 99.22% | 0.00% |
| A_fixed | 0.7034 | 91.41% | 0.00% |
| head_B_fixed | 0.2200 | 99.22% | 0.00% |
| head_AB_fixed | 1.0119 | 88.28% | 0.00% |
| early_B_dynamic | 0.2689 | 98.44% | 0.00% |
| early_AB_dynamic | 0.9073 | 88.28% | 0.00% |
| early_B_fixed | 0.2763 | 99.22% | 0.00% |
| early_AB_fixed | 1.1679 | 85.16% | 0.00% |
| recent_B | 0.1344 | 99.22% | 0.00% |
| recent_AB | 0.7025 | 91.41% | 0.00% |
| early_random0_B | 0.2080 | 99.22% | 0.00% |
| early_random0_AB | 1.0247 | 85.94% | 0.00% |
| early_random1_B | 0.1692 | 99.22% | 0.00% |
| early_random1_AB | 0.8649 | 88.28% | 0.00% |
| early_random2_B | 0.1350 | 99.22% | 0.00% |
| early_random2_AB | 0.7071 | 91.41% | 0.00% |
| early_donor_B | 0.3839 | 98.44% | 0.00% |
| early_donor_AB | 1.5775 | 75.00% | 0.00% |
| clean | 0.1350 | 99.22% | 0.00% |

| Contrast | NLL effect [paired 95% CI] |
|---|---:|
| head_interaction_dynamic | +0.1395 [+0.0789, +0.2032] |
| head_random0_interaction | +0.0215 [+0.0128, +0.0328] |
| head_random1_interaction | +0.0183 [+0.0107, +0.0257] |
| head_random2_interaction | -0.0007 [-0.0034, +0.0021] |
| head_interaction_fixed | +0.2235 [+0.1571, +0.2925] |
| early_interaction_dynamic | +0.2121 [+0.1460, +0.2882] |
| early_interaction_fixed | +0.3233 [+0.2464, +0.4044] |
| recent_interaction_fixed | -0.0004 [-0.0022, +0.0014] |
| early_random0_interaction_fixed | +0.2483 [+0.1891, +0.3127] |
| early_random1_interaction_fixed | +0.1273 [+0.0849, +0.1764] |
| early_random2_interaction_fixed | +0.0037 [+0.0021, +0.0055] |
| A_fixed_minus_dynamic_NLL | +0.1421 [+0.1030, +0.1839] |
| head_B_fixed_minus_dynamic_NLL | +0.0046 [-0.0021, +0.0118] |
| head_AB_fixed_minus_dynamic_NLL | +0.2308 [+0.1779, +0.2875] |
| early_B_fixed_minus_dynamic_NLL | +0.0073 [+0.0027, +0.0124] |
| early_AB_fixed_minus_dynamic_NLL | +0.2606 [+0.1963, +0.3296] |

Effective-alpha relative change after A: {'mean': 0.0973834854667075, 'paired_95ci': [0.0949950415379135, 0.09965026015561307]}

### transfer: 64 independent pairs

| Arm | Correct NLL | Correct full-vocabulary accuracy | Donor full-vocabulary accuracy |
|---|---:|---:|---:|
| A_dynamic | 0.7490 | 91.41% | 0.00% |
| head_B_dynamic | 0.1668 | 100.00% | 0.00% |
| head_AB_dynamic | 1.2298 | 83.59% | 0.00% |
| head_random0_B | 0.0812 | 100.00% | 0.00% |
| head_random0_AB | 0.7715 | 92.19% | 0.00% |
| head_random1_B | 0.0850 | 100.00% | 0.00% |
| head_random1_AB | 0.7952 | 91.41% | 0.00% |
| head_random2_B | 0.0759 | 100.00% | 0.00% |
| head_random2_AB | 0.7517 | 91.41% | 0.00% |
| clean_fixed | 0.0759 | 100.00% | 0.00% |
| A_fixed | 1.0817 | 82.03% | 0.00% |
| head_B_fixed | 0.1768 | 100.00% | 0.00% |
| head_AB_fixed | 1.7311 | 71.88% | 0.00% |
| early_B_dynamic | 0.1854 | 100.00% | 0.00% |
| early_AB_dynamic | 1.2877 | 81.25% | 0.00% |
| early_B_fixed | 0.2003 | 100.00% | 0.00% |
| early_AB_fixed | 1.8521 | 67.19% | 0.00% |
| recent_B | 0.0756 | 100.00% | 0.00% |
| recent_AB | 1.0804 | 82.03% | 0.00% |
| early_random0_B | 0.1587 | 100.00% | 0.00% |
| early_random0_AB | 1.5973 | 71.09% | 0.00% |
| early_random1_B | 0.0897 | 100.00% | 0.00% |
| early_random1_AB | 1.2441 | 80.47% | 0.00% |
| early_random2_B | 0.0760 | 100.00% | 0.00% |
| early_random2_AB | 1.0842 | 82.03% | 0.00% |
| early_donor_B | 0.3928 | 99.22% | 0.00% |
| early_donor_AB | 2.6015 | 49.22% | 0.00% |
| clean | 0.0759 | 100.00% | 0.00% |

| Contrast | NLL effect [paired 95% CI] |
|---|---:|
| head_interaction_dynamic | +0.3898 [+0.2980, +0.4877] |
| head_random0_interaction | +0.0172 [+0.0116, +0.0231] |
| head_random1_interaction | +0.0370 [+0.0282, +0.0463] |
| head_random2_interaction | +0.0027 [-0.0013, +0.0066] |
| head_interaction_fixed | +0.5484 [+0.4485, +0.6560] |
| early_interaction_dynamic | +0.4291 [+0.3375, +0.5304] |
| early_interaction_fixed | +0.6460 [+0.5353, +0.7662] |
| recent_interaction_fixed | -0.0010 [-0.0034, +0.0014] |
| early_random0_interaction_fixed | +0.4328 [+0.3574, +0.5168] |
| early_random1_interaction_fixed | +0.1486 [+0.1076, +0.1953] |
| early_random2_interaction_fixed | +0.0024 [+0.0003, +0.0044] |
| A_fixed_minus_dynamic_NLL | +0.3327 [+0.2456, +0.4316] |
| head_B_fixed_minus_dynamic_NLL | +0.0100 [+0.0037, +0.0169] |
| head_AB_fixed_minus_dynamic_NLL | +0.5013 [+0.3928, +0.6186] |
| early_B_fixed_minus_dynamic_NLL | +0.0148 [+0.0084, +0.0222] |
| early_AB_fixed_minus_dynamic_NLL | +0.5644 [+0.4425, +0.6979] |

Effective-alpha relative change after A: {'mean': 0.11228984396439046, 'paired_95ci': [0.11045076749578583, 0.11419918627216248]}

### unchanged_query: 64 independent pairs

| Arm | Correct NLL | Correct full-vocabulary accuracy | Donor full-vocabulary accuracy |
|---|---:|---:|---:|
| A_dynamic | 0.4970 | 97.66% | 0.00% |
| head_B_dynamic | 0.2486 | 98.44% | 0.00% |
| head_AB_dynamic | 0.7457 | 93.75% | 0.00% |
| head_random0_B | 0.1879 | 98.44% | 0.00% |
| head_random0_AB | 0.5208 | 96.88% | 0.00% |
| head_random1_B | 0.1842 | 98.44% | 0.00% |
| head_random1_AB | 0.5245 | 97.66% | 0.00% |
| head_random2_B | 0.1707 | 98.44% | 0.00% |
| head_random2_AB | 0.4895 | 98.44% | 0.00% |
| clean_fixed | 0.1742 | 98.44% | 0.00% |
| A_fixed | 0.6349 | 93.75% | 0.00% |
| head_B_fixed | 0.2611 | 98.44% | 0.00% |
| head_AB_fixed | 0.9943 | 89.06% | 0.00% |
| early_B_dynamic | 0.1737 | 98.44% | 0.00% |
| early_AB_dynamic | 0.4966 | 97.66% | 0.00% |
| early_B_fixed | 0.1744 | 98.44% | 0.00% |
| early_AB_fixed | 0.6354 | 93.75% | 0.00% |
| recent_B | 0.1743 | 98.44% | 0.00% |
| recent_AB | 0.6349 | 93.75% | 0.00% |
| early_random0_B | 0.1743 | 98.44% | 0.00% |
| early_random0_AB | 0.6348 | 93.75% | 0.00% |
| early_random1_B | 0.1742 | 98.44% | 0.00% |
| early_random1_AB | 0.6352 | 92.97% | 0.00% |
| early_random2_B | 0.1743 | 98.44% | 0.00% |
| early_random2_AB | 0.6353 | 92.97% | 0.00% |
| early_donor_B | 0.1746 | 98.44% | 0.00% |
| early_donor_AB | 0.6345 | 93.75% | 0.00% |
| clean | 0.1742 | 98.44% | 0.00% |

| Contrast | NLL effect [paired 95% CI] |
|---|---:|
| head_interaction_dynamic | +0.1742 [+0.1072, +0.2512] |
| head_random0_interaction | +0.0101 [+0.0026, +0.0188] |
| head_random1_interaction | +0.0175 [+0.0104, +0.0246] |
| head_random2_interaction | -0.0040 [-0.0089, +0.0000] |
| head_interaction_fixed | +0.2725 [+0.1868, +0.3680] |
| early_interaction_dynamic | +0.0001 [-0.0014, +0.0020] |
| early_interaction_fixed | +0.0003 [-0.0010, +0.0015] |
| recent_interaction_fixed | -0.0000 [-0.0011, +0.0012] |
| early_random0_interaction_fixed | -0.0001 [-0.0013, +0.0010] |
| early_random1_interaction_fixed | +0.0004 [-0.0007, +0.0015] |
| early_random2_interaction_fixed | +0.0004 [-0.0009, +0.0016] |
| A_fixed_minus_dynamic_NLL | +0.1378 [+0.0705, +0.2382] |
| head_B_fixed_minus_dynamic_NLL | +0.0124 [+0.0023, +0.0232] |
| head_AB_fixed_minus_dynamic_NLL | +0.2486 [+0.1455, +0.3633] |
| early_B_fixed_minus_dynamic_NLL | +0.0008 [+0.0001, +0.0018] |
| early_AB_fixed_minus_dynamic_NLL | +0.1388 [+0.0717, +0.2390] |

Effective-alpha relative change after A: {'mean': 0.09572119207587093, 'paired_95ci': [0.0933590069616912, 0.09797994114633184]}

## dense

Selection: {'A': (4, 1), 'backup_head': (5, 1), 'head_conditional_NLL_scores': [-0.04176338098113774, 0.1304580178893957, -0.015362660205937573, -0.005021477578338818, 0.01404349163385632, 0.0665632545005792, 0.036113975942498655, 0.08483847960269486, 0.002442676779537578, -0.06823579414958658, -0.0440159451263753, 0.0027919235344597837, -0.00017293930250161793, -0.0060989483190496685, 0.012263760465430096, -0.011696359024426783], 'discovery_pairs': 32, 'selection_under_fixed_effective_alpha': False, 'control_heads': [3, 5, 8]}

### heldout: 64 independent pairs

| Arm | Correct NLL | Correct full-vocabulary accuracy | Donor full-vocabulary accuracy |
|---|---:|---:|---:|
| A_dynamic | 0.6912 | 92.19% | 0.00% |
| head_B_dynamic | 0.3304 | 98.44% | 0.00% |
| head_AB_dynamic | 0.8805 | 88.28% | 0.00% |
| head_random0_B | 0.2959 | 98.44% | 0.00% |
| head_random0_AB | 0.6915 | 93.75% | 0.00% |
| head_random1_B | 0.4013 | 97.66% | 0.00% |
| head_random1_AB | 0.9065 | 86.72% | 0.00% |
| head_random2_B | 0.3048 | 98.44% | 0.00% |
| head_random2_AB | 0.7124 | 92.19% | 0.00% |
| clean | 0.2933 | 98.44% | 0.00% |

| Contrast | NLL effect [paired 95% CI] |
|---|---:|
| head_interaction_dynamic | +0.1522 [+0.1201, +0.1864] |
| head_random0_interaction | -0.0023 [-0.0151, +0.0150] |
| head_random1_interaction | +0.1074 [+0.0782, +0.1378] |
| head_random2_interaction | +0.0096 [-0.0004, +0.0200] |

Effective-alpha relative change after A: None

### transfer: 64 independent pairs

| Arm | Correct NLL | Correct full-vocabulary accuracy | Donor full-vocabulary accuracy |
|---|---:|---:|---:|
| A_dynamic | 0.7462 | 93.75% | 0.00% |
| head_B_dynamic | 0.2589 | 100.00% | 0.00% |
| head_AB_dynamic | 0.8796 | 92.19% | 0.00% |
| head_random0_B | 0.2399 | 99.22% | 0.00% |
| head_random0_AB | 0.7371 | 94.53% | 0.00% |
| head_random1_B | 0.2983 | 100.00% | 0.00% |
| head_random1_AB | 0.9473 | 91.41% | 0.00% |
| head_random2_B | 0.2497 | 100.00% | 0.00% |
| head_random2_AB | 0.7641 | 94.53% | 0.00% |
| clean | 0.2434 | 99.22% | 0.00% |

| Contrast | NLL effect [paired 95% CI] |
|---|---:|
| head_interaction_dynamic | +0.1180 [+0.0902, +0.1475] |
| head_random0_interaction | -0.0056 [-0.0186, +0.0063] |
| head_random1_interaction | +0.1462 [+0.1051, +0.1867] |
| head_random2_interaction | +0.0116 [-0.0005, +0.0239] |

Effective-alpha relative change after A: None

### unchanged_query: 64 independent pairs

| Arm | Correct NLL | Correct full-vocabulary accuracy | Donor full-vocabulary accuracy |
|---|---:|---:|---:|
| A_dynamic | 0.5937 | 95.31% | 0.00% |
| head_B_dynamic | 0.3671 | 96.88% | 0.00% |
| head_AB_dynamic | 0.7293 | 93.75% | 0.00% |
| head_random0_B | 0.3423 | 96.88% | 0.00% |
| head_random0_AB | 0.5910 | 95.31% | 0.00% |
| head_random1_B | 0.4194 | 95.31% | 0.00% |
| head_random1_AB | 0.7793 | 93.75% | 0.00% |
| head_random2_B | 0.3391 | 96.88% | 0.00% |
| head_random2_AB | 0.5996 | 95.31% | 0.00% |
| clean | 0.3378 | 96.88% | 0.00% |

| Contrast | NLL effect [paired 95% CI] |
|---|---:|
| head_interaction_dynamic | +0.1063 [+0.0769, +0.1362] |
| head_random0_interaction | -0.0071 [-0.0160, +0.0012] |
| head_random1_interaction | +0.1041 [+0.0743, +0.1385] |
| head_random2_interaction | +0.0047 [-0.0032, +0.0145] |

Effective-alpha relative change after A: None

All directed-case values and discovery scans are in results.json. Pair bootstrap uses 4,000 resamples, grouping exchange directions; intervals are unadjusted and describe examples rather than checkpoint seeds. No weights were updated.

This first batch reuses the previous message-patching inputs and is exploratory. The parent directory contains the separately seeded confirmation.
