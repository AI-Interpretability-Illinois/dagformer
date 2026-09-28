# 96-head support: three mask seeds

Both frozen models were rerun with the same discovery, validation and test sequences. Only mask initialization and optimization minibatch order change. Full-model per-sequence endpoints match across all runs. These are mask-optimization repeats, not independently pretrained models.

## Copy retention

Normalized accuracy loss is relative to each run's full model. Paired 95% intervals resample sequences within that mask seed and are unadjusted for multiple comparisons.

| Dataset | Mask seed | Dense loss, % | DAG loss, % | DAG − dense, 95% CI |
|---|---:|---:|---:|---:|
| Copy p128 / 512 | 20260930 | 0.70 | 0.10 | -0.60 [-1.60, 0.40] |
| Copy p128 / 512 | 20261001 | -0.50 | 0.98 | 1.48 [0.49, 2.49] |
| Copy p128 / 512 | 20261002 | 1.20 | 0.69 | -0.51 [-1.69, 0.59] |
| Copy p128 / 1024 | 20260930 | -3.21 | -3.85 | -0.64 [-3.21, 1.24] |
| Copy p128 / 1024 | 20261001 | -3.73 | 0.42 | 4.15 [0.88, 8.46] |
| Copy p128 / 1024 | 20261002 | -2.49 | -1.98 | 0.51 [-1.41, 2.84] |
| Copy p256 / 1024 | 20260930 | -1.26 | -3.34 | -2.08 [-6.44, 1.08] |
| Copy p256 / 1024 | 20261001 | -1.69 | 0.94 | 2.63 [-0.20, 6.12] |
| Copy p256 / 1024 | 20261002 | 2.42 | -2.09 | -4.51 [-7.26, -2.19] |

## Natural-text NLL cost

| Mask seed | Dense ΔNLL | DAG ΔNLL | DAG − dense, 95% CI |
|---|---:|---:|---:|
| 20260930 | 0.5180 | 0.9524 | 0.4344 [0.3502, 0.5143] |
| 20261001 | 0.3655 | 0.7611 | 0.3956 [0.3023, 0.4927] |
| 20261002 | 0.5008 | 0.7389 | 0.2381 [0.1637, 0.3132] |

## Across-mask-seed ranges

Each entry is the mean across three mask seeds followed by their minimum and maximum. These ranges are not training-seed confidence intervals. Synthetic damage is normalized accuracy loss (%); natural-text damage is ΔNLL.

| Arm | Dataset | Dense damage mean [range] | DAG damage mean [range] |
|---|---|---:|---:|
| learned_96 | Copy p128 / 512 | 0.4667 [-0.5000, 1.2000] | 0.5900 [0.0983, 0.9833] |
| learned_96 | Copy p128 / 1024 | -3.1434 [-3.7306, -2.4870] | -1.8037 [-3.8502, 0.4162] |
| learned_96 | Copy p256 / 1024 | -0.1756 [-1.6860, 2.4236] | -1.4962 [-3.3403, 0.9395] |
| learned_96 | WikiText sampled positions | 0.4615 [0.3655, 0.5180] | 0.8175 [0.7389, 0.9524] |
| remove_learned_96 | Copy p128 / 512 | 98.3000 [97.4000, 99.2000] | 99.3117 [99.2134, 99.5084] |
| remove_learned_96 | Copy p128 / 1024 | 98.6528 [98.3420, 98.9637] | 99.0635 [98.1270, 99.7919] |
| remove_learned_96 | Copy p256 / 1024 | 98.2086 [97.5764, 99.0516] | 98.6778 [98.3299, 99.1649] |
| remove_learned_96 | WikiText sampled positions | 4.8575 [4.7574, 4.9342] | 5.4886 [5.1825, 5.9856] |

## KL across mask seeds

Each model uses its own dynamic full-model teacher.

| Dataset | Dense KL mean [range] | DAG KL mean [range] |
|---|---:|---:|
| Copy p128 / 512 | 0.1321 [0.1187, 0.1543] | 0.1001 [0.0835, 0.1142] |
| Copy p128 / 1024 | 0.2642 [0.2490, 0.2867] | 0.1760 [0.1128, 0.2647] |
| Copy p256 / 1024 | 0.2927 [0.2630, 0.3283] | 0.2476 [0.1974, 0.3285] |
| WikiText sampled positions | 0.5044 [0.4242, 0.5730] | 0.8674 [0.7616, 1.0298] |

Raw per-sequence arrays remain in the source JSONs. [Replication statistics](mask_seed_replication.json) retain every seed, every fixed-route reference, validation histories and pairwise support overlap. [Main experiment](summary.md).
