# Dense versus DAG: matched head-message interchange

Both frozen 300M-scale models receive exactly the same counterfactual copy pairs and interventions in 64-dimensional head messages at three known historical value-token positions. Q/K are changed before the full-dimensional norm; V before attention. No query activation or final output is directly changed. These positions are supplied by construction (oracle locations), not discovered.

The search budget is identical: 36 single-layer/stream scans, then 64 single-head scans in each model’s top four layer/stream groups. Top-1/2/4/8/16 sets are frozen before held-out evaluation. Selection is by discovery donor-margin increase. The sets are local intervention interfaces, not complete or minimal circuits. Native DAG source-edge counts are not compared here.

32 discovery pairs, 64 held-out pairs at period 64, 64 distance-transfer pairs at period 128, and 64 unchanged-query controls; both interchange directions. The data are exactly those in the native-message pilot datasets.json. CI: 4,000 paired bootstrap resamples of base blocks, unadjusted, keeping the directions together. Architecture contrasts use the same resampled blocks.

Three head-permutation controls per size preserve layer/stream counts and share the head permutation across streams within a layer. Selected/control sites can overlap. They are not norm or language-model-damage matched; actual intervention L2 is reported. The unchanged-query control measures another synthetic copy target, not general natural-text capability.

Intervention L2 is the per-case Euclidean norm pooled across all edited coordinates; relative L2 divides by the norm of the current recipient coordinates being replaced, then averages across cases. Top-k counts head/stream identities, each replaced at three oracle positions (3k spatiotemporal edits). All-QKV is a broad positive control, not a theoretical upper bound on a selected subset.

dense: 304,137,216 total parameters. Selected layer/stream groups: [[0, 'v'], [4, 'v'], [0, 'q'], [0, 'k']]. Top 8 sites: [[0, 'v', 5], [4, 'v', 1], [0, 'v', 14], [4, 'v', 2], [0, 'v', 4], [4, 'v', 7], [0, 'v', 0], [4, 'v', 4]]. Unwrapped parity error 0.0; all identity errors 0.0.

Actual selected/control overlap — dense: top1/random0: 0/1, top1/random1: 0/1, top1/random2: 0/1, top2/random0: 0/2, top2/random1: 0/2, top2/random2: 0/2, top4/random0: 1/4, top4/random1: 1/4, top4/random2: 1/4, top8/random0: 3/8, top8/random1: 2/8, top8/random2: 2/8, top16/random0: 5/16, top16/random1: 4/16, top16/random2: 6/16.

dag: 336,447,805 total parameters. Selected layer/stream groups: [[0, 'v'], [6, 'v'], [7, 'v'], [0, 'q']]. Top 8 sites: [[6, 'v', 11], [7, 'v', 1], [0, 'v', 5], [0, 'v', 0], [0, 'v', 14], [7, 'v', 11], [7, 'v', 9], [6, 'v', 13]]. Unwrapped parity error 0.0; all identity errors 0.0.

Actual selected/control overlap — dag: top1/random0: 0/1, top1/random1: 0/1, top1/random2: 0/1, top2/random0: 0/2, top2/random1: 0/2, top2/random2: 0/2, top4/random0: 0/4, top4/random1: 0/4, top4/random2: 0/4, top8/random0: 2/8, top8/random1: 1/8, top8/random2: 0/8, top16/random0: 5/16, top16/random1: 4/16, top16/random2: 5/16.

## heldout

Pairs for which both models are correct in both directions: 61/64. Full-sample metrics below; common-success contrasts are retained in comparison.json.

| Model | Arm | Margin recovery [95% CI] | Donor full-vocab accuracy | Donor two-choice accuracy | Intervention L2 | Relative L2 |
|---|---|---:|---:|---:|---:|---:|
| dense | reference | 0.000 [0.000, 0.000] | 0.00% | 0.00% | 0.00 | 0.000 |
| dag | reference | 0.000 [0.000, 0.000] | 0.00% | 0.00% | 0.00 | 0.000 |
| dense | donor_baseline | 1.000 [1.000, 1.000] | 98.44% | 100.00% | 0.00 | 0.000 |
| dag | donor_baseline | 1.000 [1.000, 1.000] | 99.22% | 100.00% | 0.00 | 0.000 |
| dense | all_qkv | 1.000 [1.000, 1.000] | 98.44% | 100.00% | 27.60 | 0.020 |
| dag | all_qkv | 1.000 [1.000, 1.000] | 99.22% | 100.00% | 603.11 | 0.216 |
| dense | top1 | 0.153 [0.138, 0.169] | 0.78% | 3.12% | 0.94 | 1.430 |
| dag | top1 | 0.313 [0.295, 0.331] | 4.69% | 6.25% | 120.00 | 1.438 |
| dense | top1/random0 | 0.017 [0.014, 0.020] | 0.00% | 0.00% | 0.67 | 1.469 |
| dag | top1/random0 | 0.035 [0.028, 0.042] | 0.00% | 0.00% | 74.89 | 1.238 |
| dense | top1/random1 | 0.001 [0.000, 0.002] | 0.00% | 0.00% | 0.63 | 1.407 |
| dag | top1/random1 | 0.035 [0.028, 0.042] | 0.00% | 0.00% | 74.89 | 1.238 |
| dense | top1/random2 | 0.002 [0.001, 0.004] | 0.00% | 0.00% | 0.59 | 1.417 |
| dag | top1/random2 | 0.000 [-0.000, 0.000] | 0.00% | 0.00% | 28.55 | 1.254 |
| dense | top2 | 0.302 [0.286, 0.320] | 4.69% | 8.59% | 48.78 | 1.217 |
| dag | top2 | 0.457 [0.445, 0.470] | 25.00% | 33.59% | 138.83 | 1.424 |
| dense | top2/random0 | 0.135 [0.124, 0.146] | 0.00% | 0.78% | 52.33 | 1.333 |
| dag | top2/random0 | 0.035 [0.028, 0.042] | 0.00% | 0.00% | 75.93 | 1.238 |
| dense | top2/random1 | 0.054 [0.048, 0.060] | 0.00% | 0.00% | 45.40 | 1.350 |
| dag | top2/random1 | 0.034 [0.028, 0.042] | 0.00% | 0.00% | 75.56 | 1.239 |
| dense | top2/random2 | 0.120 [0.109, 0.131] | 0.00% | 0.78% | 53.20 | 1.341 |
| dag | top2/random2 | 0.000 [0.000, 0.001] | 0.00% | 0.00% | 31.17 | 1.251 |
| dense | top4 | 0.658 [0.631, 0.686] | 65.62% | 78.91% | 50.86 | 0.987 |
| dag | top4 | 0.572 [0.555, 0.589] | 69.53% | 76.56% | 110.50 | 1.203 |
| dense | top4/random0 | 0.136 [0.125, 0.147] | 0.00% | 0.78% | 60.26 | 1.252 |
| dag | top4/random0 | 0.101 [0.087, 0.117] | 1.56% | 1.56% | 32.27 | 1.244 |
| dense | top4/random1 | 0.217 [0.198, 0.236] | 1.56% | 3.91% | 51.55 | 1.218 |
| dag | top4/random1 | 0.020 [0.016, 0.025] | 0.00% | 0.00% | 16.57 | 1.245 |
| dense | top4/random2 | 0.120 [0.109, 0.131] | 0.00% | 0.78% | 63.70 | 1.318 |
| dag | top4/random2 | 0.034 [0.029, 0.041] | 0.00% | 0.00% | 41.36 | 1.479 |
| dense | top8 | 0.936 [0.926, 0.945] | 97.66% | 99.22% | 32.85 | 0.450 |
| dag | top8 | 0.850 [0.835, 0.864] | 96.88% | 98.44% | 112.75 | 0.953 |
| dense | top8/random0 | 0.489 [0.467, 0.512] | 27.34% | 41.41% | 77.80 | 1.170 |
| dag | top8/random0 | 0.379 [0.359, 0.400] | 12.50% | 15.62% | 110.56 | 1.310 |
| dense | top8/random1 | 0.227 [0.206, 0.249] | 1.56% | 5.47% | 68.70 | 1.199 |
| dag | top8/random1 | 0.136 [0.121, 0.152] | 1.56% | 1.56% | 27.11 | 1.116 |
| dense | top8/random2 | 0.304 [0.280, 0.329] | 6.25% | 15.62% | 70.58 | 1.223 |
| dag | top8/random2 | 0.034 [0.028, 0.041] | 0.00% | 0.00% | 57.17 | 1.354 |
| dense | top16 | 0.971 [0.965, 0.976] | 97.66% | 100.00% | 22.27 | 0.297 |
| dag | top16 | 0.915 [0.905, 0.925] | 96.88% | 100.00% | 119.55 | 0.947 |
| dense | top16/random0 | 0.737 [0.710, 0.764] | 81.25% | 91.41% | 51.95 | 0.807 |
| dag | top16/random0 | 0.646 [0.626, 0.666] | 78.91% | 84.38% | 99.73 | 1.038 |
| dense | top16/random1 | 0.244 [0.219, 0.270] | 3.12% | 8.59% | 67.83 | 1.185 |
| dag | top16/random1 | 0.296 [0.271, 0.325] | 7.81% | 14.06% | 38.92 | 1.110 |
| dense | top16/random2 | 0.546 [0.515, 0.575] | 40.62% | 61.72% | 55.50 | 1.008 |
| dag | top16/random2 | 0.442 [0.416, 0.470] | 28.91% | 35.94% | 75.37 | 1.022 |

| Arm | DAG−dense recovery [95% CI] | DAG−dense donor full-vocab accuracy [95% CI] |
|---|---:|---:|
| all_qkv | -0.000 [-0.000, +0.000] | +0.78 pp [-1.56, +3.91] |
| top1 | +0.160 [+0.133, +0.186] | +3.91 pp [+0.00, +7.81] |
| top2 | +0.155 [+0.135, +0.175] | +20.31 pp [+14.06, +27.34] |
| top4 | -0.086 [-0.120, -0.052] | +3.91 pp [-6.25, +14.06] |
| top8 | -0.086 [-0.100, -0.073] | -0.78 pp [-3.91, +2.34] |
| top16 | -0.056 [-0.067, -0.045] | -0.78 pp [-5.47, +3.12] |

## transfer

Pairs for which both models are correct in both directions: 63/64. Full-sample metrics below; common-success contrasts are retained in comparison.json.

| Model | Arm | Margin recovery [95% CI] | Donor full-vocab accuracy | Donor two-choice accuracy | Intervention L2 | Relative L2 |
|---|---|---:|---:|---:|---:|---:|
| dense | reference | 0.000 [0.000, 0.000] | 0.00% | 0.00% | 0.00 | 0.000 |
| dag | reference | 0.000 [0.000, 0.000] | 0.00% | 0.00% | 0.00 | 0.000 |
| dense | donor_baseline | 1.000 [1.000, 1.000] | 99.22% | 100.00% | 0.00 | 0.000 |
| dag | donor_baseline | 1.000 [1.000, 1.000] | 100.00% | 100.00% | 0.00 | 0.000 |
| dense | all_qkv | 1.000 [1.000, 1.000] | 99.22% | 100.00% | 26.39 | 0.020 |
| dag | all_qkv | 1.000 [1.000, 1.000] | 100.00% | 100.00% | 613.59 | 0.223 |
| dense | top1 | 0.170 [0.156, 0.184] | 0.00% | 0.78% | 0.92 | 1.415 |
| dag | top1 | 0.374 [0.351, 0.399] | 17.19% | 19.53% | 112.25 | 1.398 |
| dense | top1/random0 | 0.019 [0.014, 0.023] | 0.00% | 0.00% | 0.65 | 1.436 |
| dag | top1/random0 | 0.016 [0.011, 0.022] | 0.00% | 0.00% | 65.43 | 1.141 |
| dense | top1/random1 | 0.000 [-0.000, 0.001] | 0.00% | 0.00% | 0.59 | 1.370 |
| dag | top1/random1 | 0.016 [0.011, 0.022] | 0.00% | 0.00% | 65.43 | 1.141 |
| dense | top1/random2 | 0.001 [0.001, 0.002] | 0.00% | 0.00% | 0.58 | 1.409 |
| dag | top1/random2 | 0.000 [-0.000, 0.000] | 0.00% | 0.00% | 26.81 | 1.211 |
| dense | top2 | 0.351 [0.335, 0.368] | 4.69% | 11.72% | 48.53 | 1.191 |
| dag | top2 | 0.541 [0.525, 0.558] | 59.38% | 66.41% | 133.19 | 1.404 |
| dense | top2/random0 | 0.147 [0.134, 0.160] | 0.00% | 0.00% | 50.28 | 1.270 |
| dag | top2/random0 | 0.017 [0.011, 0.023] | 0.00% | 0.00% | 66.61 | 1.141 |
| dense | top2/random1 | 0.074 [0.064, 0.084] | 0.00% | 0.00% | 46.45 | 1.301 |
| dag | top2/random1 | 0.016 [0.011, 0.022] | 0.00% | 0.00% | 66.10 | 1.143 |
| dense | top2/random2 | 0.129 [0.117, 0.141] | 0.00% | 0.00% | 51.00 | 1.280 |
| dag | top2/random2 | 0.001 [0.000, 0.001] | 0.00% | 0.00% | 29.58 | 1.203 |
| dense | top4 | 0.689 [0.670, 0.707] | 72.66% | 88.28% | 49.33 | 0.939 |
| dag | top4 | 0.644 [0.625, 0.662] | 78.91% | 81.25% | 103.38 | 1.163 |
| dense | top4/random0 | 0.147 [0.134, 0.160] | 0.00% | 0.00% | 58.27 | 1.176 |
| dag | top4/random0 | 0.109 [0.096, 0.125] | 1.56% | 1.56% | 30.66 | 1.213 |
| dense | top4/random1 | 0.246 [0.228, 0.265] | 0.00% | 5.47% | 52.73 | 1.195 |
| dag | top4/random1 | 0.018 [0.015, 0.021] | 0.00% | 0.00% | 16.66 | 1.154 |
| dense | top4/random2 | 0.129 [0.117, 0.141] | 0.00% | 0.00% | 61.95 | 1.268 |
| dag | top4/random2 | 0.027 [0.022, 0.033] | 0.00% | 0.00% | 32.67 | 1.367 |
| dense | top8 | 0.935 [0.925, 0.944] | 96.88% | 100.00% | 34.93 | 0.479 |
| dag | top8 | 0.874 [0.862, 0.886] | 100.00% | 100.00% | 111.17 | 0.940 |
| dense | top8/random0 | 0.553 [0.532, 0.575] | 48.44% | 67.19% | 75.18 | 1.116 |
| dag | top8/random0 | 0.430 [0.407, 0.454] | 27.34% | 32.81% | 103.71 | 1.278 |
| dense | top8/random1 | 0.253 [0.235, 0.271] | 0.00% | 5.47% | 69.62 | 1.175 |
| dag | top8/random1 | 0.132 [0.122, 0.143] | 0.00% | 0.00% | 26.86 | 1.025 |
| dense | top8/random2 | 0.328 [0.307, 0.352] | 4.69% | 10.94% | 68.14 | 1.164 |
| dag | top8/random2 | 0.027 [0.022, 0.033] | 0.00% | 0.00% | 50.27 | 1.300 |
| dense | top16 | 0.970 [0.963, 0.976] | 96.88% | 100.00% | 23.02 | 0.309 |
| dag | top16 | 0.939 [0.930, 0.947] | 99.22% | 100.00% | 113.69 | 0.904 |
| dense | top16/random0 | 0.762 [0.742, 0.781] | 85.94% | 94.53% | 49.60 | 0.753 |
| dag | top16/random0 | 0.702 [0.682, 0.722] | 90.62% | 92.19% | 93.78 | 0.990 |
| dense | top16/random1 | 0.266 [0.246, 0.286] | 2.34% | 6.25% | 69.03 | 1.170 |
| dag | top16/random1 | 0.284 [0.263, 0.305] | 7.03% | 9.38% | 33.94 | 0.999 |
| dense | top16/random2 | 0.557 [0.531, 0.582] | 50.00% | 67.19% | 55.13 | 0.982 |
| dag | top16/random2 | 0.408 [0.382, 0.434] | 22.66% | 25.78% | 71.89 | 0.996 |

| Arm | DAG−dense recovery [95% CI] | DAG−dense donor full-vocab accuracy [95% CI] |
|---|---:|---:|
| all_qkv | +0.000 [-0.000, +0.000] | +0.78 pp [+0.00, +2.34] |
| top1 | +0.205 [+0.176, +0.234] | +17.19 pp [+9.38, +25.78] |
| top2 | +0.190 [+0.169, +0.211] | +54.69 pp [+46.09, +63.28] |
| top4 | -0.045 [-0.071, -0.019] | +6.25 pp [-3.91, +16.41] |
| top8 | -0.061 [-0.073, -0.049] | +3.12 pp [+0.78, +6.25] |
| top16 | -0.031 [-0.042, -0.021] | +2.34 pp [+0.00, +5.47] |

## Unchanged-query control

| Model | Arm | Correct-answer logp change [95% CI] | Correct full-vocab accuracy |
|---|---|---:|---:|
| dense | reference | +0.0000 [+0.0000, +0.0000] | 96.88% |
| dag | reference | +0.0000 [+0.0000, +0.0000] | 98.44% |
| dense | all_qkv | +0.0000 [+0.0000, +0.0000] | 96.88% |
| dag | all_qkv | +0.0009 [-0.0003, +0.0024] | 98.44% |
| dense | top1 | +0.0012 [-0.0002, +0.0031] | 96.88% |
| dag | top1 | +0.0002 [-0.0002, +0.0007] | 98.44% |
| dense | top2 | +0.0007 [-0.0005, +0.0023] | 96.88% |
| dag | top2 | +0.0002 [-0.0003, +0.0009] | 98.44% |
| dense | top4 | +0.0023 [+0.0009, +0.0042] | 96.88% |
| dag | top4 | +0.0027 [-0.0003, +0.0079] | 98.44% |
| dense | top8 | +0.0012 [-0.0002, +0.0030] | 96.88% |
| dag | top8 | +0.0017 [-0.0005, +0.0054] | 98.44% |
| dense | top16 | +0.0016 [+0.0002, +0.0033] | 96.88% |
| dag | top16 | +0.0020 [-0.0011, +0.0073] | 98.44% |

All per-case full-vocabulary log probabilities, margins, two-choice metrics, output KL and intervention L2 are in results.json. The comparison uses one checkpoint per architecture, different total parameter counts, synthetic token-marginal repetition, and known intervention positions. No model was trained. Search considers only heads in the four discovery-selected layer/stream groups; it does not establish the globally smallest set.

Run: `CUDA_VISIBLE_DEVICES=2 /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python scripts/circuit_common_message_patching.py`.
