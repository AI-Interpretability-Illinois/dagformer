# Counterfactual routing-message interchange — 300M FourWay

This is a fixed-checkpoint, inference-only synthetic-copy experiment. It does not establish a dense-versus-DAG interpretability advantage or natural-language circuit semantics.

The experiment finds a transferable information-read interface: exchanging projected source content changes the copied answer, while exchanging the coefficients alone has little effect. The masks were selected once on discovery data. All predeclared top-k sizes and three controls per size are reported below; top-8 is an illustrative sparse setting, not a held-out optimum.

Three full token-marginal random blocks are followed by a shared unfinished fourth prefix. A counterfactual pair differs only at three historical occurrences of one value; the next-token query prefix is identical. Both exchange directions are evaluated. Interventions affect reads at those three past value positions, never query states or final outputs.

These historical value-token locations are supplied by the task construction (oracle locations), not discovered by the mask search. This tests information transmission at known locations. The two candidate value tokens are sampled from the WikiText training-token marginal subject to being distinct and absent from the rest of their random block; hence they are also absent from the unfinished query prefix.

Discovery uses 32 independent pairs at period 64. Held-out evaluation uses 64 new pairs at period 64; distance transfer uses 64 new pairs at period 128. The unchanged-query control reuses held-out contexts but asks for a different, unchanged block position. Exact token IDs are retained in datasets.json.

Layer selection (best two KV layers), head selection (best four heads), and edge ranking all use only discovery donor-margin increases from complete-message interventions. Top-k masks are single-edge rankings, not proven minimal sufficient circuits. Random controls preserve layer, stream, source, head grouping, and edge count while moving the selected heads to other heads; neither edit norm nor language-model damage is matched.

For projected source content z and effective coefficient a=pred+corr, alpha/content/message exchange uses donor-a times current-recipient-z, current-recipient-a times donor-z, or donor-a times donor-z respectively. Dynamic arms recompute downstream correction normally. Fixed arms hold recipient effective coefficients at all layers/streams/positions before the selected exchange, so they measure content flow conditional on the recipient routing.

Confidence intervals are unadjusted 95% pair bootstrap intervals (4,000 resamples), grouping the two directions of each independent base block. Margin recovery is (patched-minus-recipient donor margin)/(donor-minus-recipient donor margin), where donor margin is log p(donor answer)-log p(recipient answer). Values can lie outside [0,1].

Identity intervention and last-token projection checks: {'last_logit_projection_max_error': 0.0}; maximum identity logit difference on all stages is 0.0.

## Selected paths

Layers: [6, 7]; heads (zero-based layer/head): [[6, 11], [7, 1], [7, 11], [7, 9]]. Exact ordered edges and discovery scores are in results.json.

The broad discovery scan includes routed layers 1 through 11. Selected heads and top-k edges use layers [6, 7]; final layer 11 is absent. The first eight edges use streams ['v'] and sources [0, 1, 2]. Source 0 is the embedding, and source i>0 is layer-(i-1) output. The 300M checkpoint has no V norm. These are pathway interventions in a model whose upstream computation remains present, not eight self-contained computational operations.

Each selected edge is instantiated at all three oracle positions: top-8 therefore replaces 24 spatiotemporal message instances, each carrying a 64-dimensional head projection.

## heldout

Baseline full-vocabulary accuracy: 99.22%; both baseline directions correct: 63/64 pairs. Period 64.

| Arm | Margin recovery [95% CI] | Donor two-choice accuracy | Donor full-vocab accuracy | Donor logp |
|---|---:|---:|---:|---:|
| all/alpha/dynamic | 0.029 [0.022, 0.036] | 0.78% | 0.00% | -14.032 |
| all/content/dynamic | 0.998 [0.998, 0.999] | 100.00% | 99.22% | -0.139 |
| all/message/dynamic | 0.999 [0.998, 0.999] | 100.00% | 99.22% | -0.138 |
| all/alpha/fixed | 0.020 [0.014, 0.026] | 0.78% | 0.00% | -14.235 |
| all/content/fixed | 0.967 [0.956, 0.978] | 99.22% | 97.66% | -0.382 |
| all/message/fixed | 0.978 [0.969, 0.986] | 99.22% | 97.66% | -0.288 |
| stream/q | 0.035 [0.026, 0.044] | 0.00% | 0.00% | -13.781 |
| stream/k | 0.089 [0.077, 0.102] | 0.78% | 0.00% | -12.875 |
| stream/v | 0.955 [0.942, 0.967] | 100.00% | 93.75% | -0.520 |
| stream/r | 0.349 [0.334, 0.366] | 15.62% | 7.81% | -5.779 |
| stream/kv | 0.999 [0.998, 0.999] | 100.00% | 99.22% | -0.137 |
| stream/qkv | 0.999 [0.998, 0.999] | 100.00% | 99.22% | -0.138 |
| layer/6/kv | 0.397 [0.373, 0.420] | 21.88% | 18.75% | -5.119 |
| layer/7/kv | 0.314 [0.290, 0.339] | 7.03% | 3.91% | -7.548 |
| selected_four_heads | 0.598 [0.578, 0.617] | 83.59% | 71.09% | -1.777 |
| top4/alpha | 0.001 [-0.000, 0.002] | 0.00% | 0.00% | -14.732 |
| top4/content | 0.387 [0.372, 0.401] | 16.41% | 13.28% | -4.713 |
| top4/message | 0.385 [0.370, 0.400] | 12.50% | 8.59% | -4.637 |
| top4/message_fixed | 0.383 [0.367, 0.398] | 13.28% | 9.38% | -4.803 |
| top4/random0 | 0.000 [-0.000, 0.000] | 0.00% | 0.00% | -14.742 |
| top4/random1 | 0.010 [0.009, 0.012] | 0.00% | 0.00% | -14.458 |
| top4/random2 | 0.000 [0.000, 0.001] | 0.00% | 0.00% | -14.731 |
| top8/alpha | 0.003 [0.000, 0.005] | 0.00% | 0.00% | -14.697 |
| top8/content | 0.553 [0.535, 0.569] | 70.31% | 65.62% | -2.209 |
| top8/message | 0.553 [0.536, 0.570] | 71.09% | 64.06% | -2.036 |
| top8/message_fixed | 0.543 [0.524, 0.559] | 64.84% | 54.69% | -2.328 |
| top8/random0 | 0.000 [-0.000, 0.000] | 0.00% | 0.00% | -14.738 |
| top8/random1 | 0.033 [0.027, 0.040] | 0.00% | 0.00% | -13.815 |
| top8/random2 | 0.011 [0.009, 0.012] | 0.00% | 0.00% | -14.442 |
| top16/alpha | 0.011 [0.006, 0.017] | 0.78% | 0.00% | -14.507 |
| top16/content | 0.585 [0.566, 0.603] | 72.66% | 64.06% | -2.116 |
| top16/message | 0.586 [0.567, 0.604] | 79.69% | 63.28% | -1.914 |
| top16/message_fixed | 0.579 [0.560, 0.598] | 75.00% | 60.94% | -2.144 |
| top16/random0 | -0.001 [-0.004, -0.000] | 0.00% | 0.00% | -14.758 |
| top16/random1 | 0.038 [0.032, 0.045] | 0.00% | 0.00% | -13.653 |
| top16/random2 | 0.010 [0.007, 0.012] | 0.00% | 0.00% | -14.453 |
| top32/alpha | 0.012 [0.007, 0.017] | 0.78% | 0.00% | -14.483 |
| top32/content | 0.592 [0.572, 0.611] | 76.56% | 69.53% | -2.129 |
| top32/message | 0.597 [0.578, 0.616] | 82.81% | 70.31% | -1.838 |
| top32/message_fixed | 0.590 [0.570, 0.609] | 79.69% | 64.06% | -2.067 |
| top32/random0 | -0.001 [-0.004, 0.000] | 0.00% | 0.00% | -14.751 |
| top32/random1 | 0.039 [0.032, 0.046] | 0.00% | 0.00% | -13.640 |
| top32/random2 | 0.010 [0.007, 0.013] | 0.00% | 0.00% | -14.436 |
| reference | 0.000 [0.000, 0.000] | 0.00% | 0.00% | -14.743 |
| donor_baseline | 1.000 [1.000, 1.000] | 100.00% | 99.22% | -0.135 |

## transfer

Baseline full-vocabulary accuracy: 100.00%; both baseline directions correct: 64/64 pairs. Period 128.

| Arm | Margin recovery [95% CI] | Donor two-choice accuracy | Donor full-vocab accuracy | Donor logp |
|---|---:|---:|---:|---:|
| all/alpha/dynamic | 0.014 [0.008, 0.020] | 0.00% | 0.00% | -13.647 |
| all/content/dynamic | 0.998 [0.998, 0.999] | 100.00% | 100.00% | -0.079 |
| all/message/dynamic | 0.999 [0.998, 0.999] | 100.00% | 100.00% | -0.077 |
| all/alpha/fixed | 0.012 [0.006, 0.018] | 0.00% | 0.00% | -13.688 |
| all/content/fixed | 0.980 [0.972, 0.986] | 100.00% | 100.00% | -0.153 |
| all/message/fixed | 0.987 [0.982, 0.992] | 100.00% | 100.00% | -0.106 |
| stream/q | 0.020 [0.012, 0.029] | 0.00% | 0.00% | -13.496 |
| stream/k | 0.069 [0.059, 0.080] | 0.00% | 0.00% | -12.547 |
| stream/v | 0.968 [0.958, 0.978] | 100.00% | 96.88% | -0.320 |
| stream/r | 0.325 [0.310, 0.339] | 6.25% | 6.25% | -5.935 |
| stream/kv | 0.999 [0.998, 0.999] | 100.00% | 100.00% | -0.077 |
| stream/qkv | 0.999 [0.998, 0.999] | 100.00% | 100.00% | -0.077 |
| layer/6/kv | 0.427 [0.400, 0.454] | 31.25% | 24.22% | -4.465 |
| layer/7/kv | 0.327 [0.299, 0.355] | 11.72% | 7.81% | -6.935 |
| selected_four_heads | 0.685 [0.667, 0.703] | 96.09% | 89.06% | -1.154 |
| top4/alpha | -0.000 [-0.001, 0.001] | 0.00% | 0.00% | -14.008 |
| top4/content | 0.439 [0.425, 0.454] | 31.25% | 28.91% | -3.632 |
| top4/message | 0.439 [0.424, 0.453] | 26.56% | 23.44% | -3.548 |
| top4/message_fixed | 0.438 [0.423, 0.452] | 26.56% | 23.44% | -3.683 |
| top4/random0 | -0.000 [-0.000, 0.000] | 0.00% | 0.00% | -14.009 |
| top4/random1 | 0.007 [0.006, 0.008] | 0.00% | 0.00% | -13.827 |
| top4/random2 | 0.001 [0.000, 0.001] | 0.00% | 0.00% | -13.992 |
| top8/alpha | 0.000 [-0.001, 0.001] | 0.00% | 0.00% | -14.004 |
| top8/content | 0.624 [0.606, 0.641] | 84.38% | 79.69% | -1.434 |
| top8/message | 0.625 [0.607, 0.641] | 86.72% | 82.81% | -1.338 |
| top8/message_fixed | 0.614 [0.597, 0.631] | 83.59% | 77.34% | -1.625 |
| top8/random0 | 0.000 [-0.000, 0.000] | 0.00% | 0.00% | -14.002 |
| top8/random1 | 0.029 [0.023, 0.036] | 0.00% | 0.00% | -13.213 |
| top8/random2 | 0.007 [0.006, 0.008] | 0.00% | 0.00% | -13.811 |
| top16/alpha | 0.002 [-0.001, 0.006] | 0.00% | 0.00% | -13.946 |
| top16/content | 0.662 [0.644, 0.679] | 90.62% | 82.03% | -1.406 |
| top16/message | 0.664 [0.647, 0.681] | 92.19% | 85.16% | -1.279 |
| top16/message_fixed | 0.655 [0.638, 0.672] | 91.41% | 82.03% | -1.517 |
| top16/random0 | -0.000 [-0.001, -0.000] | 0.00% | 0.00% | -14.015 |
| top16/random1 | 0.031 [0.024, 0.038] | 0.00% | 0.00% | -13.182 |
| top16/random2 | 0.008 [0.007, 0.010] | 0.00% | 0.00% | -13.781 |
| top32/alpha | 0.003 [-0.000, 0.007] | 0.00% | 0.00% | -13.931 |
| top32/content | 0.681 [0.663, 0.699] | 93.75% | 85.16% | -1.316 |
| top32/message | 0.686 [0.670, 0.703] | 96.09% | 87.50% | -1.162 |
| top32/message_fixed | 0.677 [0.660, 0.694] | 92.97% | 80.47% | -1.391 |
| top32/random0 | 0.000 [0.000, 0.001] | 0.00% | 0.00% | -13.993 |
| top32/random1 | 0.033 [0.026, 0.041] | 0.00% | 0.00% | -13.120 |
| top32/random2 | 0.009 [0.007, 0.011] | 0.00% | 0.00% | -13.762 |
| reference | 0.000 [0.000, 0.000] | 0.00% | 0.00% | -14.007 |
| donor_baseline | 1.000 [1.000, 1.000] | 100.00% | 100.00% | -0.076 |

## Unchanged-query control

The correct answer is unchanged across donor and recipient. Here donor_probability means the probability of the wrong changed value from the donor; it is not a donor-correctness score. This control measures selectivity across two synthetic copy targets, not general language-model capability.

| Arm | Correct-answer logp change | Correct full-vocab accuracy | Wrong changed-value probability |
|---|---:|---:|---:|
| all/alpha/dynamic | -0.0009 [-0.0025, +0.0003] | 98.44% | 0.00004 |
| all/content/dynamic | -0.0008 [-0.0020, +0.0001] | 98.44% | 0.00008 |
| all/message/dynamic | +0.0001 [-0.0011, +0.0012] | 98.44% | 0.00008 |
| all/alpha/fixed | -0.0004 [-0.0016, +0.0005] | 98.44% | 0.00004 |
| all/content/fixed | -0.0005 [-0.0017, +0.0004] | 98.44% | 0.00009 |
| all/message/fixed | -0.0004 [-0.0015, +0.0010] | 98.44% | 0.00008 |
| stream/q | -0.0015 [-0.0028, -0.0003] | 98.44% | 0.00004 |
| stream/k | +0.0007 [-0.0016, +0.0035] | 98.44% | 0.00004 |
| stream/v | +0.0009 [-0.0009, +0.0033] | 98.44% | 0.00004 |
| stream/r | -0.0005 [-0.0016, +0.0003] | 98.44% | 0.00006 |
| stream/kv | -0.0003 [-0.0013, +0.0005] | 98.44% | 0.00008 |
| stream/qkv | +0.0001 [-0.0013, +0.0014] | 98.44% | 0.00008 |
| layer/6/kv | -0.0004 [-0.0012, +0.0003] | 98.44% | 0.00004 |
| layer/7/kv | -0.0000 [-0.0007, +0.0007] | 98.44% | 0.00004 |
| selected_four_heads | -0.0005 [-0.0014, +0.0004] | 98.44% | 0.00004 |
| top4/alpha | -0.0002 [-0.0007, +0.0001] | 98.44% | 0.00004 |
| top4/content | +0.0004 [-0.0001, +0.0010] | 98.44% | 0.00004 |
| top4/message | +0.0003 [-0.0002, +0.0011] | 98.44% | 0.00004 |
| top4/message_fixed | -0.0001 [-0.0004, +0.0002] | 98.44% | 0.00004 |
| top4/random0 | -0.0000 [-0.0006, +0.0006] | 98.44% | 0.00004 |
| top4/random1 | -0.0001 [-0.0004, +0.0001] | 98.44% | 0.00004 |
| top4/random2 | +0.0000 [-0.0009, +0.0011] | 98.44% | 0.00004 |
| top8/alpha | +0.0003 [-0.0002, +0.0011] | 98.44% | 0.00004 |
| top8/content | +0.0001 [-0.0004, +0.0008] | 98.44% | 0.00004 |
| top8/message | -0.0001 [-0.0004, +0.0003] | 98.44% | 0.00004 |
| top8/message_fixed | -0.0003 [-0.0007, -0.0000] | 98.44% | 0.00004 |
| top8/random0 | -0.0007 [-0.0016, -0.0000] | 98.44% | 0.00004 |
| top8/random1 | +0.0004 [-0.0001, +0.0011] | 98.44% | 0.00004 |
| top8/random2 | +0.0002 [-0.0004, +0.0012] | 98.44% | 0.00004 |
| top16/alpha | -0.0000 [-0.0009, +0.0010] | 98.44% | 0.00004 |
| top16/content | -0.0003 [-0.0009, +0.0002] | 98.44% | 0.00004 |
| top16/message | +0.0003 [-0.0004, +0.0012] | 98.44% | 0.00004 |
| top16/message_fixed | -0.0003 [-0.0010, +0.0004] | 98.44% | 0.00004 |
| top16/random0 | -0.0001 [-0.0008, +0.0005] | 98.44% | 0.00004 |
| top16/random1 | -0.0007 [-0.0023, +0.0003] | 98.44% | 0.00004 |
| top16/random2 | -0.0002 [-0.0007, +0.0003] | 98.44% | 0.00004 |
| top32/alpha | -0.0002 [-0.0006, +0.0002] | 98.44% | 0.00004 |
| top32/content | -0.0002 [-0.0006, +0.0001] | 98.44% | 0.00004 |
| top32/message | -0.0008 [-0.0023, +0.0004] | 98.44% | 0.00004 |
| top32/message_fixed | -0.0009 [-0.0025, +0.0003] | 98.44% | 0.00004 |
| top32/random0 | +0.0004 [-0.0003, +0.0014] | 98.44% | 0.00004 |
| top32/random1 | +0.0002 [-0.0009, +0.0015] | 98.44% | 0.00004 |
| top32/random2 | -0.0003 [-0.0009, +0.0003] | 98.44% | 0.00004 |
| reference | +0.0000 [+0.0000, +0.0000] | 98.44% | 0.00004 |
| donor_baseline | +0.0000 [+0.0000, +0.0000] | 98.44% | 0.00008 |

## Scope

The native graph omits substantial upstream computation inside each source state. An edited edge is a projected source-state read, not a complete isolated circuit. Full-dimensional Q/K normalization couples heads, so an intervention can change other heads through normalization. A top-k result must be compared with direct paired random-control contrasts in control_comparisons.json; differences in intervention norm and capability cost remain unresolved.

All metrics and per-directed-case values, including recipient and donor logp, donor-minus-recipient margin, two-choice probability, full-vocabulary correctness, and output KL relative to the recipient, are retained in results.json. No model weights were trained or saved.

Run: `CUDA_VISIBLE_DEVICES=2 /scratch/yurenh2/venvs/dagformer-eval-20260917/bin/python scripts/circuit_message_patching.py --seed 20260930 --discovery-pairs 32`.
