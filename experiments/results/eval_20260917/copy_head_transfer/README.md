# Transfer of the six historical copy heads

The head set is fixed from the original localization: L4/h1, L3/h11, L6/h11, L3/h13, L3/h2, L4/h4.
No heads are reselected on these inputs. Q/K head deviations are scaled in
the predictor, correction, or both channels. Controls use the same number
of other heads per layer, matching edit norms separately per layer/token.

Each period uses 32 new 1024-token sequences, formed by repeating
blocks sampled from the WikiText training-token marginal. They are synthetic
token sequences, not contiguous natural text. Accuracy is teacher-forced
next-token accuracy beginning at the first token of the second block.
The historical localization used a Dolma-token marginal and a different seed;
absolute reference accuracies across the two runs are not directly paired.
Natural-text NLL uses WikiText validation windows. Brackets are paired normal
95% intervals over sequences/windows, unadjusted for multiple comparisons.

## Unedited reference

| Period | Accuracy after first block | Accuracy over second half |
|---|---:|---:|
| 64 | 98.54% | 99.05% |
| 128 | 95.57% | 96.32% |
| 256 | 95.13% | 96.01% |
| 512 | 75.13% | 75.13% |

Reference natural-text NLL: 3.571818.

## Named-head accuracy changes

Accuracy changes are percentage points. Gamma below one weakens head differences;
gamma above one amplifies them. All tested doses are shown.

| Channel | Gamma | Period 64 | Period 128 | Period 256 | Period 512 | Natural NLL rise |
|---|---:|---|---|---|---|---|
| pred | 0 | -3.958 [-7.540, -0.377] | -14.185 [-17.040, -11.329] | -44.291 [-46.328, -42.254] | -61.652 [-70.816, -52.487] | +0.150 [+0.138, +0.161] |
| pred | 0.5 | -0.703 [-1.673, +0.267] | -1.716 [-3.132, -0.300] | -2.551 [-3.556, -1.546] | -9.467 [-10.427, -8.506] | +0.028 [+0.025, +0.031] |
| pred | 1.5 | +0.059 [-0.094, +0.211] | +0.551 [-0.037, +1.139] | +0.456 [+0.062, +0.849] | +2.063 [+1.031, +3.095] | +0.010 [+0.008, +0.013] |
| pred | 2 | +0.160 [-0.144, +0.463] | +0.875 [-0.116, +1.867] | +0.464 [-0.058, +0.986] | +2.399 [+0.829, +3.969] | +0.035 [+0.031, +0.040] |
| corr | 0 | -9.212 [-12.320, -6.105] | -38.550 [-41.434, -35.666] | -80.131 [-83.765, -76.497] | -70.715 [-81.461, -59.970] | +0.188 [+0.171, +0.205] |
| corr | 0.5 | -0.869 [-1.634, -0.104] | -3.526 [-5.390, -1.663] | -8.854 [-9.866, -7.842] | -35.919 [-40.650, -31.188] | +0.052 [+0.045, +0.060] |
| corr | 1.5 | +0.007 [-0.188, +0.201] | +0.610 [+0.121, +1.100] | +0.598 [+0.015, +1.182] | +4.358 [+2.342, +6.374] | +0.015 [+0.012, +0.018] |
| corr | 2 | -0.085 [-0.481, +0.312] | +0.621 [+0.038, +1.203] | +0.358 [-0.423, +1.139] | +5.194 [+2.243, +8.146] | +0.054 [+0.048, +0.060] |
| both | 0 | -85.010 [-86.845, -83.174] | -89.994 [-94.065, -85.922] | -91.732 [-95.915, -87.548] | -72.565 [-83.596, -61.533] | +0.419 [+0.381, +0.458] |
| both | 0.5 | -4.411 [-7.793, -1.029] | -20.030 [-22.751, -17.309] | -67.692 [-70.775, -64.609] | -69.879 [-80.447, -59.311] | +0.159 [+0.145, +0.173] |
| both | 1.5 | +0.205 [+0.052, +0.359] | +0.907 [+0.122, +1.692] | +0.773 [+0.139, +1.407] | +4.968 [+2.586, +7.351] | +0.039 [+0.034, +0.043] |
| both | 2 | +0.319 [+0.056, +0.582] | +1.203 [+0.006, +2.401] | +0.826 [-0.059, +1.711] | +6.079 [+2.652, +9.506] | +0.118 [+0.109, +0.127] |

## Repetition interrupted by new tokens

After three copies of a 128-token block, the remaining tokens are independent
samples from the same marginal. True next-token NLL measures adaptation to the
new tail. False-lag probability and argmax rates refer to the token 128 positions
back, only where it differs from the actual target. Higher false-lag scores
indicate more copying of a now-wrong token, not better prediction.

Unedited tail NLL is 8.5940; false-lag probability is 0.837% and false-lag argmax rate is 2.889%.

| Channel | Gamma | Tail NLL change | False-lag probability change (points) | False-lag argmax change (points) |
|---|---:|---|---|---|
| pred | 0 | +0.086 [+0.072, +0.099] | -0.145 [-0.165, -0.125] | -0.320 [-0.518, -0.122] |
| pred | 0.5 | +0.012 [+0.006, +0.018] | -0.032 [-0.042, -0.021] | -0.285 [-0.414, -0.157] |
| pred | 1.5 | +0.018 [+0.015, +0.022] | +0.017 [+0.010, +0.024] | -0.025 [-0.175, +0.125] |
| pred | 2 | +0.045 [+0.038, +0.051] | +0.030 [+0.017, +0.043] | -0.005 [-0.178, +0.168] |
| corr | 0 | +0.056 [+0.043, +0.069] | -0.180 [-0.197, -0.162] | -0.335 [-0.515, -0.154] |
| corr | 0.5 | +0.012 [+0.005, +0.018] | -0.042 [-0.055, -0.029] | -0.177 [-0.290, -0.064] |
| corr | 1.5 | +0.030 [+0.024, +0.035] | +0.027 [+0.017, +0.037] | -0.049 [-0.195, +0.096] |
| corr | 2 | +0.088 [+0.076, +0.099] | +0.039 [+0.019, +0.059] | -0.005 [-0.177, +0.168] |
| both | 0 | +0.246 [+0.221, +0.270] | -0.256 [-0.279, -0.233] | -0.704 [-0.928, -0.480] |
| both | 0.5 | +0.062 [+0.050, +0.074] | -0.160 [-0.178, -0.142] | -0.349 [-0.511, -0.187] |
| both | 1.5 | +0.057 [+0.049, +0.065] | +0.039 [+0.023, +0.055] | +0.039 [-0.139, +0.218] |
| both | 2 | +0.134 [+0.119, +0.148] | +0.057 [+0.029, +0.085] | -0.044 [-0.234, +0.146] |

[Measured random controls](control_comparisons.md) and all per-sequence
values remain available. These are fixed-checkpoint interventions, not
a test of what separately retrained architectures can learn.
