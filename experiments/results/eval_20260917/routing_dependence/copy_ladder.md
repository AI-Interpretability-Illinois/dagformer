# Copy behavior of the trained 150M routing ladder

All models use the same 128 WikiText test windows and 16 synthetic sequences
at each repetition period. Token blocks are sampled from the same training-token
marginal. Copy accuracy scores teacher-forced predictions in the second half
of each sequence, following the original routing-dependence protocol.
Specifically, these 1,024-token runs score target indices 513–1023 (zero-based),
511 targets per sequence. The first token of the second half is omitted for every model.

All checkpoints were trained for 6,000 updates / 3.146B tokens. They have
different parameter counts; this is neither a matched-FLOPs nor a speed test.
Each cell's difference is variant minus full DAGFormer. Intervals are unadjusted
paired normal intervals across the same windows or synthetic sequences, not
uncertainty across training seeds.

| Model | Parameters | Natural-text NLL | NLL difference [95% interval] |
|---|---:|---:|---|
| full DAGFormer | 182,561,679 | 4.086310 | reference |
| dense | 152,593,152 | 4.281582 | +0.1953 [+0.1878, +0.2027] |
| static | 152,594,447 | 4.237025 | +0.1507 [+0.1444, +0.1570] |
| postable | 153,919,232 | 4.249013 | +0.1627 [+0.1565, +0.1689] |
| identcorr | 153,448,335 | 4.098992 | +0.0127 [+0.0070, +0.0184] |
| staticcorr | 153,448,335 | 4.095718 | +0.0094 [+0.0037, +0.0151] |
| lite | 154,773,120 | 4.073378 | -0.0129 [-0.0184, -0.0075] |

| Model | Period | Copy accuracy | Accuracy difference (points) [95% interval] | Copy NLL difference [95% interval] |
|---|---:|---:|---|---|
| full DAGFormer | 64 | 98.704% | reference | reference |
| full DAGFormer | 128 | 95.499% | reference | reference |
| full DAGFormer | 256 | 89.885% | reference | reference |
| dense | 64 | 92.661% | -6.0421 [-7.3753, -4.7088] | +0.4927 [+0.4379, +0.5475] |
| dense | 128 | 86.999% | -8.5005 [-9.4683, -7.5327] | +0.7091 [+0.6310, +0.7871] |
| dense | 256 | 78.315% | -11.5704 [-12.9395, -10.2014] | +0.8538 [+0.7908, +0.9169] |
| static | 64 | 95.120% | -3.5837 [-4.7899, -2.3775] | +0.2476 [+0.2023, +0.2930] |
| static | 128 | 90.986% | -4.5132 [-5.9794, -3.0470] | +0.3891 [+0.3045, +0.4738] |
| static | 256 | 82.669% | -7.2162 [-8.3833, -6.0492] | +0.5498 [+0.4963, +0.6033] |
| postable | 64 | 95.793% | -2.9110 [-4.2050, -1.6169] | +0.2544 [+0.2107, +0.2982] |
| postable | 128 | 91.475% | -4.0240 [-5.0050, -3.0430] | +0.3553 [+0.3262, +0.3843] |
| postable | 256 | 82.962% | -6.9227 [-8.2039, -5.6415] | +0.5164 [+0.4610, +0.5718] |
| identcorr | 64 | 98.361% | -0.3425 [-1.2532, +0.5682] | +0.0195 [-0.0107, +0.0496] |
| identcorr | 128 | 95.780% | +0.2813 [-0.9869, +1.5496] | -0.0324 [-0.0562, -0.0087] |
| identcorr | 256 | 91.157% | +1.2720 [+0.2639, +2.2802] | -0.1488 [-0.1912, -0.1065] |
| staticcorr | 64 | 98.593% | -0.1101 [-1.3391, +1.1189] | +0.0287 [-0.0056, +0.0629] |
| staticcorr | 128 | 95.952% | +0.4525 [-1.0784, +1.9835] | -0.0050 [-0.0332, +0.0232] |
| staticcorr | 256 | 92.282% | +2.3973 [+0.7131, +4.0814] | -0.1821 [-0.2824, -0.0818] |
| lite | 64 | 98.716% | +0.0122 [-1.0507, +1.0751] | +0.0139 [-0.0146, +0.0423] |
| lite | 128 | 96.196% | +0.6972 [-1.7656, +3.1599] | -0.0379 [-0.1272, +0.0514] |
| lite | 256 | 91.292% | +1.4066 [-0.0562, +2.8693] | -0.1453 [-0.2185, -0.0720] |

Static and identity labels describe the external predictor. Variants with
corrections still have input-dependent routing through the local hidden states.
