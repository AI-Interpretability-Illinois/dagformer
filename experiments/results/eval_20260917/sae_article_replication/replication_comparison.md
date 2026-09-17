# Fixed SAE directions on additional articles

The first sample has 128 windows from 31 WikiText test documents. The additional
sample has 64 windows from 14 other documents, excluding all first-sample documents.
Each intervention is paired with its own unedited reference; the two article
samples are not paired with each other. All directions and controls were fixed
before running the additional sample.

The criterion below requires the alpha -4 and +4 intervals to lie on opposite
sides of zero. Failure to meet it does not establish a zero effect. All intervals
are unadjusted window-bootstrap intervals. The separate block tables show
sensitivity to grouping adjacent windows.

| Feature | First / additional target counts | First -4 / +4 ΔNLL | Additional -4 ΔNLL [95% CI] | Additional +4 ΔNLL [95% CI] | Opposite signs supported, first / additional |
|---|---:|---:|---|---|---|
| 7326 | 55 / 36 | +0.0086 / +0.0689 | +0.0205 [-0.0480, +0.0720] | +0.0088 [-0.0150, +0.0331] | not established / not established |
| 1986 | 406 / 104 | +0.0315 / +0.0245 | +0.0758 [+0.0187, +0.1558] | -0.0329 [-0.1153, +0.0286] | not established / not established |
| 3583 | 246 / 117 | +0.0439 / +0.0126 | +0.0370 [-0.0016, +0.0759] | -0.0174 [-0.0464, +0.0103] | not established / not established |
| 3560 | 1139 / 422 | -0.0582 / +0.0833 | -0.0666 [-0.0886, -0.0453] | +0.0938 [+0.0724, +0.1141] | yes / yes |
| 7019 | 0 / 0 | no targets | no target tokens | no target tokens | no targets / no targets |
| 7068 | 4134 / 2276 | -0.0521 / +0.0773 | -0.0429 [-0.0506, -0.0358] | +0.0622 [+0.0538, +0.0715] | yes / yes |
| 452 | 36 / 5 | +0.0041 / +0.0526 | -0.2550 [-0.8383, +0.1989] | +0.3344 [-0.2179, +1.0526] | not established / not established |
| 4222 | 84 / 35 | -0.0563 / +0.0818 | -0.0210 [-0.0716, +0.0264] | +0.0242 [-0.0302, +0.0818] | yes / not established |

## Dose spans and costs

The signed span is +4 minus -4. Other-token NLL costs are listed at both doses,
without selecting the more favorable direction.

| Feature | First span | Additional span [95% CI] | Additional other-token ΔNLL -4 / +4 |
|---|---:|---|---:|
| 7326 | +0.0603 | -0.0117 [-0.0825, +0.0775] | +0.0357 / +0.0277 |
| 1986 | -0.0070 | -0.1086 [-0.2671, +0.0074] | +0.0169 / +0.0127 |
| 3583 | -0.0313 | -0.0545 [-0.1177, +0.0088] | +0.0343 / +0.0169 |
| 3560 | +0.1415 | +0.1604 [+0.1186, +0.2009] | +0.0150 / +0.0119 |
| 7019 | no targets | no target tokens | +0.0289 / +0.0350 |
| 7068 | +0.1294 | +0.1052 [+0.0897, +0.1218] | +0.0135 / +0.0113 |
| 452 | +0.0485 | +0.5894 [-0.4136, +1.8910] | +0.0168 / +0.0221 |
| 4222 | +0.1381 | +0.0452 [-0.0523, +0.1495] | +0.0179 / +0.0064 |

[All additional-sample results](README.md), [individual tokens](individual_tokens.md),
[whole-head control contrasts](control_comparisons.md), and
[block-bootstrap components](block_bootstrap.md) retain the full evidence.
The [pre-run protocol](../provenance/sae_replication_protocol.md) records sample selection.
