# Transfer of the eight previously selected SAE directions

Additional 64 WikiText test windows from 14 documents, excluding all 31 earlier test documents. The eight historical directions, five whole-head controls and R/Q/K/V splits are fixed before this run. This is an additional article sample within WikiText-2.

These are the fixed directions used in the historical large-generation check.
No direction or token set was selected on the new WikiText test windows.
Edits act on the correction channel at all positions; alpha units match the
original mean active feature coefficient. NLL changes are token-weighted.
Intervals resample the 64 windows in paired form, with 10,000 draws; they
are unadjusted and do not cover training-seed or feature-selection uncertainty.

The count column gives target tokens / windows containing target tokens.
Bootstrap draws with no target tokens are omitted; valid-draw counts are in JSON.

Token labels are raw tokenizer pieces, including whitespace and fragments.
They are not a semantic annotation of what each feature represents.

Negative target-token ΔNLL means that those gold tokens become more probable.
The signed dose span is ΔNLL(+4) − ΔNLL(−4), calculated from paired windows.

| Feature | Target tokens | Tokens / windows | ΔNLL at −4 [95% CI] | ΔNLL at +4 [95% CI] | Dose span [95% CI] |
|---|---|---:|---|---|---|
| 7326 | o, 's, uth | 36 / 18 | +0.0205 [-0.0480, +0.0720] | +0.0088 [-0.0150, +0.0331] | -0.0117 [-0.0825, +0.0775] |
| 1986 | 201, 199, 188, 189, 190, 186 | 104 / 22 | +0.0758 [+0.0187, +0.1558] | -0.0329 [-0.1153, +0.0286] | -0.1086 [-0.2671, +0.0074] |
| 3583 | Ġbrother, Ġson, Ġwife, Ġuncle, Ġtwo, Ġfather | 117 / 52 | +0.0370 [-0.0016, +0.0759] | -0.0174 [-0.0464, +0.0103] | -0.0545 [-0.1177, +0.0088] |
| 3560 | Ġ", ĠI, "ĊĊ | 422 / 45 | -0.0666 [-0.0886, -0.0453] | +0.0938 [+0.0724, +0.1141] | +0.1604 [+0.1186, +0.2009] |
| 7019 | ", _, _", _,, ,", ." | 0 / 0 | no target tokens | no target tokens | no target tokens |
| 7068 | Ġ, Ġ$, Ġsix, Ġfive | 2276 / 64 | -0.0429 [-0.0506, -0.0358] | +0.0622 [+0.0538, +0.0715] | +0.1052 [+0.0897, +0.1218] |
| 452 | ew, end, ynthia, il, osen | 5 / 5 | -0.2550 [-0.8383, +0.1989] | +0.3344 [-0.2179, +1.0526] | +0.5894 [-0.4136, +1.8910] |
| 4222 | Ġcan, Ġmust, Ġmay, 'll, âĢĻll | 35 / 20 | -0.0210 [-0.0716, +0.0264] | +0.0242 [-0.0302, +0.0818] | +0.0452 [-0.0523, +0.1495] |

## Original screen, other tokens and permuted controls

5 whole-head permutations within each layer; the same permutation acts on Q/K/V, source positions are preserved, and shared R remains unchanged; each preserves direction norm and per-stream/source head means.
They preserve coefficient values and norms. The five-control span range
is descriptive and is not a confidence interval. Cross-corpus differences
also change which target tokens occur and their contexts.

| Feature | Original target ΔNLL −4 / +4 | New other-token ΔNLL −4 / +4 | Five permuted target-span range |
|---|---|---|---|
| 7326 | +0.1382 / -0.0714 | +0.0357 / +0.0277 | [-0.0626, +0.0103] |
| 1986 | +0.1082 / -0.0513 | +0.0169 / +0.0127 | [-0.0684, -0.0192] |
| 3583 | -0.0454 / +0.0932 | +0.0343 / +0.0169 | [-0.0129, +0.0706] |
| 3560 | -0.0459 / +0.0630 | +0.0150 / +0.0119 | [-0.0388, +0.2159] |
| 7019 | -0.0418 / +0.0776 | +0.0289 / +0.0350 | no target tokens |
| 7068 | -0.0428 / +0.0621 | +0.0135 / +0.0113 | [+0.0004, +0.0462] |
| 452 | +0.0751 / -0.0272 | +0.0168 / +0.0221 | [-0.1431, +0.7685] |
| 4222 | -0.0327 / +0.0475 | +0.0179 / +0.0064 | [-0.0419, +0.0567] |

## Language-model cost of the controls

The permutations preserve direction norms, not language-model capability.
Their other-token NLL costs must be considered alongside target effects.
A feature/control contrast at unequal damage does not isolate semantic
specificity at a fixed capability cost. Ranges below contain the five
measured controls, not confidence intervals.

| Feature | Control other-token ΔNLL range at −4 | Range at +4 |
|---|---|---|
| 7326 | [+0.0391, +0.0565] | [+0.0249, +0.0460] |
| 1986 | [+0.0144, +0.0209] | [+0.0093, +0.0134] |
| 3583 | [+0.0331, +0.0729] | [+0.0099, +0.0255] |
| 3560 | [+0.0066, +0.0148] | [+0.0043, +0.0206] |
| 7019 | [+0.0161, +0.0343] | [+0.0131, +0.0333] |
| 7068 | [+0.0115, +0.0225] | [+0.0033, +0.0103] |
| 452 | [+0.0081, +0.0155] | [+0.0134, +0.0232] |
| 4222 | [+0.0093, +0.0150] | [+0.0036, +0.0090] |

## Shared residual and Q/K/V components

The same feature direction is restricted to shared R or Q/K/V coordinates.
The two component effects need not add because the network is nonlinear.

| Feature | Component | Target ΔNLL −4 | Target ΔNLL +4 | Other-token ΔNLL −4 / +4 |
|---|---|---|---|---|
| 7326 | r_only | +0.0209 [-0.0291, +0.0615] | -0.0175 [-0.0413, +0.0006] | +0.0146 / +0.0159 |
| 7326 | qkv_only | -0.0104 [-0.0292, +0.0029] | +0.0212 [+0.0086, +0.0376] | +0.0171 / +0.0116 |
| 1986 | r_only | +0.0368 [+0.0066, +0.0761] | -0.0233 [-0.0653, +0.0095] | +0.0110 / +0.0058 |
| 1986 | qkv_only | +0.0274 [+0.0003, +0.0671] | -0.0219 [-0.0705, +0.0116] | +0.0063 / +0.0070 |
| 3583 | r_only | +0.0111 [-0.0121, +0.0336] | -0.0080 [-0.0285, +0.0124] | +0.0194 / -0.0001 |
| 3583 | qkv_only | +0.0311 [-0.0020, +0.0660] | -0.0102 [-0.0422, +0.0188] | +0.0177 / +0.0203 |
| 3560 | r_only | -0.0809 [-0.0944, -0.0672] | +0.0875 [+0.0740, +0.1010] | +0.0045 / +0.0049 |
| 3560 | qkv_only | +0.0146 [-0.0022, +0.0300] | +0.0066 [-0.0055, +0.0198] | +0.0091 / +0.0062 |
| 7019 | r_only | no target tokens | no target tokens | +0.0054 / +0.0035 |
| 7019 | qkv_only | no target tokens | no target tokens | +0.0212 / +0.0288 |
| 7068 | r_only | -0.0158 [-0.0202, -0.0113] | +0.0238 [+0.0188, +0.0288] | +0.0121 / +0.0047 |
| 7068 | qkv_only | -0.0282 [-0.0337, -0.0233] | +0.0367 [+0.0313, +0.0428] | +0.0013 / +0.0067 |
| 452 | r_only | -0.1251 [-0.3523, +0.0268] | +0.1305 [-0.0217, +0.3331] | -0.0005 / +0.0112 |
| 452 | qkv_only | -0.1304 [-0.4894, +0.1601] | +0.2268 [-0.1861, +0.7575] | +0.0197 / +0.0124 |
| 4222 | r_only | +0.0095 [-0.0158, +0.0354] | -0.0175 [-0.0425, +0.0083] | +0.0076 / -0.0004 |
| 4222 | qkv_only | -0.0286 [-0.0674, +0.0108] | +0.0390 [-0.0007, +0.0789] | +0.0101 / +0.0065 |

[Direct paired span contrasts](control_comparisons.md) compare the feature
with each control while retaining the pairing of all four dose arms.

[Individual-token diagnostics](individual_tokens.md) retain occurrence counts
and paired intervals within each fixed target set.
