# Transfer of the eight previously selected SAE directions

This mechanism follow-up reuses the same 128 WikiText test windows as
the initial transfer run. Whole-head controls and the R/Q/K/V split
were added after inspecting that run; this is not a second corpus replication.

These are the fixed directions used in the historical large-generation check.
No direction or token set was selected on the new WikiText test windows.
Edits act on the correction channel at all positions; alpha units match the
original mean active feature coefficient. NLL changes are token-weighted.
Intervals resample the 128 windows in paired form, with 10,000 draws; they
are unadjusted and do not cover training-seed or feature-selection uncertainty.

The count column gives target tokens / windows containing target tokens.
Bootstrap draws with no target tokens are omitted; valid-draw counts are in JSON.

Token labels are raw tokenizer pieces, including whitespace and fragments.
They are not a semantic annotation of what each feature represents.

Negative target-token ΔNLL means that those gold tokens become more probable.
The signed dose span is ΔNLL(+4) − ΔNLL(−4), calculated from paired windows.

| Feature | Target tokens | Tokens / windows | ΔNLL at −4 [95% CI] | ΔNLL at +4 [95% CI] | Dose span [95% CI] |
|---|---|---:|---|---|---|
| 7326 | o, 's, uth | 55 / 29 | +0.0086 [-0.0718, +0.0810] | +0.0689 [-0.0075, +0.1600] | +0.0603 [-0.0803, +0.2201] |
| 1986 | 201, 199, 188, 189, 190, 186 | 406 / 67 | +0.0315 [+0.0104, +0.0544] | +0.0245 [-0.0022, +0.0515] | -0.0070 [-0.0552, +0.0393] |
| 3583 | Ġbrother, Ġson, Ġwife, Ġuncle, Ġtwo, Ġfather | 246 / 100 | +0.0439 [+0.0168, +0.0722] | +0.0126 [-0.0159, +0.0396] | -0.0313 [-0.0847, +0.0191] |
| 3560 | Ġ", ĠI, "ĊĊ | 1139 / 94 | -0.0582 [-0.0805, -0.0373] | +0.0833 [+0.0659, +0.1012] | +0.1415 [+0.1034, +0.1813] |
| 7019 | ", _, _", _,, ,", ." | 0 / 0 | no target tokens | no target tokens | no target tokens |
| 7068 | Ġ, Ġ$, Ġsix, Ġfive | 4134 / 127 | -0.0521 [-0.0578, -0.0465] | +0.0773 [+0.0701, +0.0848] | +0.1294 [+0.1169, +0.1424] |
| 452 | ew, end, ynthia, il, osen | 36 / 22 | +0.0041 [-0.0789, +0.0817] | +0.0526 [-0.0462, +0.1821] | +0.0485 [-0.1180, +0.2517] |
| 4222 | Ġcan, Ġmust, Ġmay, 'll, âĢĻll | 84 / 38 | -0.0563 [-0.0873, -0.0302] | +0.0818 [+0.0549, +0.1125] | +0.1381 [+0.0866, +0.1983] |

## Original screen, other tokens and permuted controls

5 whole-head permutations within each layer; the same permutation acts on Q/K/V, source positions are preserved, and shared R remains unchanged; each preserves direction norm and per-stream/source head means.
They preserve coefficient values and norms. The five-control span range
is descriptive and is not a confidence interval. Cross-corpus differences
also change which target tokens occur and their contexts.

| Feature | Original target ΔNLL −4 / +4 | New other-token ΔNLL −4 / +4 | Five permuted target-span range |
|---|---|---|---|
| 7326 | +0.1382 / -0.0714 | +0.0353 / +0.0286 | [+0.0124, +0.1026] |
| 1986 | +0.1082 / -0.0513 | +0.0172 / +0.0134 | [-0.0517, -0.0008] |
| 3583 | -0.0454 / +0.0932 | +0.0326 / +0.0194 | [+0.0019, +0.0848] |
| 3560 | -0.0459 / +0.0630 | +0.0136 / +0.0132 | [-0.0138, +0.1653] |
| 7019 | -0.0418 / +0.0776 | +0.0261 / +0.0352 | no target tokens |
| 7068 | -0.0428 / +0.0621 | +0.0133 / +0.0116 | [+0.0031, +0.0597] |
| 452 | +0.0751 / -0.0272 | +0.0168 / +0.0222 | [-0.0359, +0.0184] |
| 4222 | -0.0327 / +0.0475 | +0.0170 / +0.0069 | [-0.0164, +0.0560] |

## Language-model cost of the controls

The permutations preserve direction norms, not language-model capability.
Their other-token NLL costs must be considered alongside target effects.
A feature/control contrast at unequal damage does not isolate semantic
specificity at a fixed capability cost. Ranges below contain the five
measured controls, not confidence intervals.

| Feature | Control other-token ΔNLL range at −4 | Range at +4 |
|---|---|---|
| 7326 | [+0.0361, +0.0546] | [+0.0242, +0.0452] |
| 1986 | [+0.0153, +0.0218] | [+0.0088, +0.0138] |
| 3583 | [+0.0318, +0.0708] | [+0.0115, +0.0255] |
| 3560 | [+0.0043, +0.0129] | [+0.0067, +0.0229] |
| 7019 | [+0.0158, +0.0336] | [+0.0144, +0.0329] |
| 7068 | [+0.0119, +0.0217] | [+0.0048, +0.0103] |
| 452 | [+0.0092, +0.0164] | [+0.0128, +0.0241] |
| 4222 | [+0.0092, +0.0143] | [+0.0044, +0.0089] |

## Shared residual and Q/K/V components

The same feature direction is restricted to shared R or Q/K/V coordinates.
The two component effects need not add because the network is nonlinear.

| Feature | Component | Target ΔNLL −4 | Target ΔNLL +4 | Other-token ΔNLL −4 / +4 |
|---|---|---|---|---|
| 7326 | r_only | -0.0090 [-0.0931, +0.0672] | +0.0448 [-0.0156, +0.1079] | +0.0136 / +0.0158 |
| 7326 | qkv_only | +0.0197 [-0.0326, +0.0613] | +0.0153 [-0.0235, +0.0714] | +0.0174 / +0.0120 |
| 1986 | r_only | +0.0230 [+0.0107, +0.0360] | -0.0001 [-0.0138, +0.0138] | +0.0118 / +0.0058 |
| 1986 | qkv_only | +0.0034 [-0.0103, +0.0178] | +0.0159 [+0.0008, +0.0304] | +0.0056 / +0.0078 |
| 3583 | r_only | +0.0040 [-0.0133, +0.0212] | +0.0166 [+0.0008, +0.0332] | +0.0188 / +0.0010 |
| 3583 | qkv_only | +0.0459 [+0.0221, +0.0716] | +0.0015 [-0.0248, +0.0266] | +0.0164 / +0.0214 |
| 3560 | r_only | -0.0750 [-0.0833, -0.0673] | +0.0805 [+0.0729, +0.0884] | +0.0039 / +0.0059 |
| 3560 | qkv_only | +0.0186 [+0.0005, +0.0359] | +0.0043 [-0.0116, +0.0213] | +0.0085 / +0.0066 |
| 7019 | r_only | no target tokens | no target tokens | +0.0058 / +0.0030 |
| 7019 | qkv_only | no target tokens | no target tokens | +0.0182 / +0.0296 |
| 7068 | r_only | -0.0195 [-0.0223, -0.0168] | +0.0311 [+0.0278, +0.0343] | +0.0119 / +0.0054 |
| 7068 | qkv_only | -0.0335 [-0.0386, -0.0287] | +0.0439 [+0.0381, +0.0501] | +0.0012 / +0.0064 |
| 452 | r_only | +0.0151 [-0.0355, +0.0677] | -0.0082 [-0.0628, +0.0532] | +0.0001 / +0.0107 |
| 452 | qkv_only | +0.0032 [-0.0829, +0.0912] | +0.0703 [-0.0276, +0.1956] | +0.0187 / +0.0126 |
| 4222 | r_only | +0.0066 [-0.0063, +0.0175] | +0.0040 [-0.0079, +0.0170] | +0.0073 / -0.0000 |
| 4222 | qkv_only | -0.0663 [-0.0984, -0.0396] | +0.0799 [+0.0549, +0.1079] | +0.0098 / +0.0068 |

[Direct paired span contrasts](control_comparisons.md) compare the feature
with each control while retaining the pairing of all four dose arms.

[Individual-token diagnostics](individual_tokens.md) retain occurrence counts
and paired intervals within each fixed target set.
