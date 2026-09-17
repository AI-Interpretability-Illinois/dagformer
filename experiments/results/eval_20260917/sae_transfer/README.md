# Transfer of the eight previously selected SAE directions

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

Controls permute each direction within every layer and Q/K/V/R stream.
They preserve coefficient values and norms. The five-control span range
is descriptive and is not a confidence interval. Cross-corpus differences
also change which target tokens occur and their contexts.

| Feature | Original target ΔNLL −4 / +4 | New other-token ΔNLL −4 / +4 | Five permuted target-span range |
|---|---|---|---|
| 7326 | +0.1382 / -0.0714 | +0.0353 / +0.0286 | [-3.0215, -0.0617] |
| 1986 | +0.1082 / -0.0513 | +0.0172 / +0.0134 | [-0.1263, +0.4736] |
| 3583 | -0.0454 / +0.0932 | +0.0326 / +0.0194 | [-0.6571, -0.0142] |
| 3560 | -0.0459 / +0.0630 | +0.0136 / +0.0132 | [+0.0176, +0.3826] |
| 7019 | -0.0418 / +0.0776 | +0.0261 / +0.0352 | no target tokens |
| 7068 | -0.0428 / +0.0621 | +0.0133 / +0.0116 | [-0.0860, +0.2933] |
| 452 | +0.0751 / -0.0272 | +0.0168 / +0.0222 | [-1.8890, +7.0370] |
| 4222 | -0.0327 / +0.0475 | +0.0170 / +0.0069 | [-0.1746, +0.1725] |

## Language-model cost of the controls

The permutations preserve direction norms, not language-model capability.
Their other-token NLL costs must be considered alongside target effects.
A feature/control contrast at unequal damage does not isolate semantic
specificity at a fixed capability cost. Ranges below contain the five
measured controls, not confidence intervals.

| Feature | Control other-token ΔNLL range at −4 | Range at +4 |
|---|---|---|
| 7326 | [+0.5090, +6.6855] | [+1.1300, +6.9322] |
| 1986 | [+0.1850, +0.3285] | [+0.0983, +0.5054] |
| 3583 | [+0.0712, +0.7320] | [+0.1364, +0.3424] |
| 3560 | [+0.0641, +0.0915] | [+0.0656, +0.0956] |
| 7019 | [+0.1513, +3.0892] | [+0.2110, +0.7933] |
| 7068 | [+0.0765, +0.2064] | [+0.0804, +0.8223] |
| 452 | [+0.4359, +1.2744] | [+0.2425, +4.3568] |
| 4222 | [+0.0480, +0.1095] | [+0.0469, +0.1175] |
