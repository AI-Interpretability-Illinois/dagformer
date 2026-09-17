# Paired feature-minus-control dose spans

Each contrast is (feature +4 minus feature -4) minus
(control +4 minus control -4), paired across the same windows.
A positive value means a larger signed dose response, not higher
language-model accuracy. Intervals are unadjusted window bootstrap
intervals; control NLL costs are reported in the main table.

| Feature | Control | Signed span difference [95% CI] |
|---|---|---|
| 7326 | random0 | -0.0220 [-0.1073, +0.0464] |
| 7326 | random1 | +0.0017 [-0.0633, +0.0655] |
| 7326 | random2 | +0.0509 [-0.0232, +0.1162] |
| 7326 | random3 | +0.0179 [-0.0100, +0.0493] |
| 7326 | random4 | +0.0269 [-0.0324, +0.1022] |
| 1986 | random0 | -0.0557 [-0.1293, -0.0062] |
| 1986 | random1 | -0.0402 [-0.0912, -0.0039] |
| 1986 | random2 | -0.0527 [-0.1484, +0.0170] |
| 1986 | random3 | -0.0850 [-0.1787, -0.0180] |
| 1986 | random4 | -0.0894 [-0.1797, -0.0241] |
| 3583 | random0 | -0.0680 [-0.1331, -0.0027] |
| 3583 | random1 | -0.0473 [-0.1277, +0.0268] |
| 3583 | random2 | -0.1250 [-0.2120, -0.0476] |
| 3583 | random3 | -0.0436 [-0.1382, +0.0449] |
| 3583 | random4 | -0.0416 [-0.1649, +0.0681] |
| 3560 | random0 | -0.0555 [-0.0888, -0.0236] |
| 3560 | random1 | +0.0631 [+0.0366, +0.0960] |
| 3560 | random2 | +0.0022 [-0.0301, +0.0352] |
| 3560 | random3 | +0.1992 [+0.1653, +0.2330] |
| 3560 | random4 | +0.0527 [+0.0292, +0.0797] |
| 7019 | random0 | no target tokens |
| 7019 | random1 | no target tokens |
| 7019 | random2 | no target tokens |
| 7019 | random3 | no target tokens |
| 7019 | random4 | no target tokens |
| 7068 | random0 | +0.0974 [+0.0827, +0.1146] |
| 7068 | random1 | +0.0734 [+0.0617, +0.0860] |
| 7068 | random2 | +0.1048 [+0.0916, +0.1198] |
| 7068 | random3 | +0.0781 [+0.0658, +0.0918] |
| 7068 | random4 | +0.0589 [+0.0440, +0.0753] |
| 452 | random0 | +0.1381 [-0.2794, +0.5998] |
| 452 | random1 | +0.3493 [-0.2369, +1.1163] |
| 452 | random2 | +0.1495 [-0.3030, +0.6976] |
| 452 | random3 | +0.7325 [-0.3895, +2.1263] |
| 452 | random4 | -0.1790 [-0.4898, +0.0816] |
| 4222 | random0 | +0.0213 [-0.0522, +0.1046] |
| 4222 | random1 | +0.0871 [+0.0030, +0.1597] |
| 4222 | random2 | +0.0847 [-0.0468, +0.2030] |
| 4222 | random3 | +0.0167 [-0.0964, +0.1265] |
| 4222 | random4 | -0.0115 [-0.0864, +0.0616] |
