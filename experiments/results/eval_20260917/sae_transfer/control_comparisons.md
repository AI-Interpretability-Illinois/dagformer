# Paired feature-minus-control dose spans

Each contrast is (feature +4 minus feature -4) minus
(control +4 minus control -4), paired across the same windows.
A positive value means a larger signed dose response, not higher
language-model accuracy. Intervals are unadjusted window bootstrap
intervals; control NLL costs are reported in the main table.

| Feature | Control | Signed span difference [95% CI] |
|---|---|---|
| 7326 | random0 | +3.0171 [+2.0670, +4.1517] |
| 7326 | random1 | +1.2924 [+0.4622, +2.2617] |
| 7326 | random2 | +3.0818 [+1.6715, +4.2184] |
| 7326 | random3 | +2.6577 [+0.9540, +4.1488] |
| 7326 | random4 | +0.1220 [-0.3849, +0.7009] |
| 1986 | random0 | +0.1193 [+0.0160, +0.2248] |
| 1986 | random1 | -0.4805 [-0.5846, -0.3841] |
| 1986 | random2 | +0.0958 [+0.0477, +0.1399] |
| 1986 | random3 | -0.2485 [-0.3593, -0.1395] |
| 1986 | random4 | -0.0134 [-0.1325, +0.1098] |
| 3583 | random0 | +0.6258 [+0.4829, +0.7594] |
| 3583 | random1 | +0.6177 [+0.4972, +0.7450] |
| 3583 | random2 | +0.5819 [+0.4450, +0.7277] |
| 3583 | random3 | -0.0171 [-0.1213, +0.0835] |
| 3583 | random4 | +0.0466 [-0.0608, +0.1473] |
| 3560 | random0 | -0.2411 [-0.3055, -0.1766] |
| 3560 | random1 | +0.0775 [+0.0025, +0.1480] |
| 3560 | random2 | +0.1240 [+0.0636, +0.1774] |
| 3560 | random3 | +0.1169 [+0.0301, +0.1887] |
| 3560 | random4 | -0.2214 [-0.3089, -0.1232] |
| 7019 | random0 | no target tokens |
| 7019 | random1 | no target tokens |
| 7019 | random2 | no target tokens |
| 7019 | random3 | no target tokens |
| 7019 | random4 | no target tokens |
| 7068 | random0 | +0.1634 [+0.1359, +0.1926] |
| 7068 | random1 | -0.0552 [-0.0895, -0.0194] |
| 7068 | random2 | -0.0390 [-0.0676, -0.0092] |
| 7068 | random3 | -0.1639 [-0.2022, -0.1265] |
| 7068 | random4 | +0.2154 [+0.1868, +0.2457] |
| 452 | random0 | -6.9885 [-8.6179, -5.5654] |
| 452 | random1 | +1.9375 [+1.0522, +3.1284] |
| 452 | random2 | +0.9481 [+0.3007, +1.8454] |
| 452 | random3 | +1.7876 [+0.9080, +3.0386] |
| 452 | random4 | -0.5113 [-1.2308, +0.3364] |
| 4222 | random0 | -0.0343 [-0.1639, +0.1122] |
| 4222 | random1 | +0.1189 [-0.0274, +0.2637] |
| 4222 | random2 | +0.0403 [-0.1054, +0.1897] |
| 4222 | random3 | +0.2011 [+0.0733, +0.3418] |
| 4222 | random4 | +0.3127 [+0.1692, +0.4488] |
