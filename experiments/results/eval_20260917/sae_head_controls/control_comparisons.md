# Paired feature-minus-control dose spans

Each contrast is (feature +4 minus feature -4) minus
(control +4 minus control -4), paired across the same windows.
A positive value means a larger signed dose response, not higher
language-model accuracy. Intervals are unadjusted window bootstrap
intervals; control NLL costs are reported in the main table.

| Feature | Control | Signed span difference [95% CI] |
|---|---|---|
| 7326 | random0 | -0.0234 [-0.1168, +0.0847] |
| 7326 | random1 | -0.0423 [-0.2372, +0.1771] |
| 7326 | random2 | +0.0412 [-0.1306, +0.2403] |
| 7326 | random3 | +0.0479 [-0.0542, +0.1855] |
| 7326 | random4 | +0.0136 [-0.1559, +0.2478] |
| 1986 | random0 | +0.0209 [-0.0123, +0.0512] |
| 1986 | random1 | +0.0253 [+0.0021, +0.0490] |
| 1986 | random2 | +0.0448 [+0.0115, +0.0771] |
| 1986 | random3 | +0.0082 [-0.0233, +0.0386] |
| 1986 | random4 | -0.0062 [-0.0367, +0.0241] |
| 3583 | random0 | -0.1038 [-0.1601, -0.0509] |
| 3583 | random1 | -0.0440 [-0.0934, +0.0048] |
| 3583 | random2 | -0.1162 [-0.1832, -0.0497] |
| 3583 | random3 | -0.0332 [-0.1050, +0.0343] |
| 3583 | random4 | -0.0848 [-0.1777, +0.0062] |
| 3560 | random0 | -0.0238 [-0.0657, +0.0182] |
| 3560 | random1 | +0.0836 [+0.0481, +0.1199] |
| 3560 | random2 | -0.0144 [-0.0479, +0.0205] |
| 3560 | random3 | +0.1553 [+0.1312, +0.1808] |
| 3560 | random4 | +0.0268 [-0.0001, +0.0574] |
| 7019 | random0 | no target tokens |
| 7019 | random1 | no target tokens |
| 7019 | random2 | no target tokens |
| 7019 | random3 | no target tokens |
| 7019 | random4 | no target tokens |
| 7068 | random0 | +0.1173 [+0.1017, +0.1339] |
| 7068 | random1 | +0.0811 [+0.0693, +0.0933] |
| 7068 | random2 | +0.1262 [+0.1108, +0.1426] |
| 7068 | random3 | +0.0885 [+0.0780, +0.0999] |
| 7068 | random4 | +0.0697 [+0.0548, +0.0855] |
| 452 | random0 | +0.0301 [-0.1479, +0.1882] |
| 452 | random1 | +0.0844 [-0.0946, +0.2814] |
| 452 | random2 | +0.0685 [-0.1203, +0.2733] |
| 452 | random3 | +0.0436 [-0.2558, +0.3672] |
| 452 | random4 | +0.0757 [-0.1018, +0.2390] |
| 4222 | random0 | +0.0822 [-0.0044, +0.1688] |
| 4222 | random1 | +0.1544 [+0.0879, +0.2234] |
| 4222 | random2 | +0.1545 [+0.0894, +0.2188] |
| 4222 | random3 | +0.0823 [+0.0303, +0.1346] |
| 4222 | random4 | +0.1054 [+0.0318, +0.1807] |
