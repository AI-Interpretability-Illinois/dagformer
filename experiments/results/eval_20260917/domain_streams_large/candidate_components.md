# Candidate likelihood components

These are changes from the neutral reference. A larger code-minus-prose
score can arise from either increasing code likelihood or decreasing prose
likelihood. Code-candidate preference is a two-candidate scoring endpoint.

| Channel / stream | Dose | Content set | Code logp change | Prose logp change | Code preference change (points) |
|---|---:|---|---:|---:|---:|
| pred/all | 8 | original_test | +0.0131 | +0.0202 | +0.00 |
| pred/all | 8 | new_content | +0.0077 | +0.0191 | +0.00 |
| pred/all | 16 | original_test | +0.0248 | +0.0380 | +0.00 |
| pred/all | 16 | new_content | +0.0165 | +0.0402 | +0.00 |
| pred/all | 32 | original_test | +0.0491 | +0.0713 | +0.00 |
| pred/all | 32 | new_content | +0.0330 | +0.0784 | +0.00 |
| pred/all | 64 | original_test | +0.0950 | +0.1264 | +0.00 |
| pred/all | 64 | new_content | +0.0630 | +0.1413 | +0.00 |
| pred/q | 8 | original_test | -0.0029 | +0.0029 | +0.00 |
| pred/q | 8 | new_content | -0.0018 | +0.0005 | +0.00 |
| pred/q | 16 | original_test | -0.0033 | +0.0054 | +0.00 |
| pred/q | 16 | new_content | -0.0080 | +0.0024 | +0.00 |
| pred/q | 32 | original_test | -0.0040 | +0.0106 | +0.00 |
| pred/q | 32 | new_content | -0.0144 | +0.0064 | +0.00 |
| pred/q | 64 | original_test | -0.0063 | +0.0185 | +0.00 |
| pred/q | 64 | new_content | -0.0271 | +0.0093 | +0.00 |
| pred/k | 8 | original_test | +0.0047 | -0.0007 | +0.00 |
| pred/k | 8 | new_content | +0.0043 | +0.0002 | +0.00 |
| pred/k | 16 | original_test | +0.0064 | -0.0013 | +0.00 |
| pred/k | 16 | new_content | +0.0081 | +0.0010 | +0.00 |
| pred/k | 32 | original_test | +0.0079 | -0.0037 | +0.00 |
| pred/k | 32 | new_content | +0.0155 | +0.0002 | +0.00 |
| pred/k | 64 | original_test | +0.0196 | -0.0175 | +0.00 |
| pred/k | 64 | new_content | +0.0310 | -0.0071 | +0.00 |
| pred/v | 8 | original_test | +0.0073 | +0.0152 | +0.00 |
| pred/v | 8 | new_content | +0.0069 | +0.0153 | +0.00 |
| pred/v | 16 | original_test | +0.0177 | +0.0291 | +0.00 |
| pred/v | 16 | new_content | +0.0119 | +0.0328 | +0.00 |
| pred/v | 32 | original_test | +0.0319 | +0.0612 | +0.00 |
| pred/v | 32 | new_content | +0.0241 | +0.0658 | +0.00 |
| pred/v | 64 | original_test | +0.0626 | +0.1126 | +0.00 |
| pred/v | 64 | new_content | +0.0456 | +0.1292 | +0.00 |
| pred/r | 8 | original_test | +0.0050 | +0.0053 | +0.00 |
| pred/r | 8 | new_content | +0.0011 | +0.0011 | +0.00 |
| pred/r | 16 | original_test | +0.0078 | +0.0053 | +0.00 |
| pred/r | 16 | new_content | +0.0040 | +0.0020 | +0.00 |
| pred/r | 32 | original_test | +0.0139 | +0.0075 | +0.00 |
| pred/r | 32 | new_content | +0.0077 | +0.0056 | +0.00 |
| pred/r | 64 | original_test | +0.0252 | +0.0157 | +0.00 |
| pred/r | 64 | new_content | +0.0174 | +0.0129 | +0.00 |
| corr/all | 8 | original_test | -1.7620 | -1.8320 | +0.00 |
| corr/all | 8 | new_content | -1.7433 | -2.0204 | +0.00 |
| corr/all | 16 | original_test | -4.2871 | -9.9534 | +75.00 |
| corr/all | 16 | new_content | -3.7644 | -10.1320 | +75.00 |
| corr/all | 32 | original_test | -4.1737 | -11.1261 | +100.00 |
| corr/all | 32 | new_content | -3.7220 | -11.4972 | +96.88 |
| corr/all | 64 | original_test | -3.5274 | -8.9868 | +62.50 |
| corr/all | 64 | new_content | -3.1211 | -9.2444 | +73.44 |
| corr/q | 8 | original_test | +0.1466 | -0.5064 | +0.00 |
| corr/q | 8 | new_content | +0.1747 | -0.5487 | +0.00 |
| corr/q | 16 | original_test | +0.1368 | -1.3496 | +12.50 |
| corr/q | 16 | new_content | +0.2168 | -1.3788 | +3.12 |
| corr/q | 32 | original_test | -0.1012 | -1.5807 | +12.50 |
| corr/q | 32 | new_content | -0.0434 | -1.6787 | +3.12 |
| corr/q | 64 | original_test | -0.1523 | -1.3664 | +12.50 |
| corr/q | 64 | new_content | -0.0612 | -1.4583 | +3.12 |
| corr/k | 8 | original_test | -0.1951 | -0.5391 | +0.00 |
| corr/k | 8 | new_content | -0.2259 | -0.5674 | +0.00 |
| corr/k | 16 | original_test | -0.4198 | -1.5222 | +0.00 |
| corr/k | 16 | new_content | -0.4360 | -1.5391 | +0.00 |
| corr/k | 32 | original_test | -0.6626 | -1.9941 | +12.50 |
| corr/k | 32 | new_content | -0.5841 | -2.2039 | +3.12 |
| corr/k | 64 | original_test | -0.8238 | -2.1697 | +12.50 |
| corr/k | 64 | new_content | -0.7236 | -2.4735 | +3.12 |
| corr/v | 8 | original_test | +0.0871 | -0.1864 | +0.00 |
| corr/v | 8 | new_content | +0.1770 | -0.3062 | +0.00 |
| corr/v | 16 | original_test | -0.0678 | -0.6215 | +0.00 |
| corr/v | 16 | new_content | +0.0845 | -0.9314 | +0.00 |
| corr/v | 32 | original_test | -0.4010 | -1.4937 | +0.00 |
| corr/v | 32 | new_content | -0.2790 | -1.8308 | +0.00 |
| corr/v | 64 | original_test | -1.0037 | -2.5399 | +0.00 |
| corr/v | 64 | new_content | -0.8042 | -2.7998 | +0.00 |
| corr/r | 8 | original_test | -1.9052 | -1.8358 | +0.00 |
| corr/r | 8 | new_content | -1.9806 | -2.1883 | +0.00 |
| corr/r | 16 | original_test | -4.4070 | -9.6164 | +56.25 |
| corr/r | 16 | new_content | -3.9047 | -9.8924 | +68.75 |
| corr/r | 32 | original_test | -4.4449 | -12.1290 | +100.00 |
| corr/r | 32 | new_content | -3.9728 | -12.5502 | +96.88 |
| corr/r | 64 | original_test | -4.1681 | -12.3054 | +100.00 |
| corr/r | 64 | new_content | -3.7241 | -12.6826 | +100.00 |
