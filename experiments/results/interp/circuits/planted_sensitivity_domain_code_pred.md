# Planted-circuit power analysis (sensitivity)

Baseline score -5.1357 (se 0.3411), NLL 5.5165.

| k | magnitude | induced rotation | \|shift\| | shift/se | dNLL |
|---|---|---|---|---|---|
| 16 | 0.01 | 0.00013 | 0.0003 | 0.0 | -0.001 |
| 16 | 0.05 | 0.00073 | 0.0047 | 0.0 | -0.000 |
| 16 | 0.1 | 0.00156 | 0.0037 | 0.0 | -0.001 |
| 16 | 0.2 | 0.00410 | 0.0065 | 0.0 | -0.001 |
| 16 | 0.5 | 0.00459 | 0.2659 | 0.8 | +0.042 |
| 16 | 1 | 0.00984 | 0.0647 | 0.2 | +0.023 |
| 16 | 5 | 0.05986 | 2.9150 | 8.5 | +3.389 |
| 256 | 0.01 | 0.00181 | 0.0049 | 0.0 | -0.000 |
| 256 | 0.05 | 0.00785 | 0.0561 | 0.2 | -0.001 |
| 256 | 0.1 | 0.01966 | 0.1465 | 0.4 | +0.031 |
| 256 | 0.2 | 0.03669 | 0.0864 | 0.3 | +0.118 |
| 256 | 0.5 | 0.10211 | 0.8532 | 2.5 | +0.921 |
| 256 | 1 | 0.17802 | 0.5429 | 1.6 | +1.818 |
| 256 | 5 | 1.05845 | 3.3466 | 9.8 | +6.410 |
| 3234 | 0.01 | 0.01173 | 0.0047 | 0.0 | +0.002 |
| 3234 | 0.05 | 0.05866 | 0.0546 | 0.2 | +0.059 |
| 3234 | 0.1 | 0.11732 | 0.4488 | 1.3 | +0.382 |
| 3234 | 0.2 | 0.23464 | 0.1122 | 0.3 | +1.181 |
| 3234 | 0.5 | 0.58661 | 2.7958 | 8.2 | +4.699 |
| 3234 | 1 | 1.17322 | 4.1798 | 12.3 | +7.934 |
| 3234 | 5 | 5.86612 | 4.0649 | 11.9 | +10.699 |

Criterion: shift >= 2 se with dNLL <= 0.05 (rows above the gate do not count: they move the score by destroying the model, not by steering it).

**Nothing in the swept range is detectable.** Whatever the real instruction does, this stage could not have seen it.

The planted edits point in random directions, which is the worst case: a genuinely behavioural direction of the same magnitude should do better. So this floor bounds the verification stage from above rather than pinning it.
