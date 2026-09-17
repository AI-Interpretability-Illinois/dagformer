# Named edits under prompt-cluster uncertainty

All differences are percentage points from the unedited model. The primary
endpoint remains target-first. Target-without-alternative and alternative-anywhere
are secondary literal-mention diagnostics; they do not grade semantic truth.

| Source | Condition | Arm | Target first | Alternative first | No candidate | Target without alternative | Alternative anywhere |
|---|---|---|---|---|---|---|---|
| context_generation | original/neutral | pred/circuit | +1.46 [+0.68, +2.34] | +0.00 [+0.00, +0.00] | -1.46 [-2.34, -0.68] | +1.46 [+0.68, +2.34] | +0.00 [+0.00, +0.00] |
| context_generation | original/neutral | corr/circuit | +1.56 [+0.59, +2.54] | +0.00 [+0.00, +0.00] | -1.56 [-2.54, -0.59] | +1.56 [+0.59, +2.54] | +0.00 [+0.00, +0.00] |
| context_generation | original/neutral | both/circuit | +2.15 [+1.17, +3.12] | +0.00 [+0.00, +0.00] | -2.15 [-3.12, -1.17] | +2.15 [+1.17, +3.12] | +0.00 [+0.00, +0.00] |
| context_generation | original/deceptive | pred/circuit | +2.44 [+1.56, +3.42] | +0.00 [+0.00, +0.00] | -2.44 [-3.42, -1.56] | +2.44 [+1.56, +3.42] | +0.00 [+0.00, +0.00] |
| context_generation | original/deceptive | corr/circuit | +1.76 [+0.98, +2.64] | +0.00 [+0.00, +0.00] | -1.76 [-2.64, -0.98] | +1.76 [+0.98, +2.64] | +0.00 [+0.00, +0.00] |
| context_generation | original/deceptive | both/circuit | +3.22 [+2.25, +4.30] | +0.00 [+0.00, +0.00] | -3.22 [-4.30, -2.25] | +3.22 [+2.25, +4.30] | +0.00 [+0.00, +0.00] |
| context_generation | qa/neutral | pred/circuit | +2.25 [+1.35, +3.24] | +0.00 [+0.00, +0.00] | -2.25 [-3.24, -1.35] | +2.25 [+1.35, +3.24] | +0.00 [+0.00, +0.00] |
| context_generation | qa/neutral | corr/circuit | +1.56 [+0.87, +2.38] | +0.00 [+0.00, +0.00] | -1.56 [-2.38, -0.87] | +1.56 [+0.87, +2.38] | +0.00 [+0.00, +0.00] |
| context_generation | qa/neutral | both/circuit | +2.15 [+1.26, +3.11] | +0.00 [+0.00, +0.00] | -2.15 [-3.11, -1.26] | +2.15 [+1.26, +3.11] | +0.00 [+0.00, +0.00] |
| context_generation | qa/deceptive | pred/circuit | +0.20 [-0.39, +0.78] | +0.00 [+0.00, +0.00] | -0.20 [-0.78, +0.39] | +0.20 [-0.39, +0.78] | +0.00 [+0.00, +0.00] |
| context_generation | qa/deceptive | corr/circuit | +0.68 [+0.20, +1.27] | +0.00 [+0.00, +0.00] | -0.68 [-1.27, -0.20] | +0.68 [+0.20, +1.27] | +0.00 [+0.00, +0.00] |
| context_generation | qa/deceptive | both/circuit | +0.78 [+0.29, +1.37] | +0.00 [+0.00, +0.00] | -0.78 [-1.37, -0.29] | +0.78 [+0.29, +1.37] | +0.00 [+0.00, +0.00] |
| context_generation | dialogue/neutral | pred/circuit | +1.95 [+1.17, +2.83] | +0.00 [+0.00, +0.00] | -1.95 [-2.83, -1.17] | +1.95 [+1.17, +2.83] | +0.00 [+0.00, +0.00] |
| context_generation | dialogue/neutral | corr/circuit | +1.56 [+0.88, +2.34] | +0.00 [+0.00, +0.00] | -1.56 [-2.34, -0.88] | +1.56 [+0.88, +2.34] | +0.00 [+0.00, +0.00] |
| context_generation | dialogue/neutral | both/circuit | +2.05 [+1.27, +2.93] | +0.00 [+0.00, +0.00] | -2.05 [-2.93, -1.27] | +2.05 [+1.27, +2.93] | +0.00 [+0.00, +0.00] |
| context_generation | dialogue/deceptive | pred/circuit | +2.34 [+1.46, +3.32] | +0.00 [+0.00, +0.00] | -2.34 [-3.32, -1.46] | +2.34 [+1.46, +3.32] | +0.00 [+0.00, +0.00] |
| context_generation | dialogue/deceptive | corr/circuit | +2.25 [+1.37, +3.22] | +0.00 [+0.00, +0.00] | -2.25 [-3.22, -1.37] | +2.25 [+1.37, +3.22] | +0.00 [+0.00, +0.00] |
| context_generation | dialogue/deceptive | both/circuit | +2.73 [+1.76, +3.81] | +0.00 [+0.00, +0.00] | -2.73 [-3.81, -1.76] | +2.73 [+1.76, +3.81] | +0.00 [+0.00, +0.00] |
| context_attribute_generation | attribute_qa/neutral | pred/circuit | -6.64 [-8.75, -4.64] | +0.29 [-0.48, +1.05] | +6.35 [+4.49, +8.34] | -6.64 [-8.75, -4.64] | +0.29 [-0.48, +1.05] |
| context_attribute_generation | attribute_qa/neutral | corr/circuit | -0.78 [-2.35, +0.69] | -1.17 [-1.97, -0.48] | +1.95 [+0.67, +3.36] | -0.78 [-2.35, +0.69] | -1.17 [-1.97, -0.48] |
| context_attribute_generation | attribute_qa/neutral | both/circuit | -5.27 [-7.39, -3.24] | -0.88 [-1.76, -0.10] | +6.15 [+4.27, +8.15] | -5.27 [-7.39, -3.24] | -0.88 [-1.76, -0.10] |
| context_attribute_generation | attribute_qa/deceptive | pred/circuit | -5.86 [-7.91, -3.91] | +0.78 [-0.49, +2.15] | +5.08 [+3.32, +6.84] | +0.00 [-2.73, +2.64] | -5.08 [-7.42, -2.83] |
| context_attribute_generation | attribute_qa/deceptive | corr/circuit | +0.29 [-1.37, +1.86] | -2.44 [-3.71, -1.27] | +2.15 [+0.88, +3.42] | +6.45 [+4.30, +8.69] | -8.59 [-10.55, -6.64] |
| context_attribute_generation | attribute_qa/deceptive | both/circuit | -3.03 [-5.08, -1.07] | -2.15 [-3.42, -0.88] | +5.18 [+3.42, +6.93] | +4.69 [+1.95, +7.42] | -9.86 [-12.11, -7.62] |
