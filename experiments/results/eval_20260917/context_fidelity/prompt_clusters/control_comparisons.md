# Named-minus-control comparisons with prompt clusters

Intervals resample rendered prompts for each fixed control, not the population
of possible control directions. Differences are percentage points.

| Source | Condition | Channel | Control | Target first | Target without alternative |
|---|---|---|---|---|---|
| context_generation | original/neutral | pred | 0 | +1.46 [+0.68, +2.34] | +1.46 [+0.68, +2.34] |
| context_generation | original/neutral | pred | 1 | +1.56 [+0.78, +2.34] | +1.56 [+0.78, +2.34] |
| context_generation | original/neutral | corr | 0 | +0.88 [+0.00, +1.76] | +0.88 [+0.00, +1.76] |
| context_generation | original/neutral | corr | 1 | +1.66 [+0.78, +2.64] | +1.66 [+0.78, +2.64] |
| context_generation | original/neutral | both | 0 | +1.56 [+0.68, +2.54] | +1.56 [+0.68, +2.54] |
| context_generation | original/neutral | both | 1 | +2.15 [+1.27, +3.12] | +2.15 [+1.27, +3.12] |
| context_generation | original/deceptive | pred | 0 | +2.25 [+1.37, +3.22] | +2.25 [+1.37, +3.22] |
| context_generation | original/deceptive | pred | 1 | +2.34 [+1.46, +3.32] | +2.34 [+1.46, +3.32] |
| context_generation | original/deceptive | corr | 0 | +0.98 [+0.20, +1.86] | +0.98 [+0.20, +1.86] |
| context_generation | original/deceptive | corr | 1 | +1.37 [+0.68, +2.15] | +1.37 [+0.68, +2.15] |
| context_generation | original/deceptive | both | 0 | +1.86 [+0.88, +2.83] | +1.86 [+0.88, +2.83] |
| context_generation | original/deceptive | both | 1 | +3.03 [+2.05, +4.10] | +3.03 [+2.05, +4.10] |
| context_generation | qa/neutral | pred | 0 | +2.64 [+1.57, +3.83] | +2.64 [+1.57, +3.83] |
| context_generation | qa/neutral | pred | 1 | +3.71 [+2.51, +5.01] | +3.71 [+2.51, +5.01] |
| context_generation | qa/neutral | corr | 0 | +2.34 [+1.36, +3.47] | +2.34 [+1.36, +3.47] |
| context_generation | qa/neutral | corr | 1 | +3.71 [+2.47, +5.08] | +3.71 [+2.47, +5.08] |
| context_generation | qa/neutral | both | 0 | +2.93 [+1.81, +4.14] | +2.93 [+1.81, +4.14] |
| context_generation | qa/neutral | both | 1 | +4.88 [+3.41, +6.45] | +4.88 [+3.41, +6.45] |
| context_generation | qa/deceptive | pred | 0 | +0.59 [+0.00, +1.27] | +0.59 [+0.00, +1.27] |
| context_generation | qa/deceptive | pred | 1 | +1.56 [+0.78, +2.44] | +1.56 [+0.78, +2.44] |
| context_generation | qa/deceptive | corr | 0 | +1.37 [+0.68, +2.15] | +1.37 [+0.68, +2.15] |
| context_generation | qa/deceptive | corr | 1 | +2.93 [+1.95, +4.00] | +2.93 [+1.95, +4.00] |
| context_generation | qa/deceptive | both | 0 | +1.46 [+0.78, +2.25] | +1.46 [+0.78, +2.25] |
| context_generation | qa/deceptive | both | 1 | +3.81 [+2.73, +5.08] | +3.81 [+2.73, +5.08] |
| context_generation | dialogue/neutral | pred | 0 | +2.05 [+1.17, +2.93] | +2.05 [+1.17, +2.93] |
| context_generation | dialogue/neutral | pred | 1 | +3.03 [+2.05, +4.10] | +3.03 [+2.05, +4.10] |
| context_generation | dialogue/neutral | corr | 0 | +2.05 [+1.07, +3.12] | +2.05 [+1.07, +3.12] |
| context_generation | dialogue/neutral | corr | 1 | +3.61 [+2.54, +4.79] | +3.61 [+2.54, +4.79] |
| context_generation | dialogue/neutral | both | 0 | +2.25 [+1.37, +3.22] | +2.25 [+1.37, +3.22] |
| context_generation | dialogue/neutral | both | 1 | +4.69 [+3.42, +6.05] | +4.69 [+3.42, +6.05] |
| context_generation | dialogue/deceptive | pred | 0 | +2.44 [+1.56, +3.42] | +2.44 [+1.56, +3.42] |
| context_generation | dialogue/deceptive | pred | 1 | +4.49 [+3.32, +5.86] | +4.49 [+3.32, +5.86] |
| context_generation | dialogue/deceptive | corr | 0 | +3.42 [+2.34, +4.59] | +3.42 [+2.34, +4.59] |
| context_generation | dialogue/deceptive | corr | 1 | +4.88 [+3.61, +6.25] | +4.88 [+3.61, +6.25] |
| context_generation | dialogue/deceptive | both | 0 | +3.61 [+2.54, +4.79] | +3.61 [+2.54, +4.79] |
| context_generation | dialogue/deceptive | both | 1 | +6.05 [+4.69, +7.52] | +6.05 [+4.69, +7.52] |
| context_attribute_generation | attribute_qa/neutral | pred | 0 | -6.54 [-8.71, -4.47] | -6.54 [-8.71, -4.47] |
| context_attribute_generation | attribute_qa/neutral | pred | 1 | -6.25 [-8.20, -4.39] | -6.25 [-8.20, -4.39] |
| context_attribute_generation | attribute_qa/neutral | corr | 0 | -0.98 [-2.67, +0.77] | -0.98 [-2.67, +0.77] |
| context_attribute_generation | attribute_qa/neutral | corr | 1 | -1.17 [-2.80, +0.39] | -1.17 [-2.80, +0.39] |
| context_attribute_generation | attribute_qa/neutral | both | 0 | -5.96 [-8.23, -3.71] | -5.96 [-8.23, -3.71] |
| context_attribute_generation | attribute_qa/neutral | both | 1 | -5.57 [-7.67, -3.51] | -5.57 [-7.67, -3.51] |
| context_attribute_generation | attribute_qa/deceptive | pred | 0 | -5.08 [-7.13, -3.12] | +0.78 [-1.86, +3.42] |
| context_attribute_generation | attribute_qa/deceptive | pred | 1 | -4.59 [-6.54, -2.64] | +0.88 [-1.76, +3.42] |
| context_attribute_generation | attribute_qa/deceptive | corr | 0 | +0.39 [-1.37, +2.15] | +5.27 [+2.93, +7.52] |
| context_attribute_generation | attribute_qa/deceptive | corr | 1 | +0.10 [-1.56, +1.76] | +4.88 [+2.73, +7.03] |
| context_attribute_generation | attribute_qa/deceptive | both | 0 | -2.44 [-4.59, -0.39] | +4.39 [+1.66, +7.13] |
| context_attribute_generation | attribute_qa/deceptive | both | 1 | -2.54 [-4.59, -0.49] | +3.32 [+0.68, +5.96] |
