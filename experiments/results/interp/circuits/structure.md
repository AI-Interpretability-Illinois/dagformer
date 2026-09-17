# Structure of the routing difference

## domain_code / corr

eligible 3234  chosen 829  (`delta_all`)

### Compression ladder

Cosine with the real Delta over eligible coordinates. `params` is the size of the description; the cheapest row that stays high is the description length of the behavioural routing difference.

| rung | compression | params | cosine | expl.var |
|---|---|---|---|---|
| 1 edge-sparse | top-16 | 16 | 0.806 | 0.649 |
| 1 edge-sparse | top-64 | 64 | 0.894 | 0.799 |
| 1 edge-sparse | top-256 | 256 | 0.952 | 0.906 |
| 1 edge-sparse | top-1024 | 1024 | 0.991 | 0.981 |
| 2 group-sparse | top-2 heads | 42 | 0.511 | 0.261 |
| 2 group-sparse | top-8 heads | 192 | 0.779 | 0.607 |
| 2 group-sparse | top-32 heads | 849 | 0.897 | 0.804 |
| 2 group-sparse | top-2 slots | 14 | 0.488 | 0.238 |
| 2 group-sparse | top-8 slots | 64 | 0.750 | 0.562 |
| 2 group-sparse | top-32 slots | 271 | 0.855 | 0.731 |
| 3 low-rank | rank-1 | 551 | 0.944 | 0.892 |
| 3 low-rank | rank-2 | 1102 | 0.962 | 0.925 |
| 3 low-rank | rank-4 | 2204 | 0.987 | 0.974 |
| 3 low-rank | rank-8 | 4408 | 0.999 | 0.998 |
| 4 structured | mean per head | 187 | 0.236 | 0.056 |
| 4 structured | mean per slot | 539 | 0.440 | 0.194 |
| 4 structured | mean per layer | 11 | 0.090 | 0.008 |
| 4 structured | mean per stream | 4 | 0.055 | 0.003 |
| 4 structured | mean per srcdist | 11 | 0.065 | 0.004 |
| 4 structured | mean per stream_srcdist | 44 | 0.147 | 0.022 |
| 4 structured | mean per stream_layer | 44 | 0.131 | 0.017 |
| 4 structured | mean per stream_layer_srcdist | 264 | 0.299 | 0.089 |
| 5 unstructured | sign x mean\|Delta\| | 1 | 0.397 | 0.157 |

Rank-1 on shuffled Delta reaches cosine 0.567; anything at or below that is what a structureless vector gives for free.

### Concentration vs a shuffled Delta

| grouping | groups | top-8 mass | null top-8 (p95) | groups for 50% | null |
|---|---|---|---|---|---|
| head | 187 | 0.183 | 0.147 (0.162) | 38 | 45 |
| slot | 539 | 0.114 | 0.109 (0.115) | 96 | 108 |
| layer | 11 | 0.931 | 0.910 (0.922) | 3 | 4 |
| stream | 4 | 1.000 | 1.000 (1.000) | 2 | 2 |
| srcdist | 11 | 0.843 | 0.905 (0.925) | 5 | 4 |
| stream_srcdist | 44 | 0.408 | 0.421 (0.452) | 11 | 10 |
| stream_layer | 44 | 0.527 | 0.422 (0.446) | 8 | 10 |
| stream_layer_srcdist | 264 | 0.281 | 0.127 (0.133) | 34 | 65 |

### Connectivity of the selected set

- 829 edges over 176 (layer, head) nodes
- with a selected predecessor: 0.63 (random subset of the same size: 0.68)
- with a selected successor: 0.70 (random: 0.68)
- longest chain: 6 (random: 6.0)

Connectivity at or near the random level means the set is a collection of independent nudges, not a path.

## domain_code / pred

eligible 3234  chosen 513  (`delta_all`)

### Compression ladder

Cosine with the real Delta over eligible coordinates. `params` is the size of the description; the cheapest row that stays high is the description length of the behavioural routing difference.

| rung | compression | params | cosine | expl.var |
|---|---|---|---|---|
| 1 edge-sparse | top-16 | 16 | 0.592 | 0.351 |
| 1 edge-sparse | top-64 | 64 | 0.758 | 0.575 |
| 1 edge-sparse | top-256 | 256 | 0.880 | 0.775 |
| 1 edge-sparse | top-1024 | 1024 | 0.974 | 0.948 |
| 2 group-sparse | top-2 heads | 66 | 0.219 | 0.048 |
| 2 group-sparse | top-8 heads | 240 | 0.486 | 0.236 |
| 2 group-sparse | top-32 heads | 930 | 0.680 | 0.463 |
| 2 group-sparse | top-2 slots | 20 | 0.248 | 0.062 |
| 2 group-sparse | top-8 slots | 76 | 0.477 | 0.227 |
| 2 group-sparse | top-32 slots | 309 | 0.645 | 0.416 |
| 3 low-rank | rank-1 | 551 | 0.859 | 0.737 |
| 3 low-rank | rank-2 | 1102 | 0.924 | 0.853 |
| 3 low-rank | rank-4 | 2204 | 0.978 | 0.957 |
| 3 low-rank | rank-8 | 4408 | 0.998 | 0.996 |
| 4 structured | mean per head | 187 | 0.353 | 0.125 |
| 4 structured | mean per slot | 539 | 0.591 | 0.349 |
| 4 structured | mean per layer | 11 | 0.131 | 0.017 |
| 4 structured | mean per stream | 4 | 0.062 | 0.004 |
| 4 structured | mean per srcdist | 11 | 0.122 | 0.015 |
| 4 structured | mean per stream_srcdist | 44 | 0.200 | 0.040 |
| 4 structured | mean per stream_layer | 44 | 0.196 | 0.038 |
| 4 structured | mean per stream_layer_srcdist | 264 | 0.345 | 0.119 |
| 5 unstructured | sign x mean\|Delta\| | 1 | 0.556 | 0.309 |

Rank-1 on shuffled Delta reaches cosine 0.525; anything at or below that is what a structureless vector gives for free.

### Concentration vs a shuffled Delta

| grouping | groups | top-8 mass | null top-8 (p95) | groups for 50% | null |
|---|---|---|---|---|---|
| head | 187 | 0.125 | 0.112 (0.123) | 52 | 50 |
| slot | 539 | 0.074 | 0.064 (0.069) | 126 | 127 |
| layer | 11 | 0.859 | 0.910 (0.923) | 4 | 4 |
| stream | 4 | 1.000 | 1.000 (1.000) | 2 | 2 |
| srcdist | 11 | 0.864 | 0.909 (0.922) | 5 | 4 |
| stream_srcdist | 44 | 0.391 | 0.409 (0.427) | 11 | 11 |
| stream_layer | 44 | 0.440 | 0.410 (0.427) | 10 | 11 |
| stream_layer_srcdist | 264 | 0.199 | 0.085 (0.093) | 48 | 74 |

### Connectivity of the selected set

- 513 edges over 160 (layer, head) nodes
- with a selected predecessor: 0.67 (random subset of the same size: 0.68)
- with a selected successor: 0.72 (random: 0.68)
- longest chain: 6 (random: 6.0)

Connectivity at or near the random level means the set is a collection of independent nudges, not a path.

## honesty / corr

eligible 3234  chosen 13  (`delta_all`)

### Compression ladder

Cosine with the real Delta over eligible coordinates. `params` is the size of the description; the cheapest row that stays high is the description length of the behavioural routing difference.

| rung | compression | params | cosine | expl.var |
|---|---|---|---|---|
| 1 edge-sparse | top-16 | 16 | 0.662 | 0.438 |
| 1 edge-sparse | top-64 | 64 | 0.803 | 0.645 |
| 1 edge-sparse | top-256 | 256 | 0.913 | 0.834 |
| 1 edge-sparse | top-1024 | 1024 | 0.983 | 0.966 |
| 2 group-sparse | top-2 heads | 60 | 0.413 | 0.170 |
| 2 group-sparse | top-8 heads | 246 | 0.540 | 0.292 |
| 2 group-sparse | top-32 heads | 933 | 0.750 | 0.562 |
| 2 group-sparse | top-2 slots | 20 | 0.415 | 0.172 |
| 2 group-sparse | top-8 slots | 72 | 0.575 | 0.331 |
| 2 group-sparse | top-32 slots | 303 | 0.712 | 0.506 |
| 3 low-rank | rank-1 | 551 | 0.901 | 0.813 |
| 3 low-rank | rank-2 | 1102 | 0.943 | 0.889 |
| 3 low-rank | rank-4 | 2204 | 0.986 | 0.972 |
| 3 low-rank | rank-8 | 4408 | 0.999 | 0.998 |
| 4 structured | mean per head | 187 | 0.301 | 0.091 |
| 4 structured | mean per slot | 539 | 0.472 | 0.223 |
| 4 structured | mean per layer | 11 | 0.105 | 0.011 |
| 4 structured | mean per stream | 4 | 0.023 | 0.001 |
| 4 structured | mean per srcdist | 11 | 0.087 | 0.008 |
| 4 structured | mean per stream_srcdist | 44 | 0.137 | 0.019 |
| 4 structured | mean per stream_layer | 44 | 0.155 | 0.024 |
| 4 structured | mean per stream_layer_srcdist | 264 | 0.318 | 0.101 |
| 5 unstructured | sign x mean\|Delta\| | 1 | 0.497 | 0.247 |

Rank-1 on shuffled Delta reaches cosine 0.506; anything at or below that is what a structureless vector gives for free.

### Concentration vs a shuffled Delta

| grouping | groups | top-8 mass | null top-8 (p95) | groups for 50% | null |
|---|---|---|---|---|---|
| head | 187 | 0.137 | 0.122 (0.131) | 43 | 49 |
| slot | 539 | 0.080 | 0.078 (0.084) | 109 | 119 |
| layer | 11 | 0.921 | 0.909 (0.920) | 4 | 4 |
| stream | 4 | 1.000 | 1.000 (1.000) | 2 | 2 |
| srcdist | 11 | 0.820 | 0.909 (0.919) | 5 | 4 |
| stream_srcdist | 44 | 0.365 | 0.414 (0.432) | 12 | 11 |
| stream_layer | 44 | 0.482 | 0.413 (0.436) | 9 | 11 |
| stream_layer_srcdist | 264 | 0.240 | 0.098 (0.107) | 39 | 71 |

### Connectivity of the selected set

- 13 edges over 12 (layer, head) nodes
- with a selected predecessor: 0.31 (random subset of the same size: 0.33)
- with a selected successor: 0.31 (random: 0.32)
- longest chain: 2 (random: 2.6)

Connectivity at or near the random level means the set is a collection of independent nudges, not a path.

## honesty_fewshot / corr

eligible 3234  chosen 0  (`delta_all`)

### Compression ladder

Cosine with the real Delta over eligible coordinates. `params` is the size of the description; the cheapest row that stays high is the description length of the behavioural routing difference.

| rung | compression | params | cosine | expl.var |
|---|---|---|---|---|
| 1 edge-sparse | top-16 | 16 | 0.845 | 0.714 |
| 1 edge-sparse | top-64 | 64 | 0.911 | 0.830 |
| 1 edge-sparse | top-256 | 256 | 0.956 | 0.913 |
| 1 edge-sparse | top-1024 | 1024 | 0.991 | 0.982 |
| 2 group-sparse | top-2 heads | 54 | 0.673 | 0.454 |
| 2 group-sparse | top-8 heads | 222 | 0.799 | 0.638 |
| 2 group-sparse | top-32 heads | 906 | 0.930 | 0.865 |
| 2 group-sparse | top-2 slots | 18 | 0.664 | 0.441 |
| 2 group-sparse | top-8 slots | 72 | 0.788 | 0.621 |
| 2 group-sparse | top-32 slots | 292 | 0.895 | 0.802 |
| 3 low-rank | rank-1 | 551 | 0.944 | 0.891 |
| 3 low-rank | rank-2 | 1102 | 0.966 | 0.934 |
| 3 low-rank | rank-4 | 2204 | 0.991 | 0.982 |
| 3 low-rank | rank-8 | 4408 | 0.999 | 0.998 |
| 4 structured | mean per head | 187 | 0.265 | 0.070 |
| 4 structured | mean per slot | 539 | 0.443 | 0.196 |
| 4 structured | mean per layer | 11 | 0.106 | 0.011 |
| 4 structured | mean per stream | 4 | 0.071 | 0.005 |
| 4 structured | mean per srcdist | 11 | 0.100 | 0.010 |
| 4 structured | mean per stream_srcdist | 44 | 0.167 | 0.028 |
| 4 structured | mean per stream_layer | 44 | 0.138 | 0.019 |
| 4 structured | mean per stream_layer_srcdist | 264 | 0.280 | 0.078 |
| 5 unstructured | sign x mean\|Delta\| | 1 | 0.374 | 0.140 |

Rank-1 on shuffled Delta reaches cosine 0.598; anything at or below that is what a structureless vector gives for free.

### Concentration vs a shuffled Delta

| grouping | groups | top-8 mass | null top-8 (p95) | groups for 50% | null |
|---|---|---|---|---|---|
| head | 187 | 0.193 | 0.150 (0.165) | 35 | 46 |
| slot | 539 | 0.130 | 0.109 (0.115) | 90 | 110 |
| layer | 11 | 0.932 | 0.908 (0.926) | 3 | 4 |
| stream | 4 | 1.000 | 1.000 (1.000) | 2 | 2 |
| srcdist | 11 | 0.850 | 0.907 (0.926) | 4 | 4 |
| stream_srcdist | 44 | 0.413 | 0.427 (0.452) | 11 | 10 |
| stream_layer | 44 | 0.553 | 0.426 (0.453) | 7 | 10 |
| stream_layer_srcdist | 264 | 0.272 | 0.128 (0.136) | 36 | 66 |

## honesty_fewshot / pred

eligible 3234  chosen 0  (`delta_all`)

### Compression ladder

Cosine with the real Delta over eligible coordinates. `params` is the size of the description; the cheapest row that stays high is the description length of the behavioural routing difference.

| rung | compression | params | cosine | expl.var |
|---|---|---|---|---|
| 1 edge-sparse | top-16 | 16 | 0.611 | 0.373 |
| 1 edge-sparse | top-64 | 64 | 0.746 | 0.556 |
| 1 edge-sparse | top-256 | 256 | 0.866 | 0.750 |
| 1 edge-sparse | top-1024 | 1024 | 0.970 | 0.941 |
| 2 group-sparse | top-2 heads | 66 | 0.160 | 0.026 |
| 2 group-sparse | top-8 heads | 252 | 0.406 | 0.164 |
| 2 group-sparse | top-32 heads | 870 | 0.729 | 0.531 |
| 2 group-sparse | top-2 slots | 16 | 0.303 | 0.092 |
| 2 group-sparse | top-8 slots | 69 | 0.448 | 0.200 |
| 2 group-sparse | top-32 slots | 280 | 0.664 | 0.440 |
| 3 low-rank | rank-1 | 551 | 0.821 | 0.674 |
| 3 low-rank | rank-2 | 1102 | 0.902 | 0.814 |
| 3 low-rank | rank-4 | 2204 | 0.974 | 0.948 |
| 3 low-rank | rank-8 | 4408 | 0.997 | 0.994 |
| 4 structured | mean per head | 187 | 0.471 | 0.222 |
| 4 structured | mean per slot | 539 | 0.674 | 0.455 |
| 4 structured | mean per layer | 11 | 0.173 | 0.030 |
| 4 structured | mean per stream | 4 | 0.069 | 0.005 |
| 4 structured | mean per srcdist | 11 | 0.103 | 0.011 |
| 4 structured | mean per stream_srcdist | 44 | 0.177 | 0.031 |
| 4 structured | mean per stream_layer | 44 | 0.209 | 0.044 |
| 4 structured | mean per stream_layer_srcdist | 264 | 0.344 | 0.119 |
| 5 unstructured | sign x mean\|Delta\| | 1 | 0.570 | 0.325 |

Rank-1 on shuffled Delta reaches cosine 0.571; anything at or below that is what a structureless vector gives for free.

### Concentration vs a shuffled Delta

| grouping | groups | top-8 mass | null top-8 (p95) | groups for 50% | null |
|---|---|---|---|---|---|
| head | 187 | 0.119 | 0.112 (0.122) | 50 | 51 |
| slot | 539 | 0.063 | 0.065 (0.070) | 125 | 129 |
| layer | 11 | 0.861 | 0.909 (0.920) | 4 | 4 |
| stream | 4 | 1.000 | 1.000 (1.000) | 2 | 2 |
| srcdist | 11 | 0.864 | 0.910 (0.920) | 4 | 4 |
| stream_srcdist | 44 | 0.401 | 0.408 (0.425) | 11 | 11 |
| stream_layer | 44 | 0.419 | 0.407 (0.422) | 11 | 11 |
| stream_layer_srcdist | 264 | 0.164 | 0.085 (0.092) | 55 | 76 |

## honesty / pred

eligible 3234  chosen 2  (`delta_all`)

### Compression ladder

Cosine with the real Delta over eligible coordinates. `params` is the size of the description; the cheapest row that stays high is the description length of the behavioural routing difference.

| rung | compression | params | cosine | expl.var |
|---|---|---|---|---|
| 1 edge-sparse | top-16 | 16 | 0.546 | 0.298 |
| 1 edge-sparse | top-64 | 64 | 0.712 | 0.507 |
| 1 edge-sparse | top-256 | 256 | 0.851 | 0.725 |
| 1 edge-sparse | top-1024 | 1024 | 0.969 | 0.938 |
| 2 group-sparse | top-2 heads | 66 | 0.194 | 0.038 |
| 2 group-sparse | top-8 heads | 219 | 0.470 | 0.221 |
| 2 group-sparse | top-32 heads | 882 | 0.681 | 0.464 |
| 2 group-sparse | top-2 slots | 22 | 0.136 | 0.018 |
| 2 group-sparse | top-8 slots | 83 | 0.332 | 0.111 |
| 2 group-sparse | top-32 slots | 282 | 0.603 | 0.364 |
| 3 low-rank | rank-1 | 551 | 0.821 | 0.675 |
| 3 low-rank | rank-2 | 1102 | 0.890 | 0.793 |
| 3 low-rank | rank-4 | 2204 | 0.971 | 0.942 |
| 3 low-rank | rank-8 | 4408 | 0.997 | 0.993 |
| 4 structured | mean per head | 187 | 0.400 | 0.160 |
| 4 structured | mean per slot | 539 | 0.656 | 0.430 |
| 4 structured | mean per layer | 11 | 0.122 | 0.015 |
| 4 structured | mean per stream | 4 | 0.089 | 0.008 |
| 4 structured | mean per srcdist | 11 | 0.123 | 0.015 |
| 4 structured | mean per stream_srcdist | 44 | 0.210 | 0.044 |
| 4 structured | mean per stream_layer | 44 | 0.227 | 0.052 |
| 4 structured | mean per stream_layer_srcdist | 264 | 0.426 | 0.181 |
| 5 unstructured | sign x mean\|Delta\| | 1 | 0.591 | 0.350 |

Rank-1 on shuffled Delta reaches cosine 0.465; anything at or below that is what a structureless vector gives for free.

### Concentration vs a shuffled Delta

| grouping | groups | top-8 mass | null top-8 (p95) | groups for 50% | null |
|---|---|---|---|---|---|
| head | 187 | 0.117 | 0.109 (0.119) | 50 | 51 |
| slot | 539 | 0.059 | 0.058 (0.065) | 134 | 131 |
| layer | 11 | 0.857 | 0.910 (0.920) | 4 | 4 |
| stream | 4 | 1.000 | 1.000 (1.000) | 2 | 2 |
| srcdist | 11 | 0.851 | 0.909 (0.920) | 4 | 4 |
| stream_srcdist | 44 | 0.383 | 0.410 (0.429) | 12 | 11 |
| stream_layer | 44 | 0.417 | 0.406 (0.426) | 11 | 11 |
| stream_layer_srcdist | 264 | 0.168 | 0.078 (0.083) | 52 | 76 |

### Connectivity of the selected set

- 2 edges over 2 (layer, head) nodes
- with a selected predecessor: 0.00 (random subset of the same size: 0.02)
- with a selected successor: 0.00 (random: 0.02)
- longest chain: 1 (random: 1.0)

Connectivity at or near the random level means the set is a collection of independent nudges, not a path.
