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

### Against a behaviour-free Delta

The same ladder on 20 balanced-split draws: identical items, paraphrases and averaging depth, with the pos-vs-neg term cancelled. Only an excess over these is about the behaviour.

| compression | real | null mean | null p95 | excess |
|---|---|---|---|---|
| top16 | 0.806 | 0.695 | 0.871 | -0.066 |
| top64 | 0.894 | 0.814 | 0.923 | -0.029 |
| top256 | 0.952 | 0.913 | 0.961 | -0.009 |
| top1024 | 0.991 | 0.982 | 0.992 | -0.001 |
| head2 | 0.511 | 0.399 | 0.559 | -0.048 |
| head8 | 0.779 | 0.611 | 0.824 | -0.044 |
| head32 | 0.897 | 0.812 | 0.940 | -0.044 |
| rank1 | 0.944 | 0.891 | 0.957 | -0.013 |
| rank2 | 0.962 | 0.933 | 0.971 | -0.009 |
| sign_only | 0.397 | 0.481 | 0.604 | -0.207 |

**No compression beats the behaviour-free null.** Whatever structure the ladder above reports is a property of routing differences in general, not of this behaviour.

### Concentration vs a shuffled Delta

| grouping | groups | top-8 mass | null top-8 (p95) | groups for 50% | null |
|---|---|---|---|---|---|
| head | 187 | 0.183 | 0.147 (0.161) | 38 | 45 |
| slot | 539 | 0.114 | 0.107 (0.113) | 96 | 108 |
| layer | 11 | 0.931 | 0.911 (0.926) | 3 | 4 |
| stream | 4 | 1.000 | 1.000 (1.000) | 2 | 2 |
| srcdist | 11 | 0.843 | 0.908 (0.923) | 5 | 4 |
| stream_srcdist | 44 | 0.408 | 0.418 (0.445) | 11 | 11 |
| stream_layer | 44 | 0.527 | 0.420 (0.447) | 8 | 10 |
| stream_layer_srcdist | 264 | 0.281 | 0.127 (0.134) | 34 | 65 |

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

Rank-1 on shuffled Delta reaches cosine 0.531; anything at or below that is what a structureless vector gives for free.

### Against a behaviour-free Delta

The same ladder on 20 balanced-split draws: identical items, paraphrases and averaging depth, with the pos-vs-neg term cancelled. Only an excess over these is about the behaviour.

| compression | real | null mean | null p95 | excess |
|---|---|---|---|---|
| top16 | 0.592 | 0.539 | 0.655 | -0.063 |
| top64 | 0.758 | 0.701 | 0.797 | -0.039 |
| top256 | 0.880 | 0.851 | 0.909 | -0.029 |
| top1024 | 0.974 | 0.967 | 0.981 | -0.007 |
| head2 | 0.219 | 0.273 | 0.329 | -0.111 |
| head8 | 0.486 | 0.433 | 0.561 | -0.075 |
| head32 | 0.680 | 0.673 | 0.748 | -0.068 |
| rank1 | 0.859 | 0.804 | 0.883 | -0.025 |
| rank2 | 0.924 | 0.891 | 0.933 | -0.009 |
| sign_only | 0.556 | 0.591 | 0.684 | -0.128 |

**No compression beats the behaviour-free null.** Whatever structure the ladder above reports is a property of routing differences in general, not of this behaviour.

### Concentration vs a shuffled Delta

| grouping | groups | top-8 mass | null top-8 (p95) | groups for 50% | null |
|---|---|---|---|---|---|
| head | 187 | 0.125 | 0.114 (0.123) | 52 | 50 |
| slot | 539 | 0.074 | 0.063 (0.068) | 126 | 128 |
| layer | 11 | 0.859 | 0.909 (0.919) | 4 | 4 |
| stream | 4 | 1.000 | 1.000 (1.000) | 2 | 2 |
| srcdist | 11 | 0.864 | 0.911 (0.923) | 5 | 4 |
| stream_srcdist | 44 | 0.391 | 0.409 (0.430) | 11 | 11 |
| stream_layer | 44 | 0.440 | 0.410 (0.425) | 10 | 11 |
| stream_layer_srcdist | 264 | 0.199 | 0.084 (0.090) | 48 | 75 |

### Connectivity of the selected set

- 513 edges over 160 (layer, head) nodes
- with a selected predecessor: 0.67 (random subset of the same size: 0.68)
- with a selected successor: 0.72 (random: 0.68)
- longest chain: 6 (random: 6.0)

Connectivity at or near the random level means the set is a collection of independent nudges, not a path.
