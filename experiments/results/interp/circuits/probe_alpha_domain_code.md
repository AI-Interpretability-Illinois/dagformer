# What does alpha_pred encode?

Corpus: 8160 positions, 3234 routing coordinates.  Behaviour: `domain_code`.

Variance of alpha_pred left unexplained by (870 token types over 7152 positions):

- absolute position: **0.0474**
- current token id: **0.8295**
- current token id, after the per-position mean is removed: **0.1192**

The last figure is the one that answers whether alpha_pred responds to context at all: near 0 means the predicted topology is a lookup table on (position, token) with nothing else in it.

| probe | metric | score | shuffled-label null | trivial baseline |
|---|---|---|---|---|
| position | r2 | 0.979 | -0.036 (max -0.010) | 0.000 |
| log_freq_rank | r2 | 0.592 | -0.050 (max -0.014) | 0.000 |
| token | accuracy | 0.998 | 0.038 (max 0.048) | 0.092 |
| prev_token | accuracy | 0.275 | 0.039 (max 0.042) | 0.083 |
| polarity | accuracy | 1.000 | 0.495 (max 0.525) | 0.500 |
