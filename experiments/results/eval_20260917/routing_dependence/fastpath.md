# Predictor MHA fastpath numerical check

The April stash disabled the predictor encoder fastpath. These runs compare
enabled and disabled settings on the same loaded weights and the first 32
WikiText test windows. All use PyTorch 2.10.0+cu128. Actual fused encoder calls
are counted, rather than assuming the setting changes the execution path.

| Scale | Fused calls enabled / disabled | Disable − enable mean NLL | Paired 95% interval | Maximum relative routing L2 difference | Maximum logit difference |
|---|---:|---:|---|---:|---:|
| 75m | 64 / 0 | +0.0001669 | [-0.0001149, +0.0004486] | 3.53e-05 | 0.296875 |
| 150m | 64 / 0 | -0.0001315 | [-0.0004542, +0.0001912] | 2.08e-05 | 0.273438 |
| 300m | 64 / 0 | +0.0000760 | [-0.0001637, +0.0003158] | 9.65e-06 | 0.273438 |

Disabling the path produces small nonzero routing differences and larger
maximum BF16 logit differences. The mean NLL change has an interval spanning
zero at each scale. This is a numerical check of this evaluation environment,
not a training or throughput test; model defaults remain unchanged.
