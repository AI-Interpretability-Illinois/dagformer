# Topology analysis of the trained FourWay models (2026-10-03)

`scripts/topology_analysis.py` (jobs 22636727/22636750; `*_analysis_v1.json`, `*_topology.npz`), 128 windows
of 1024 tokens of each model's own held-out cache for the statistics, full caches for the evaluations.

## Static substitution (held-out NLL, nats)

| model | set | intact | predictor → per-position mean | predictor → global mean | local corr → mean | both → mean |
|---|---|---|---|---|---|---|
| 75M both (locality arm) | wikitext2 | 6.164 | 6.161 | 6.162 | 6.582 | 6.595 |
| 300M corrected | wikitext2 | 3.544 | 3.547 | 3.604 | 4.251 | 4.297 |
| 300M corrected | mathinstruct | 2.565 | 2.556 | 2.600 | 3.065 | 3.065 |
| 1B corrected, 5B tok | wikitext2 | 3.307 | 3.310 | 3.364 | 3.996 | 4.027 |
| 1B corrected, 10B tok | wikitext2 | 3.219 | 3.221 | 3.281 | 3.916 | 3.950 |
| 1B corrected, 10B tok | mathinstruct | 2.321 | 2.322 | 2.366 | 2.849 | 2.868 |

- The shared predictor's per-token output can be replaced by its per-position mean at a cost of at most
  0.005 nats at every size: what it contributes is a token-independent, position-dependent topology. A
  single global mean (no position dependence) costs 0.04-0.06 nats, so the position embedding of the
  predictor carries a little information.
- The local correction channel is different: replacing its per-token output by its mean costs 0.4-0.7 nats,
  more than the whole gap over dense (0.23-0.26 at 1B). In the shipped (corrected) models the per-token
  routing is load-bearing and lives entirely in the local routers; the two channels specialize.
- Read with the locality ablation and its patching test (experiments/coherence/README.md): a predictor-only
  model reaches the same loss with an almost static routing (its per-token part is worth 0.03-0.08 nats),
  so per-token routing is not needed to obtain the gain, but a model trained with local routers comes to
  rely on them.

## What the predictor learned (mean |alpha| mass on the previous layer, per stream)

| model | q | k | v | r |
|---|---|---|---|---|
| 300M corrected (11 routed layers) | 0.19-0.43 | 0.12-0.41 | 0.02-0.18 | 0.04-0.68 |
| 1B corrected, 10B tok (15 routed layers) | 0.17-0.61 | 0.15-0.45 | 0.03-0.24 | 0.06-0.66 |

Most of the mixing mass sits on skip sources, least of all for values (v reads 80-98% from earlier layers);
2-5 sources per head carry more than 5% of the head's largest weight. Per-token spread relative to the mean:
0.05-0.19 for the predictor, 0.5-1.4 for the local corrections.

## Inference-time edge pruning (job 22636936, `*_analysis.json`)

Hyperconnection entries (sources other than the previous layer) ranked by their mean |alpha| (predictor +
correction) over 128 windows; the smallest are zeroed at inference, in both channels, with the intact per-token
routing otherwise unchanged. WikiText-2 NLL:

| model | intact | keep 75% | keep 50% | keep 25% | keep 10% | previous layer only |
|---|---|---|---|---|---|---|
| 75M both (locality arm) | 6.164 | 6.260 | 6.436 | 6.673 | 6.834 | 7.355 |
| 300M corrected | 3.544 | 3.812 | 5.346 | 6.395 | 7.276 | 9.288 |
| 1B corrected, 5B tok | 3.307 | 3.461 | 5.370 | 7.015 | 7.965 | 9.024 |
| 1B corrected, 10B tok | 3.219 | 3.438 | 5.056 | 6.889 | 7.952 | 10.135 |

The learned graph is used densely: removing even the quarter of hyperconnection entries with the smallest mean
weight costs 0.15-0.27 nats (as much as the whole gap over dense), half of them costs ~1.8 nats, and removing
all skips (so each layer reads only its predecessor, as in the dense model) destroys the model. Small-mean
entries are not unused: the per-token corrections modulate them. Sparsifying the topology for inference
would need retraining, not post-hoc masking.


## Measured cost (job 22638949, `throughput.json`; one A100-40GB, bf16, 1024-token sequences, median of 15)

| model | train tok/s (bs 2) | ratio | prefill tok/s (bs 8) | ratio | peak GB |
|---|---|---|---|---|---|
| 300M dense | 36.1K | | 157.5K | | 6.0 |
| 300M DAGFormer | 20.2K | 1.79x | 98.7K | 1.60x | 7.4 |
| 300M DAGFormer, static predictor | 21.5K | 1.68x | 101.4K | 1.55x | 7.4 |
| 1B dense | 14.0K | | 52.1K | | 13.2 |
| 1B DAGFormer | 6.4K | 2.21x | 26.2K | 1.99x | 17.6 |
| 1B DAGFormer, static predictor | 6.5K | 2.15x | 26.5K | 1.97x | 17.6 |

Analytic FLOPs ratios are 1.65x (300M) and 2.13x (1B); the predictor itself costs 1-3%, the source projections
and mixing the rest.
