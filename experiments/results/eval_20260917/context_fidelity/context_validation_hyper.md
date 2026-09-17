# Context fidelity — context_validation_hyper

Fixed 7-edge circuit; 1024 new content items; 50 natural-text sequences.
Effects are paired against the unchanged checkpoint. p_true is normalized over the candidate values, not the entire vocabulary.

| Channel | Gamma | Neutral Δp | Deceptive Δp | ΔNLL (nats) | NLL rise > 0.05 |
|---|---:|---:|---:|---:|---|
| pred | 0.75 | -0.0023 | -0.0048 | +0.00060 | no |
| pred | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| pred | 1.25 | +0.0017 | +0.0041 | +0.00027 | no |
| pred | 1.5 | +0.0032 | +0.0074 | +0.00160 | no |
| corr | 0.75 | -0.0226 | -0.0249 | +0.00576 | no |
| corr | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| corr | 1.25 | +0.0194 | +0.0195 | +0.00285 | no |
| corr | 1.5 | +0.0362 | +0.0347 | +0.01232 | no |
| both | 0.75 | -0.0260 | -0.0312 | +0.00965 | no |
| both | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| both | 1.25 | +0.0207 | +0.0226 | +0.00501 | no |
| both | 1.5 | +0.0377 | +0.0387 | +0.01981 | no |

Controls: random heads; same layer/stream/source counts; L2 norm matched per layer/token on incoming activations.
Paired intervals and all control arms are in the adjacent summary JSON. They describe these content combinations, not unseen training runs.
