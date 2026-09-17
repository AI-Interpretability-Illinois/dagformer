# Context fidelity — context_validation_original

Fixed 10-edge circuit; 1024 new content items; 50 natural-text sequences.
Effects are paired against the unchanged checkpoint. p_true is normalized over the candidate values, not the entire vocabulary.

| Channel | Gamma | Neutral Δp | Deceptive Δp | ΔNLL (nats) | NLL rise > 0.05 |
|---|---:|---:|---:|---:|---|
| pred | 0.75 | -0.0106 | -0.0157 | +0.00559 | no |
| pred | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| pred | 1.25 | +0.0085 | +0.0127 | +0.00270 | no |
| pred | 1.5 | +0.0150 | +0.0223 | +0.01268 | no |
| corr | 0.75 | -0.0326 | -0.0399 | +0.01186 | no |
| corr | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| corr | 1.25 | +0.0280 | +0.0298 | +0.00733 | no |
| corr | 1.5 | +0.0512 | +0.0516 | +0.02999 | no |
| both | 0.75 | -0.0465 | -0.0606 | +0.03075 | no |
| both | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| both | 1.25 | +0.0342 | +0.0384 | +0.01806 | no |
| both | 1.5 | +0.0601 | +0.0633 | +0.07179 | yes |

Controls: random heads; same layer/stream/source counts; L2 norm matched per layer/token on incoming activations.
Paired intervals and all control arms are in the adjacent summary JSON. They describe these content combinations, not unseen training runs.
