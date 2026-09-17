# Context fidelity — context_validation_step10500

Fixed 10-edge circuit; 1024 new content items; 50 natural-text sequences.
Effects are paired against the unchanged checkpoint. p_true is normalized over the candidate values, not the entire vocabulary.

| Channel | Gamma | Neutral Δp | Deceptive Δp | ΔNLL (nats) | NLL rise > 0.05 |
|---|---:|---:|---:|---:|---|
| pred | 0.75 | -0.0097 | -0.0144 | +0.00427 | no |
| pred | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| pred | 1.25 | +0.0079 | +0.0118 | +0.00365 | no |
| pred | 1.5 | +0.0148 | +0.0208 | +0.01463 | no |
| corr | 0.75 | -0.0331 | -0.0396 | +0.01105 | no |
| corr | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| corr | 1.25 | +0.0291 | +0.0299 | +0.00856 | no |
| corr | 1.5 | +0.0530 | +0.0512 | +0.03357 | no |
| both | 0.75 | -0.0465 | -0.0594 | +0.02876 | no |
| both | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| both | 1.25 | +0.0349 | +0.0379 | +0.02037 | no |
| both | 1.5 | +0.0614 | +0.0620 | +0.07819 | yes |

Controls: random heads; same layer/stream/source counts; L2 norm matched per layer/token on incoming activations.
Paired intervals and all control arms are in the adjacent summary JSON. They describe these content combinations, not unseen training runs.
