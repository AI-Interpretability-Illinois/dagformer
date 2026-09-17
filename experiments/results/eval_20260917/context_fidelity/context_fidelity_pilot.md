# Context fidelity — context_fidelity_pilot

Fixed 10-edge circuit; 128 new content items; 8 natural-text sequences.
Effects are paired against the unchanged checkpoint. p_true is normalized over the candidate values, not the entire vocabulary.

| Channel | Gamma | Neutral Δp | Deceptive Δp | ΔNLL (nats) | NLL rise > 0.05 |
|---|---:|---:|---:|---:|---|
| pred | 0 | -0.0535 | -0.0753 | +0.04066 | no |
| pred | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| pred | 4 | +0.0883 | +0.0937 | +0.35442 | yes |
| corr | 0 | -0.1332 | -0.1719 | +0.08655 | yes |
| corr | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| corr | 4 | +0.1570 | +0.1277 | +0.43356 | yes |
| both | 0 | -0.1668 | -0.2086 | +0.16205 | yes |
| both | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| both | 4 | +0.2456 | +0.2132 | +0.97973 | yes |

Controls: random heads; same layer/stream/source counts; not norm-matched.
Paired intervals and all control arms are in the adjacent summary JSON. They describe these content combinations, not unseen training runs.
