# Context fidelity — context_validation_sequential

Fixed 3-edge circuit; 1024 new content items; 50 natural-text sequences.
Effects are paired against the unchanged checkpoint. p_true is normalized over the candidate values, not the entire vocabulary.

| Channel | Gamma | Neutral Δp | Deceptive Δp | ΔNLL (nats) | NLL rise > 0.05 |
|---|---:|---:|---:|---:|---|
| pred | 0.75 | -0.0083 | -0.0111 | +0.00390 | no |
| pred | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| pred | 1.25 | +0.0068 | +0.0088 | +0.00147 | no |
| pred | 1.5 | +0.0122 | +0.0155 | +0.00771 | no |
| corr | 0.75 | -0.0097 | -0.0135 | +0.00175 | no |
| corr | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| corr | 1.25 | +0.0090 | +0.0115 | +0.00163 | no |
| corr | 1.5 | +0.0168 | +0.0206 | +0.00632 | no |
| both | 0.75 | -0.0193 | -0.0268 | +0.01020 | no |
| both | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| both | 1.25 | +0.0147 | +0.0180 | +0.00594 | no |
| both | 1.5 | +0.0260 | +0.0308 | +0.02266 | no |

Controls: random heads; same layer/stream/source counts; L2 norm matched per layer/token on incoming activations.
Paired intervals and all control arms are in the adjacent summary JSON. They describe these content combinations, not unseen training runs.
