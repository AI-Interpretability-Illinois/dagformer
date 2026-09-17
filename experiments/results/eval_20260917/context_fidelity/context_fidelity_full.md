# Context fidelity — context_fidelity_full

Fixed 10-edge circuit; 1024 new content items; 50 natural-text sequences.
Effects are paired against the unchanged checkpoint. p_true is normalized over the candidate values, not the entire vocabulary.

| Channel | Gamma | Neutral Δp | Deceptive Δp | ΔNLL (nats) | NLL rise > 0.05 |
|---|---:|---:|---:|---:|---|
| pred | 0 | -0.0543 | -0.0743 | +0.03783 | no |
| pred | 0.5 | -0.0232 | -0.0329 | +0.00911 | no |
| pred | 0.75 | -0.0106 | -0.0149 | +0.00253 | no |
| pred | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| pred | 1.25 | +0.0082 | +0.0118 | +0.00138 | no |
| pred | 1.5 | +0.0147 | +0.0210 | +0.00613 | no |
| pred | 2 | +0.0260 | +0.0355 | +0.02690 | no |
| pred | 4 | +0.0885 | +0.0915 | +0.33492 | yes |
| corr | 0 | -0.1304 | -0.1683 | +0.09481 | yes |
| corr | 0.5 | -0.0656 | -0.0840 | +0.02842 | no |
| corr | 0.75 | -0.0309 | -0.0374 | +0.00651 | no |
| corr | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| corr | 1.25 | +0.0259 | +0.0276 | +0.00332 | no |
| corr | 1.5 | +0.0474 | +0.0477 | +0.01519 | no |
| corr | 2 | +0.0792 | +0.0737 | +0.06767 | yes |
| corr | 4 | +0.1488 | +0.1201 | +0.48217 | yes |
| both | 0 | -0.1651 | -0.2050 | +0.14135 | yes |
| both | 0.5 | -0.0955 | -0.1239 | +0.06203 | yes |
| both | 0.75 | -0.0447 | -0.0574 | +0.01508 | no |
| both | 1 | +0.0000 | +0.0000 | +0.00000 | no |
| both | 1.25 | +0.0320 | +0.0357 | +0.00849 | no |
| both | 1.5 | +0.0561 | +0.0585 | +0.03707 | no |
| both | 2 | +0.0946 | +0.0881 | +0.16669 | yes |
| both | 4 | +0.2396 | +0.2074 | +1.05066 | yes |

Controls: random heads; same layer/stream/source counts; not norm-matched.
Paired intervals and all control arms are in the adjacent summary JSON. They describe these content combinations, not unseen training runs.
