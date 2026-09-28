# Compute axis and the EBT comparison

## Current FourWay compute

The older `experiments/METHODOLOGY_flops_loss_pareto.md` is not an accurate FLOP
specification for the evaluated implementation. In particular, its small-model
hidden dimensions, vocabulary size, and description of the predictor differ from
the actual configs. More consequentially, it omits repeated QKV projection of
earlier layer outputs.

`src/model/olmo_graph.py:FourWayDAGFormer.forward` stacks all earlier layer outputs
and projects them with the current layer's fused QKV weight before mixing. For
layer index l starting at zero, there are l+1 source vectors per token. Treating
a multiply-add as two FLOPs, the QKV projection alone costs:

```
dense:             6 d² L
current FourWay:   6 d² L(L+1)/2
extra:             3 d² L(L−1)
```

| Nominal backbone | Dense QKV MFLOPs/token | Current FourWay QKV MFLOPs/token | Extra |
|---|---:|---:|---:|
| 75M | 9.44 | 33.03 | 23.59 |
| 150M | 28.31 | 127.40 | 99.09 |
| 300M | 75.50 | 490.73 | 415.24 |
| 600M | 198.18 | 1,486.36 | 1,288.18 |
| 1B | 402.65 | 3,422.55 | 3,019.90 |

These are **forward QKV projection counts only**, not a full model or training
FLOP estimate. They exclude the predictor, correction, mixing, attention, MLP,
output projection, and backward pass. The identity is about the current
project-then-mix implementation; it is not a lower bound for all possible
implementations of this architecture.

The external predictor is a separate causal Transformer encoder with its own
embedding table. Parameter count alone neither charges repeated shared-weight
projections correctly nor describes the cost of an embedding lookup. A claim
about compute scaling needs an operation count or measured profile of the actual
forward and backward implementation, not `6ND` plus only the mixing cost from the
older note. The reviewed figures therefore use model parameters and explicitly
reported token budgets, rather than labeling their x-axis as measured FLOPs.

## Energy-Based Transformers at ICLR 2026

Primary source: [conference paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/e19a65fd53b6f9a88b354da98813465d-Paper-Conference.pdf),
*Energy-Based Transformers are Scalable Learners and Thinkers*.

The reported maximum is approximately 800M parameters. Figure 5 studies
parameter/FLOP trends; Appendix D.1.1 uses five sizes, proportional token budgets,
a common FineWeb setup for those experiments, and three seeds for the three
smallest sizes. Tables D.1–D.2 give the raw losses and distinguish non-embedding
parameter counts. The authors also study other scaling axes.

Thus models below 1B can support an architectural scaling study. Their claim of
a steeper fitted exponent is a separate claim from retaining an advantage as
models grow. Our current evidence supports the latter more directly. EBT also
explicitly reports worse present-day FLOP efficiency despite its fitted trend;
a favorable slope and current compute efficiency are different quantities.

This comparison does not establish that our mixed-data, partly incomplete-budget
checkpoint collection follows the same controlled scaling protocol.
