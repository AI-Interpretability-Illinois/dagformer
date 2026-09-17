# Full GSM8K generation after continued training

The step-10500 DAGFormer checkpoint was evaluated on all 1,319 test questions
with the same three few-shot examples, greedy decoding, 256-new-token cap,
1,024-token context and batch size one as step 9000. Both recompute the prefix
without a KV cache. The model processed 5.506B rather than 4.719B training tokens.

Flexible numeric extraction changes from **20/1,319 (1.52%) to 26/1,319 (1.97%)**.
The paired difference is +0.45 percentage points, with 95% document-bootstrap
interval [-0.23, +1.21]; the exact McNemar p value is 0.3075. Fifteen documents
gain an extracted match and nine lose one. This does not detect an improvement
under the evaluated generation protocol.

Strict-format matching rises from zero to **3/1,319 (0.23%)**. All three matching
continuations are shown in the [deterministic examples](generation_examples.md):
document 450 uses unrelated people and a malformed multiplication; document
905 gives an unrelated ratio calculation; document 947 discusses can lids and
contains an incorrect sum before emitting the matching final number. A final
number match therefore does not verify the derivation, even with `####` syntax.

The mean repeated four-gram fraction is 0.840. The
[generation audit](generation_audit.md) retains official extraction metrics,
and the [paired comparison](paired_vs_dagformer.md) retains the document-level
uncertainty. This is a continued-training comparison of one model, separate
from the equal-token baseline/DAGFormer comparison at step 9000.
