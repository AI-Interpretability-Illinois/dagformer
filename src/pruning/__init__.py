"""Structured pruning during finetuning for dense OLMo-2 and DAGFormer.

The two public pieces:

    StructuredMasker   multiplicative unit masks (heads, MLP neurons, whole
                       attention / MLP modules) attached to an
                       ``Olmo2ForCausalLM`` through forward hooks, so the same
                       object prunes a dense baseline and the OLMo backbone
                       inside ``FourWayDAGFormer`` / ``FourWayModularDAGFormer``.
    cubic_sparsity     the gradual schedule of Zhu & Gupta (2017): target
                       sparsity rises as a cubic from ``s_init`` to ``s_final``
                       between two steps, and the masker is asked to reach the
                       current target every ``prune_every`` steps.
"""
from src.pruning.masks import (  # noqa: F401
    UNIT_TYPES,
    StructuredMasker,
    unit_param_counts,
)
from src.pruning.schedule import cubic_sparsity, prune_steps  # noqa: F401
