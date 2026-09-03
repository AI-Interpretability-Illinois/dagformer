"""Train a DenseFormer baseline at our scaling-study sizes.

Mirrors scripts/pretrain_baseline.py exactly, except `create_model()` returns
a DenseFormerForCausalLM (HF-Olmo2 backbone + Depth-Weighted-Average) instead
of a plain Olmo2ForCausalLM. All training loop / DDP / data loader logic is
reused from pretrain_baseline. DenseFormer differs from the dense baseline only
by the DWA taps (~L²/2 extra scalar params), so this is a fair Pareto baseline.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pull every public symbol from pretrain_baseline so the module is a drop-in
from scripts import pretrain_baseline as _base  # noqa: E402
from src.model.denseformer.wrapper import build_denseformer  # noqa: E402


def create_model(config):
    """Replace baseline's create_model with DenseFormer build."""
    model = build_denseformer(config)
    if hasattr(_base, "is_main") and _base.is_main:
        n = sum(p.numel() for p in model.parameters())
        print(f"DenseFormer params: {n:,} ({n/1e6:.1f} M)  [dim={config.hidden_size}, "
              f"L={config.num_hidden_layers}, H={config.num_attention_heads}]")
    return model


# Monkey-patch baseline's create_model so its `main()` builds DenseFormer
_base.create_model = create_model


if __name__ == "__main__":
    _base.main()
