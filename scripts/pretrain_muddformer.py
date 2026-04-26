"""Train a MUDDFormer baseline at our scaling-study sizes.

Mirrors scripts/pretrain_baseline.py exactly, except `create_model()` returns
a MUDDFormerForCausalLM (HF-wrapped) instead of Olmo2ForCausalLM. All training
loop / DDP / data loader logic is reused from pretrain_baseline.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pull every public symbol from pretrain_baseline so the module is a drop-in
from scripts import pretrain_baseline as _base  # noqa: E402
from src.model.muddformer.wrapper import build_muddformer  # noqa: E402


def create_model(config):
    """Replace baseline's create_model with MUDDFormer build."""
    model = build_muddformer(config)
    if hasattr(_base, "is_main") and _base.is_main:
        n = sum(p.numel() for p in model.parameters())
        print(f"MUDDFormer params: {n:,} ({n/1e6:.1f} M)  [dim={config.hidden_size}, "
              f"L={config.num_hidden_layers}, H={config.num_attention_heads}]")
    return model


# Monkey-patch baseline's create_model so its `main()` builds MUDDFormer
_base.create_model = create_model


if __name__ == "__main__":
    _base.main()
