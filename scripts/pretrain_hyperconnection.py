"""Train a Hyper-Connections baseline at our scaling-study sizes.

Mirrors scripts/pretrain_baseline.py exactly, except `create_model()` returns
a HyperConnectionOlmo2ForCausalLM (Olmo2 backbone + n parallel residual
streams) instead of the plain Olmo2ForCausalLM. All training loop / DDP / data
loader logic is reused from pretrain_baseline.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Pull every public symbol from pretrain_baseline so the module is a drop-in
from scripts import pretrain_baseline as _base  # noqa: E402
from src.model.hyperconnection.wrapper import build_hyperconnection  # noqa: E402


def create_model(config):
    """Replace baseline's create_model with the Hyper-Connections build."""
    model = build_hyperconnection(config)
    if hasattr(_base, "is_main") and _base.is_main:
        n = sum(p.numel() for p in model.parameters())
        print(f"HyperConnection params: {n:,} ({n/1e6:.1f} M)  [dim={config.hidden_size}, "
              f"L={config.num_hidden_layers}, H={config.num_attention_heads}, "
              f"streams={getattr(config, 'hc_num_streams', 4)}, "
              f"dynamic={getattr(config, 'hc_dynamic', False)}]")
    return model


# Monkey-patch baseline's create_model so its `main()` builds Hyper-Connections
_base.create_model = create_model


if __name__ == "__main__":
    _base.main()
