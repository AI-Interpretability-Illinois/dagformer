"""Train a manifold-constrained Hyper-Connections (mHC; arXiv:2512.24880, official DeepSeek reference math) baseline at our scaling-study sizes.

Mirrors scripts/pretrain_baseline.py exactly, except that `create_model()` returns the OLMo-2 decoder with one
connection module per sub-layer from src/model/hyperconnection/paper_exact.py (see its docstring for what is taken from
the paper, the official code and the community implementation). All training loop / DDP / data loader logic is reused
from pretrain_baseline.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import pretrain_baseline as _base  # noqa: E402
from src.model.hyperconnection.paper_exact import build_mhc  # noqa: E402


def create_model(config):
    model = build_mhc(config)
    if hasattr(_base, "is_main") and _base.is_main:
        n = sum(p.numel() for p in model.parameters())
        extra = sum(p.numel() for k, p in model.named_parameters() if "connections" in k or "mhc_head" in k)
        print(f"mHC (n=4) params: {n:,} ({n/1e6:.1f} M; connections {extra/1e6:.2f} M)  [dim={config.hidden_size}, "
              f"L={config.num_hidden_layers}, H={config.num_attention_heads}, streams=4, one module per sub-layer]")
    return model


_base.create_model = create_model


if __name__ == "__main__":
    _base.main()
