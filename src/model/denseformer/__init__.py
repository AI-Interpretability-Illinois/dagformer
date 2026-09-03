"""DenseFormer baseline: HF-Olmo2 decoder + vendored Depth-Weighted-Average.

Public API mirrors src/model/muddformer:
    from src.model.denseformer import build_denseformer
"""
from .dwa import DWAModules
from .wrapper import DenseFormerForCausalLM, build_denseformer

__all__ = ["DWAModules", "DenseFormerForCausalLM", "build_denseformer"]
