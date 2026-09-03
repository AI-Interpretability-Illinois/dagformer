"""Hyper-Connections (arXiv:2409.19606) baseline on the Olmo2 backbone."""

from .wrapper import (
    HyperConnectionOlmo2ForCausalLM,
    HyperConnectionOlmo2Model,
    build_hyperconnection,
)

__all__ = [
    "build_hyperconnection",
    "HyperConnectionOlmo2ForCausalLM",
    "HyperConnectionOlmo2Model",
]
