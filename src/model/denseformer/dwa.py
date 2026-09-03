"""Depth-Weighted-Average (DWA) module — vendored from epfml/DenseFormer.

Source: https://github.com/epfml/DenseFormer (denseformer/denseformer.py),
Pagliardini et al., "DenseFormer: Enhancing Information Flow in Transformers
via Depth Weighted Averaging", NeurIPS 2024 (arXiv:2402.02622).

Vendored verbatim (the package is not published on PyPI). The DWA module is
the ONLY architectural addition of DenseFormer: after each transformer block
`i`, it replaces the block output with a learned weighted average of
{embedded input h_0, all previous block outputs h_1..h_{i-1}, current block
output h_i}:

    x_i = Σ_{j≤i} α_{i,j} · h_j

α is a learned lower-triangular matrix of SCALARS (static / input-independent,
shared across all tokens and heads). The added parameter count is ~L²/2
(negligible vs the transformer weights).

`InPlaceSetSlice` is a custom autograd Function used to grow the accumulator
buffer in place while keeping a correct backward pass (it avoids reallocating
and re-stacking the whole `[i, B, T, D]` accumulator each block).

`dilation` sparsifies which past taps are averaged; `period` controls which
blocks apply DWA at all. Defaults (dilation=1, period=1) give the standard
*dense* DenseFormer that averages over every previous block.

Only cosmetic changes vs upstream: module docstring + type hints. The math and
initialization are unchanged so behaviour matches the reference implementation.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch


class InPlaceSetSlice(torch.autograd.Function):
    """Write ``x_val`` into ``full_tensor[x_idx]`` in place and return the
    ``[:x_idx+1]`` prefix view, with a hand-written backward so gradients flow
    to each accumulated slice without materialising a fresh stacked tensor."""

    @staticmethod
    def forward(ctx, full_tensor, last_slice, x_idx, x_val):  # type: ignore[override]
        full_tensor[x_idx] = x_val
        ctx.x_idx = x_idx
        # new_empty(0) inherits full_tensor's dtype+device; torch.Tensor() is
        # float32 and .set_() to a bf16 source raises "Could not set BFloat16 to float".
        ret = full_tensor.new_empty(0)
        ret.set_(full_tensor[: x_idx + 1])
        return ret

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        if ctx.x_idx == 0:
            return None, None, None, grad_out[ctx.x_idx]
        else:
            return None, grad_out[: ctx.x_idx], None, grad_out[ctx.x_idx]


def apply_inplace_set(
    x_acc: Tuple[torch.Tensor, Optional[torch.Tensor]],
    x_idx: int,
    x_val: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Store ``x_val`` at position ``x_idx`` of the accumulator tuple and
    return the updated ``(full_tensor, prefix_slice)`` pair."""
    full_tensor, last_slice = x_acc
    new_slice = InPlaceSetSlice.apply(full_tensor, last_slice, x_idx, x_val)
    return full_tensor, new_slice


class DWAModules(torch.nn.Module):
    """Depth-Weighted-Average taps for a stack of ``n_blocks`` transformer
    blocks.

    Args:
        n_blocks: number of transformer blocks (== ``num_hidden_layers``).
        dilation: sparsify past taps — only every ``dilation``-th accumulated
            state participates in the average (1 = dense, use all).
        period: only blocks whose ``(idx+1) % period == 0`` apply a DWA; the
            others pass their output through unchanged (1 = every block).

    The learnable parameters are the per-block ``alphas[i]`` linear layers
    (bias-free scalar mixers), initialized so that at start each block ≈ a
    plain residual (current-block tap weight = 1, all earlier taps = 0).
    """

    def __init__(self, n_blocks: int, dilation: int = 1, period: int = 1) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.dilation = dilation
        self.period = period
        self.alphas = torch.nn.ModuleList(
            [
                torch.nn.Linear((i + 1 + dilation) // dilation, 1, bias=False)
                if (i + 1) % period == 0
                else None
                for i in range(n_blocks)
            ]
        )
        self.accumulators: Optional[List[Tuple[torch.Tensor, Optional[torch.Tensor]]]] = None
        self._init_weights()

    def _init_weights(self) -> None:
        """Identity / plain-residual init: last tap (current block) = 1, rest 0."""
        for module in self.alphas:
            if module is not None:
                module.weight.data.zero_()
                module.weight.data[0, -1] = 1.0

    def init_accumulators(self, x: torch.Tensor) -> None:
        """Seed the depth accumulators with the embedded input ``x`` (= h_0).

        Must be called once at the start of every forward pass, before the
        block loop. Allocates ``dilation`` interleaved accumulator buffers.
        """
        x_accs = []
        for i in range(self.dilation):
            current_group_size = (self.n_blocks + 1) // self.dilation
            if i < (self.n_blocks + 1) % self.dilation:
                current_group_size += 1
            x_accs.append(
                (torch.zeros((current_group_size, *x.shape), device=x.device, dtype=x.dtype), None)
            )
        x_accs[0] = apply_inplace_set(x_accs[0], 0, x)
        self.accumulators = x_accs

    def forward(self, x: torch.Tensor, block_idx: int) -> torch.Tensor:
        """Record block ``block_idx``'s output ``x`` and return the depth
        weighted average (or ``x`` unchanged when this block has no DWA)."""
        assert self.accumulators is not None, "`init_accumulators(x)` needs to be called first"
        self.accumulators[(block_idx + 1) % self.dilation] = apply_inplace_set(
            self.accumulators[(block_idx + 1) % self.dilation],
            (block_idx + 1) // self.dilation,
            x,
        )
        if (block_idx + 1) % self.period == 0:
            x = torch.tensordot(
                self.alphas[block_idx].weight.view(-1),
                self.accumulators[(block_idx + 1) % self.dilation][1],
                dims=1,
            )
        return x
