"""Fused Triton kernel: weighted-sum of layer outputs + QKV projection.

Avoids materializing per-head mixed inputs in HBM.
Computes X_h = Σ α[h,j] * layer_outputs[j] on-chip and immediately
multiplies by W_proj to produce Q, K, or V.

All layers padded to max_L sources. α=0 for non-existent sources.

Input:
    layer_outputs: [max_L, N, D]   — padded stacked layer outputs (N = B*T)
    alpha:         [N, H, max_L]   — per-head routing weights (padded, 0 for unused)
    W_proj:        [D, H*hd]       — projection weight (q_proj, k_proj, or v_proj)

Output:
    out: [N, H, hd]  — projected per-head output
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_routing_proj_kernel(
    # Pointers
    layer_out_ptr,   # [max_L, N, D]
    alpha_ptr,       # [N, H, max_L]
    w_proj_ptr,      # [D, H*hd]
    out_ptr,         # [N, H, hd]
    # Dimensions
    N,                     # B*T (total tokens)
    D: tl.constexpr,      # model dim
    H: tl.constexpr,      # num heads
    MAX_L: tl.constexpr,  # max number of source layers (padded)
    HD: tl.constexpr,     # head dim
    # Block sizes
    BLOCK_N: tl.constexpr,   # tokens per block
    BLOCK_D: tl.constexpr,   # model dim tile
    BLOCK_HD: tl.constexpr,  # head dim tile
):
    """Each program handles BLOCK_N tokens × all heads."""
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    n_mask = n_offsets < N

    # Loop over heads
    for h in range(H):
        # alpha base offset for this head
        # alpha layout: [N, H, max_L], stride: (H*max_L, max_L, 1)
        alpha_base = n_offsets * (H * MAX_L) + h * MAX_L  # [BLOCK_N]

        # Output base for this head
        out_base = n_offsets * (H * HD) + h * HD  # [BLOCK_N]

        # Process head_dim in tiles
        for hd_start in range(0, HD, BLOCK_HD):
            hd_offsets = hd_start + tl.arange(0, BLOCK_HD)  # [BLOCK_HD]
            hd_mask = hd_offsets < HD

            # Accumulator: [BLOCK_N, BLOCK_HD]
            acc = tl.zeros([BLOCK_N, BLOCK_HD], dtype=tl.float32)

            # Process model_dim in tiles
            for d_start in range(0, D, BLOCK_D):
                d_offsets = d_start + tl.arange(0, BLOCK_D)  # [BLOCK_D]
                d_mask = d_offsets < D

                # Load W_proj[d, h*HD + hd] tile: [BLOCK_D, BLOCK_HD]
                w_rows = d_offsets[:, None]  # [BLOCK_D, 1]
                w_cols = h * HD + hd_offsets[None, :]  # [1, BLOCK_HD]
                w_tile = tl.load(
                    w_proj_ptr + w_rows * (H * HD) + w_cols,
                    mask=d_mask[:, None] & hd_mask[None, :],
                    other=0.0,
                )
                # w_tile: [BLOCK_D, BLOCK_HD]

                # Compute weighted sum of layer outputs at this d-tile
                # x_mixed[n, d] = Σ_j alpha[n,j] * layer_out[j, n, d]
                x_mixed = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
                for j in range(MAX_L):
                    # Load alpha[n, h, j] — scalar per token
                    a_j = tl.load(alpha_ptr + alpha_base + j, mask=n_mask, other=0.0)
                    # layer_out[j, n, d]
                    lo_offsets = j * N * D + n_offsets[:, None] * D + d_offsets[None, :]
                    lo_tile = tl.load(
                        layer_out_ptr + lo_offsets,
                        mask=n_mask[:, None] & d_mask[None, :],
                        other=0.0,
                    )
                    # lo_tile: [BLOCK_N, BLOCK_D], a_j: [BLOCK_N]
                    x_mixed += a_j[:, None] * lo_tile

                # acc += x_mixed @ w_tile: [BLOCK_N, BLOCK_D] @ [BLOCK_D, BLOCK_HD]
                acc += tl.dot(x_mixed.to(w_tile.dtype), w_tile).to(tl.float32)

            # Store output: out[n, h, hd]
            out_offsets = out_base[:, None] + hd_offsets[None, :]  # [BLOCK_N, BLOCK_HD]
            tl.store(
                out_ptr + out_offsets,
                acc.to(out_ptr.dtype.element_ty),
                mask=n_mask[:, None] & hd_mask[None, :],
            )


def fused_routing_proj(
    layer_outputs: torch.Tensor,  # [max_L, N, D]
    alpha: torch.Tensor,          # [N, H, max_L]
    w_proj: torch.Tensor,         # [D, H*hd]
) -> torch.Tensor:
    """Fused weighted-sum + projection.

    Args:
        layer_outputs: [max_L, N, D] padded stacked layer outputs (N = B*T)
        alpha: [N, H, max_L] routing weights for one stream (padded with 0)
        w_proj: [D, H*hd] projection weight matrix

    Returns:
        out: [N, H, hd] projected per-head output
    """
    MAX_L, N, D = layer_outputs.shape
    H_times_hd = w_proj.shape[1]
    H = alpha.shape[1]
    HD = H_times_hd // H

    assert alpha.shape == (N, H, MAX_L), f"alpha {alpha.shape} != ({N}, {H}, {MAX_L})"

    out = torch.empty(N, H, HD, device=layer_outputs.device, dtype=layer_outputs.dtype)

    # Ensure contiguous
    layer_outputs = layer_outputs.contiguous()
    alpha = alpha.contiguous().float()
    w_proj = w_proj.contiguous()

    # Block sizes
    BLOCK_N = 64   # tokens per program
    BLOCK_D = 64   # model dim tile
    BLOCK_HD = 64  # head dim tile (= head_dim for D=1024, H=16, HD=64)

    grid = (triton.cdiv(N, BLOCK_N),)

    _fused_routing_proj_kernel[grid](
        layer_outputs, alpha, w_proj, out,
        N=N, D=D, H=H, MAX_L=MAX_L, HD=HD,
        BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, BLOCK_HD=BLOCK_HD,
    )

    return out
