"""A matrix analysis utilities for logging and monitoring.

Computes topology metrics per CLAUDE.md §7.
"""

from __future__ import annotations

import torch

from src.model.olmo_graph import create_block_upper_triangular_mask


def compute_topology_metrics(A: torch.Tensor, num_heads: int = 16) -> dict[str, float]:
    """Compute topology metrics from adjacency matrix A.

    Args:
        A: [batch, 256, 256] — gate matrix
        num_heads: heads per layer (16 for OLMo2-1B)

    Returns:
        dict with metrics: mean_A, seq_gate_frac, hyp_gate_frac, jaccard_var
    """
    batch = A.shape[0]
    num_nodes = A.shape[1]
    mask = create_block_upper_triangular_mask(num_nodes, num_heads).to(A.device)

    # Mean A over valid entries
    valid_entries = A[:, mask.bool()]  # [batch, 30720]
    mean_A = valid_entries.mean().item()

    # Classify connections: adjacent (layer diff == 1) vs skip (layer diff > 1)
    layer_idx = torch.arange(num_nodes, device=A.device) // num_heads
    layer_diff = layer_idx.unsqueeze(0) - layer_idx.unsqueeze(1)  # [256, 256]
    # layer_diff[i,j] = layer(j) - layer(i)

    adj_mask = (layer_diff == 1) & mask.bool()  # adjacent-layer connections
    skip_mask = (layer_diff > 1) & mask.bool()   # skip connections

    # Fraction of gates > 0.5
    adj_vals = A[:, adj_mask]  # [batch, 3840]
    skip_vals = A[:, skip_mask]  # [batch, 26880]

    seq_gate_frac = (adj_vals > 0.5).float().mean().item() if adj_vals.numel() > 0 else 0.0
    hyp_gate_frac = (skip_vals > 0.5).float().mean().item() if skip_vals.numel() > 0 else 0.0

    # Jaccard variance across batch
    jaccard_var = _jaccard_variance(A, mask).item() if batch > 1 else 0.0

    return {
        "topology/mean_A": mean_A,
        "topology/seq_gate_frac": seq_gate_frac,
        "topology/hyp_gate_frac": hyp_gate_frac,
        "topology/jaccard_var": jaccard_var,
    }


def _jaccard_variance(A: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute variance of pairwise Jaccard similarity across batch.

    Measures how context-dependent the topologies are.
    Higher variance = more context-dependent routing.
    """
    batch = A.shape[0]
    if batch < 2:
        return torch.tensor(0.0)

    # Binarize at 0.5 threshold for Jaccard
    binary = (A > 0.5).float()
    valid = mask.bool()

    # Extract valid entries: [batch, num_valid]
    entries = binary[:, valid]

    # Pairwise Jaccard
    jaccards = []
    for i in range(batch):
        for j in range(i + 1, batch):
            intersection = (entries[i] * entries[j]).sum()
            union = ((entries[i] + entries[j]) > 0).float().sum()
            jaccard = intersection / union.clamp(min=1.0)
            jaccards.append(jaccard)

    if not jaccards:
        return torch.tensor(0.0)

    jaccards = torch.stack(jaccards)
    return jaccards.var()
