"""Schedule functions for temperature (τ), sparsity (λ), and learning rate.

All schedules are deterministic functions of the current step.
See CLAUDE.md §3.1 for exact formulas.
"""

from __future__ import annotations

import math


def tau_schedule(step: int, total_steps: int, tau_init: float, tau_final: float) -> float:
    """Cosine annealing for Gumbel-Sigmoid temperature.

    τ(t) = τ_f + 0.5(τ_i - τ_f)(1 + cos(πt/T))

    Starts at tau_init, ends at tau_final.
    """
    if total_steps <= 0:
        return tau_final
    progress = min(step / total_steps, 1.0)
    return tau_final + 0.5 * (tau_init - tau_final) * (1 + math.cos(math.pi * progress))


def lambda_schedule(step: int, total_steps: int, lambda_max: float, warmup_frac: float = 0.2) -> float:
    """Linear ramp for sparsity coefficient.

    Ramps linearly from 0 to lambda_max over first warmup_frac of training.
    """
    if lambda_max == 0.0:
        return 0.0
    warmup_steps = int(total_steps * warmup_frac)
    if warmup_steps <= 0:
        return lambda_max
    return lambda_max * min(step / warmup_steps, 1.0)
