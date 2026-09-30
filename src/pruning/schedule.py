"""Gradual sparsity schedule (Zhu & Gupta, 2017, "To prune, or not to prune").

    s(t) = s_f + (s_i - s_f) * (1 - (t - t0) / (t1 - t0))^3     for t0 <= t <= t1

Sparsity ramps quickly at first (when there is redundancy to spare) and slows
down as it approaches the final target, giving the network time to recover
from each pruning step. Pruning is applied at discrete steps t0, t0+dt, ...,
t1 (inclusive), and training continues after t1 with the final masks fixed.
"""
from __future__ import annotations


def cubic_sparsity(step: int, s_init: float, s_final: float,
                   t_start: int, t_end: int) -> float:
    """Target sparsity at ``step`` under the cubic schedule.

    Before ``t_start`` the target is ``s_init``; after ``t_end`` it is
    ``s_final``. ``t_end == t_start`` degenerates to one-shot pruning at
    ``t_start``.
    """
    assert 0.0 <= s_init <= s_final <= 1.0, (s_init, s_final)
    assert t_end >= t_start >= 0, (t_start, t_end)
    if step < t_start:
        return s_init
    if step >= t_end:
        return s_final
    frac = (step - t_start) / float(t_end - t_start)
    return s_final + (s_init - s_final) * (1.0 - frac) ** 3


def prune_steps(t_start: int, t_end: int, every: int) -> list[int]:
    """Steps at which a pruning update is applied: t_start, t_start+every, ...,
    plus t_end itself so the final target is always reached exactly."""
    assert every >= 1, every
    assert t_end >= t_start >= 0, (t_start, t_end)
    steps = list(range(t_start, t_end, every))
    if not steps or steps[-1] != t_end:
        steps.append(t_end)
    return steps
