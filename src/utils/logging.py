"""Wandb integration for DAGFormer training.

Logs all metrics from CLAUDE.md §7.
"""

from __future__ import annotations

from typing import Any, Optional

import wandb


def init_wandb(
    project: str,
    run_name: str,
    config: dict[str, Any],
) -> Optional[wandb.sdk.wandb_run.Run]:
    """Initialize wandb run.

    Args:
        project: wandb project name
        run_name: run display name
        config: full config dict to log

    Returns:
        wandb run object (or None if wandb init fails)
    """
    try:
        run = wandb.init(
            project=project,
            name=run_name,
            config=config,
        )
        return run
    except Exception as e:
        print(f"WARNING: wandb init failed: {e}. Continuing without wandb.")
        return None


def log_metrics(
    metrics: dict[str, float],
    step: int,
    run: Optional[wandb.sdk.wandb_run.Run] = None,
) -> None:
    """Log metrics to wandb.

    Args:
        metrics: dict of metric_name → value
        step: global training step
        run: wandb run (if None, prints to stdout instead)
    """
    if run is not None:
        wandb.log(metrics, step=step)
    else:
        # Fallback: print metrics
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()]
        print(f"[step {step}] {', '.join(parts)}")


def finish_wandb(run: Optional[wandb.sdk.wandb_run.Run] = None) -> None:
    """Finish wandb run."""
    if run is not None:
        wandb.finish()
