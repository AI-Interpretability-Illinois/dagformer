"""Checkpoint save/load for predictor + optimizer + schedule state.

Only saves trainable components (predictor MLP, optimizer, schedule state).
Frozen models (OLMo, Qwen) are not checkpointed — they load from HuggingFace.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    save_dir: str,
    step: int,
    predictor: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    best_eval_nll: float,
    extra: Optional[dict] = None,
) -> str:
    """Save training checkpoint.

    Args:
        save_dir: directory to save checkpoint
        step: current global step
        predictor: the structure predictor (only MLP params are saved)
        optimizer: AdamW optimizer
        scheduler: LR scheduler
        best_eval_nll: best eval NLL so far
        extra: any additional state to save

    Returns:
        path: path to saved checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"checkpoint_step{step}.pt")

    state = {
        "step": step,
        "predictor_state_dict": predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_eval_nll": best_eval_nll,
    }
    if extra:
        state.update(extra)

    torch.save(state, path)
    print(f"Checkpoint saved: {path}")
    return path


def load_checkpoint(
    path: str,
    predictor: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Load training checkpoint.

    Args:
        path: path to checkpoint file
        predictor: structure predictor to load weights into
        optimizer: optimizer to restore state (optional — skip for eval)
        scheduler: LR scheduler to restore state (optional)
        device: device to map tensors to

    Returns:
        state dict with step, best_eval_nll, and any extras
    """
    map_location = device if device is not None else "cpu"
    state = torch.load(path, map_location=map_location)

    predictor.load_state_dict(state["predictor_state_dict"])
    print(f"Predictor state loaded from {path}")

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return {
        "step": state["step"],
        "best_eval_nll": state.get("best_eval_nll", float("inf")),
    }
