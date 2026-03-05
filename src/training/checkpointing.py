"""Checkpoint save/load for predictor + optimizer + schedule state.

Phase 1: Only saves predictor MLP, optimizer, schedule state.
Phase 2: Also saves OLMo state_dict (weights being fine-tuned).
Frozen Qwen is never checkpointed — loads from HuggingFace.
"""

from __future__ import annotations

import glob
import os
import re
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
    olmo: Optional[nn.Module] = None,
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
        olmo: OLMo model (Phase 2 only — saves fine-tuned weights)
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

    if olmo is not None:
        # Save OLMo state_dict separately — it's too large (~2.4GB) for
        # PyTorch's default zip serialization which crashes with
        # "unexpected pos" error on large files.
        olmo_path = path.replace(".pt", "_olmo.pt")
        torch.save(olmo.state_dict(), olmo_path, _use_new_zipfile_serialization=False)
        state["olmo_state_path"] = olmo_path
        print(f"OLMo state saved: {olmo_path}")

    torch.save(state, path)
    print(f"Checkpoint saved: {path}")
    return path


def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    """Find the latest checkpoint in save_dir by step number.

    Returns:
        Path to latest checkpoint, or None if no checkpoints found.
    """
    if not os.path.isdir(save_dir):
        return None

    pattern = os.path.join(save_dir, "checkpoint_step*.pt")
    files = glob.glob(pattern)
    if not files:
        return None

    # Extract step numbers and find max
    step_re = re.compile(r"checkpoint_step(\d+)\.pt$")
    best_step = -1
    best_path = None
    for f in files:
        m = step_re.search(f)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best_path = f

    return best_path


def load_checkpoint(
    path: str,
    predictor: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    olmo: Optional[nn.Module] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Load training checkpoint.

    Args:
        path: path to checkpoint file
        predictor: structure predictor to load weights into
        optimizer: optimizer to restore state (optional — skip for eval)
        scheduler: LR scheduler to restore state (optional)
        olmo: OLMo model to restore weights (Phase 2 only)
        device: device to map tensors to

    Returns:
        state dict with step, best_eval_nll, and any extras
    """
    map_location = device if device is not None else "cpu"
    state = torch.load(path, map_location=map_location)

    predictor.load_state_dict(state["predictor_state_dict"])
    print(f"Predictor state loaded from {path}")

    if olmo is not None:
        if "olmo_state_path" in state:
            # New format: OLMo saved separately
            olmo_path = state["olmo_state_path"]
            olmo_state = torch.load(olmo_path, map_location=map_location)
            olmo.load_state_dict(olmo_state)
            del olmo_state
            print(f"OLMo state loaded from {olmo_path}")
        elif "olmo_state_dict" in state:
            # Legacy format: OLMo in same file
            olmo.load_state_dict(state["olmo_state_dict"])
            print(f"OLMo state loaded from {path}")

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])

    return {
        "step": state["step"],
        "best_eval_nll": state.get("best_eval_nll", float("inf")),
    }
