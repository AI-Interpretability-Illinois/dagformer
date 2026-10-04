"""Copy a stopped scaling run without changing its optimizer or token position."""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import time


def relocate_resume_checkpoint(path, planned_updates, family):
    import torch

    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    updates = state.get("completed_updates")
    steps = {int(s["step"]) for s in state.get("optimizer_state_dict", {}).get("state", {}).values()
             if "step" in s}
    if updates is None or steps != {updates} or not 0 < updates < planned_updates:
        raise ValueError(f"Invalid resume progress: updates={updates}, optimizer={steps}, planned={planned_updates}")
    if family == "fourway_corrected" and not (state.get("predictor_state_dict") and state.get("routing_state_dict")):
        raise ValueError("The corrected model needs both predictor and routing state")
    if "model_state_path" in state:
        companion = path.parent / Path(state["model_state_path"]).name
        # Read the entire companion before the caller cancels its remote job.
        model = torch.load(companion, map_location="cpu", weights_only=False)
        if not model:
            raise ValueError(f"Empty model companion: {companion}")
        del model
        if state["model_state_path"] != str(companion):
            state["model_state_path"] = str(companion)
            temporary = path.with_suffix(".migrating")
            torch.save(state, temporary)
            temporary.replace(path)
    elif not state.get("model_state_dict"):
        raise ValueError("Checkpoint has no backbone weights")
    return updates


def stage_remote_checkpoint(host, local_run, remote_run):
    for key in ("name", "family", "layers", "width", "seed", "tokens", "batch_tokens", "world_size", "optimizer_updates"):
        if local_run[key] != remote_run[key]:
            raise ValueError(f"Resume configuration differs: {key}")
    target = Path(local_run["checkpoint_dir"])
    target.mkdir(parents=True, exist_ok=True)
    rsync = ["rsync", "-a", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15"]
    subprocess.run([*rsync, "--include=*.pt", "--exclude=*",
                    f"{host}:{remote_run['checkpoint_dir']}/", str(target) + "/"], check=True, timeout=600)
    candidates = [(int(m.group(1)), p) for p in target.glob("checkpoint_step*.pt")
                  if (m := re.fullmatch(r"checkpoint_step(\d+)\.pt", p.name))]
    if not candidates:
        raise ValueError(f"No checkpoint to resume: {local_run['name']}")
    checkpoint = max(candidates)[1]
    updates = relocate_resume_checkpoint(checkpoint, local_run["optimizer_updates"], local_run["family"])
    meta = Path(local_run["metadata_dir"])
    meta.mkdir(parents=True, exist_ok=True)
    subprocess.run([*rsync, "--include=metrics.csv", "--include=train.log", "--include=invocations.jsonl", "--exclude=*",
                    f"{host}:{remote_run['metadata_dir']}/", str(meta) + "/"], check=True, timeout=90)
    record = {"source_host": host, "source_checkpoint_dir": remote_run["checkpoint_dir"],
              "checkpoint": checkpoint.name, "completed_updates": updates, "staged_unix": time.time()}
    (meta / "migration.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Staged {local_run['name']} at update {updates}", flush=True)
    return record
