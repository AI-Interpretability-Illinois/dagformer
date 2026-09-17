"""Copy a checkpoint and its side file into a self-contained run/eval directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil

import torch
import yaml


def stage(args: argparse.Namespace) -> dict:
    source = args.checkpoint.resolve()
    destination = args.out_dir.resolve()
    marker = destination / "staging.json"
    if marker.exists():
        previous = json.loads(marker.read_text())
        if (previous["source_checkpoint"] != str(source)
                or previous["includes_optimizer"] != args.keep_optimizer):
            raise ValueError(f"Destination contains another checkpoint: {destination}")
        for name, size in previous["files"].items():
            if (destination / name).stat().st_size != size:
                raise ValueError(f"Incomplete staged file: {destination / name}")
        return previous

    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(source, map_location="cpu", mmap=True, weights_only=False)
    config = yaml.safe_load(args.config.read_text())
    if args.training_data_source:
        config["data_source"] = args.training_data_source
        if args.training_data_source == "stream":
            config.pop("mmap_index_path", None)

    optimizer = checkpoint.get("optimizer_state_dict", {})
    update_counts = sorted({
        int(s["step"].item() if hasattr(s["step"], "item") else s["step"])
        for s in optimizer.get("state", {}).values() if "step" in s
    })
    payload = {key: value for key, value in checkpoint.items()
               if args.keep_optimizer or key != "optimizer_state_dict"}
    files = {}
    if "model_state_path" in checkpoint:
        recorded_side = Path(checkpoint["model_state_path"])
        adjacent_side = source.parent / recorded_side.name
        side_source = adjacent_side if adjacent_side.is_file() else recorded_side
        side_destination = destination / side_source.name
        temporary_side = side_destination.with_name(side_destination.name + ".tmp")
        shutil.copyfile(side_source, temporary_side)
        os.replace(temporary_side, side_destination)
        # The current trainer reads this path directly; evaluation also accepts
        # the relative basename when the optimizer is omitted.
        payload["model_state_path"] = (str(side_destination) if args.keep_optimizer
                                       else side_destination.name)
        files[side_destination.name] = side_destination.stat().st_size

    checkpoint_path = destination / "checkpoint.pt"
    temporary_checkpoint = destination / "checkpoint.pt.tmp"
    torch.save(payload, temporary_checkpoint)
    os.replace(temporary_checkpoint, checkpoint_path)
    files[checkpoint_path.name] = checkpoint_path.stat().st_size
    config_path = destination / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    files[config_path.name] = config_path.stat().st_size
    result = {
        "source_checkpoint": str(source),
        "source_config": str(args.config.resolve()),
        "checkpoint_step": int(checkpoint["step"]),
        "optimizer_update_counts": update_counts,
        "tokens_per_update": args.tokens_per_update,
        "processed_tokens": (update_counts[0] * args.tokens_per_update
                             if len(update_counts) == 1 and args.tokens_per_update else None),
        "includes_optimizer": args.keep_optimizer,
        "training_data_source_override": args.training_data_source,
        "files": files,
    }
    marker.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--keep-optimizer", action="store_true")
    parser.add_argument("--tokens-per-update", type=int)
    parser.add_argument("--training-data-source", choices=("stream", "mmap"))
    args = parser.parse_args()
    print(json.dumps(stage(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
