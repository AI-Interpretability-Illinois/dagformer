"""Create a continuation config from a staged checkpoint and its recorded updates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
import yaml


def optimizer_updates(checkpoint: Path) -> int:
    obj = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    steps = {int(s["step"]) for s in obj["optimizer_state_dict"]["state"].values()
             if "step" in s}
    if len(steps) != 1:
        raise ValueError(f"Inconsistent optimizer update counts: {steps}")
    return steps.pop()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", choices=["1b-dagformer", "600m-dagformer", "600m-baseline"],
                   required=True)
    p.add_argument("--check-complete", action="store_true")
    args = p.parse_args()
    source = args.root / "checkpoints/source" / args.model
    run = args.root / "runs" / (args.model + "-complete")
    run.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((source / "config.yaml").read_text())
    checkpoints = sorted(run.glob("checkpoint_step*.pt"),
                         key=lambda path: int(path.stem.split("step")[1].split("_")[0]))
    checkpoints = [path for path in checkpoints if not path.stem.endswith("_model")]
    checkpoint = checkpoints[-1] if checkpoints else source / "checkpoint.pt"
    updates = optimizer_updates(checkpoint)
    if args.check_complete:
        if updates != config["total_steps"]:
            raise ValueError(f"Training incomplete: {updates} != {config['total_steps']}")
        print(checkpoint)
        return
    if updates >= config["total_steps"]:
        raise ValueError(f"Run already completed: {updates} updates")
    config.update(tokenizer_id=str(args.root / "tokenizer"),
                  save_dir=str(run), resume_from=str(checkpoint),
                  keep_last_n=2, save_every=500,
                  wandb_run_name=args.model + "-biro-completion-20260917")
    if args.model == "1b-dagformer":
        config.update(data_source="mmap", mmap_index_path=str(
            args.root / "data/pretok/olmo_mix_21b_reconstructed"))
    else:
        # Frozen nominal suffix of the original eight-rank Dolma stream.
        index_dir = args.root / "data/pretok/dolma_600m_continuation"
        index = json.loads((index_dir / "index.json").read_text())
        origin = index["provenance"]["origin_updates"]
        if origin != 14001 or updates < origin:
            raise ValueError("Unexpected 600M continuation origin")
        config.update(data_source="mmap", mmap_index_path=str(index_dir),
                      data_num_workers=0, data_block_size=index["total_tokens"] // 1025 + 1,
                      resume_skip_samples=(updates - origin) * 64)
    if args.model.endswith("dagformer"):
        config["resume_require_optimizer"] = True
    path = run / "resume.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    (run / "launch.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "optimizer_updates": updates,
        "target_updates": config["total_steps"], "tokens_per_update": 524288,
        "remaining_tokens": (config["total_steps"] - updates) * 524288,
        "data_index": config["mmap_index_path"],
    }, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()

