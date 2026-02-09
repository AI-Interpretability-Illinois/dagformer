"""Entry point for DAGFormer training.

Usage:
    # Single GPU:
    python scripts/train.py --config configs/sanity_check.yaml

    # Multi-GPU (DDP):
    torchrun --nproc_per_node=4 scripts/train.py --config configs/phase1_full.yaml
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from src.training.trainer import TrainConfig, Trainer


def main():
    parser = argparse.ArgumentParser(description="Train DAGFormer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = TrainConfig.from_yaml(args.config)

    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    trainer = Trainer(config, local_rank=local_rank, world_size=world_size)
    trainer.train()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
