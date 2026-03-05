"""Baseline pretraining for mini OLMo-2 300M.

Standard language model pretraining — no DAGFormer, no predictor, no A matrix.
Creates an OLMo-2-style model from scratch with random initialization and
weight tying. Uses the same data pipeline as DAGFormer for fair comparison.

Architecture: 12 layers, 16 heads, hidden=1024, intermediate=4096 (~304M params)
Hyperparams based on OLMo/Pythia/Step Law literature:
  lr=5e-4, beta2=0.95, weight_decay=0.1, grad_clip=1.0, linear decay

Usage:
    # Single GPU
    python scripts/pretrain_baseline.py --config configs/pretrain_300m.yaml

    # Multi-GPU (4× A40)
    torchrun --nproc_per_node=4 scripts/pretrain_baseline.py --config configs/pretrain_300m.yaml
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import math
import os
import re
import signal
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, Olmo2Config, Olmo2ForCausalLM

from src.data.dolma import build_eval_dataloader, build_train_dataloader
from src.utils.logging import finish_wandb, init_wandb, log_metrics


class CSVLogger:
    """Append-mode CSV logger. Auto-extends columns when new keys appear."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.columns: list[str] = ["step"]
        self._rows: list[dict[str, Any]] = []

        # If file exists (resume), read existing columns
        if os.path.exists(path):
            with open(path) as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    self.columns = list(header)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        row: dict[str, Any] = {"step": step}
        for k, v in metrics.items():
            if k not in self.columns:
                self.columns.append(k)
            row[k] = v

        # Append to file
        write_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        if write_header:
            # Write header + this row
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()
                writer.writerow(row)
        else:
            # Check if we need to rewrite header (new columns appeared)
            with open(self.path) as f:
                existing_header = next(csv.reader(f), [])
            if set(self.columns) != set(existing_header):
                # Rewrite with new header — read all, rewrite
                rows_on_disk: list[dict[str, Any]] = []
                with open(self.path) as f:
                    reader = csv.DictReader(f)
                    rows_on_disk = list(reader)
                rows_on_disk.append(row)
                with open(self.path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.columns)
                    writer.writeheader()
                    writer.writerows(rows_on_disk)
            else:
                with open(self.path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.columns)
                    writer.writerow(row)


@dataclass
class PretrainConfig:
    """Configuration for baseline pretraining."""

    # Model architecture
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    vocab_size: int = 100352
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 4096

    # Tokenizer (reuse OLMo-2-1B tokenizer for same vocab)
    tokenizer_id: str = "allenai/OLMo-2-0425-1B"

    # Data
    dataset: str = "allenai/dolma"
    dataset_name: str = "v1_7"
    seq_len: int = 1024
    seed: int = 42

    # Training
    micro_batch_size: int = 32          # per-GPU micro batch
    gradient_accumulation_steps: int = 4  # effective batch = micro × accum × world_size
    total_steps: int = 10000
    lr: float = 5e-4
    beta1: float = 0.9
    beta2: float = 0.95                 # LLM standard (NOT 0.999)
    weight_decay: float = 0.1           # LLM standard (NOT 0.01)
    warmup_steps: int = 300
    max_grad_norm: float = 1.0
    lr_schedule: str = "linear"         # "linear" (decay to 0) or "cosine"

    # Eval
    eval_skip: int = 1_000_000
    eval_size: int = 50
    eval_every: int = 500

    # Logging
    wandb_project: str = "dagformer"
    wandb_run_name: str = "pretrain-300m-baseline"
    log_every: int = 10

    # Checkpointing
    save_every: int = 2000
    save_dir: str = "checkpoints/pretrain_300m_baseline"
    resume_from: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> PretrainConfig:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)

        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data.keys()) - known_keys
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")

        import dataclasses
        for f in dataclasses.fields(cls):
            if f.name in data:
                expected_type = f.type
                if expected_type == "float" or expected_type is float:
                    data[f.name] = float(data[f.name])
                elif expected_type == "int" or expected_type is int:
                    data[f.name] = int(data[f.name])
                elif expected_type == "bool" or expected_type is bool:
                    data[f.name] = bool(data[f.name])

        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_model(config: PretrainConfig) -> Olmo2ForCausalLM:
    """Create randomly initialized OLMo-2 model from config."""
    model_config = Olmo2Config(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_attention_heads,  # MHA (not GQA)
        intermediate_size=config.intermediate_size,
        vocab_size=config.vocab_size,
        tie_word_embeddings=config.tie_word_embeddings,
        max_position_embeddings=config.max_position_embeddings,
    )
    model = Olmo2ForCausalLM(model_config)

    # Count params (deduplicate tied weights)
    seen_data_ptrs: set[int] = set()
    unique_params = 0
    total_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.data_ptr() not in seen_data_ptrs:
            seen_data_ptrs.add(p.data_ptr())
            unique_params += p.numel()

    head_dim = config.hidden_size // config.num_attention_heads
    print(f"Model: {unique_params:,} unique params ({total_params:,} total)")
    print(f"  hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, head_dim={head_dim}")
    if config.tie_word_embeddings:
        print(f"  weight tying: ON (saves {config.vocab_size * config.hidden_size:,} params)")

    return model


def get_lr(step: int, config: PretrainConfig) -> float:
    """Compute learning rate with linear warmup + decay."""
    if step < config.warmup_steps:
        return config.lr * step / max(1, config.warmup_steps)

    # Decay phase
    progress = (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
    progress = min(progress, 1.0)

    if config.lr_schedule == "linear":
        return config.lr * max(0.0, 1.0 - progress)
    else:  # cosine
        return config.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    save_dir: str,
    step: int,
    model: Olmo2ForCausalLM,
    optimizer: torch.optim.Optimizer,
    best_eval_nll: float,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"checkpoint_step{step}.pt")
    state = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_eval_nll": best_eval_nll,
    }
    torch.save(state, path)
    print(f"Checkpoint saved: {path}")
    return path


def find_latest_checkpoint(save_dir: str) -> Optional[str]:
    if not os.path.isdir(save_dir):
        return None
    pattern = os.path.join(save_dir, "checkpoint_step*.pt")
    files = glob.glob(pattern)
    if not files:
        return None

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline OLMo-2 300M pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = PretrainConfig.from_yaml(args.config)

    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = (local_rank == 0)

    # Reproducibility
    torch.manual_seed(config.seed + local_rank)
    torch.cuda.manual_seed_all(config.seed + local_rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Compute effective batch info
    tokens_per_step = config.micro_batch_size * config.gradient_accumulation_steps * world_size * config.seq_len
    total_tokens = tokens_per_step * config.total_steps

    if is_main:
        print(f"{'=' * 60}")
        print(f"Mini OLMo-2 Baseline Pretraining")
        print(f"{'=' * 60}")
        print(f"World size: {world_size}")
        print(f"Tokens/step: {tokens_per_step:,} "
              f"({config.micro_batch_size} × {config.gradient_accumulation_steps} accum "
              f"× {world_size} GPUs × {config.seq_len} seq)")
        print(f"Total tokens: {total_tokens / 1e9:.2f}B ({config.total_steps} steps)")
        print()

    # Create model
    if is_main:
        model = create_model(config)
    else:
        model = create_model(config)

    model = model.to(device, dtype=torch.bfloat16)

    # DDP wrapping
    model_raw = model
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
        model_raw = model.module

    # Optimizer (only over unique params to avoid double-counting tied weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    # Tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)

    # Data
    train_loader = build_train_dataloader(
        olmo_tokenizer=tokenizer,
        seq_len=config.seq_len,
        batch_size=config.micro_batch_size,
        dataset_name=config.dataset,
        dataset_version=config.dataset_name,
        rank=local_rank,
        world_size=world_size,
    )

    # Eval data (rank 0 only)
    eval_batches: list[dict] = []
    if is_main:
        cache_path = os.path.join(config.save_dir, "eval_cache.pt")
        eval_batches = build_eval_dataloader(
            olmo_tokenizer=tokenizer,
            seq_len=config.seq_len,
            batch_size=config.micro_batch_size,
            dataset_name=config.dataset,
            dataset_version=config.dataset_name,
            eval_skip=config.eval_skip,
            eval_size=config.eval_size,
            cache_path=cache_path,
        )

    # Resume from checkpoint
    global_step = 0
    best_eval_nll = float("inf")
    resume_path = config.resume_from or find_latest_checkpoint(config.save_dir)
    if resume_path:
        if is_main:
            print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model_raw.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt["step"] + 1
        best_eval_nll = ckpt.get("best_eval_nll", float("inf"))
        del ckpt
        if is_main:
            print(f"Resumed at step {global_step}")

    # Wandb
    wandb_run = None
    if is_main:
        wandb_run = init_wandb(
            project=config.wandb_project,
            run_name=config.wandb_run_name,
            config=config.to_dict(),
        )

    # CSV logger (always-on fallback for wandb)
    csv_logger: Optional[CSVLogger] = None
    if is_main:
        os.makedirs(config.save_dir, exist_ok=True)
        csv_path = os.path.join(config.save_dir, "metrics.csv")
        csv_logger = CSVLogger(csv_path)
        print(f"CSV logging to: {csv_path}")

    # Signal handler for SLURM preemption
    def save_on_signal(signum: int, frame: Any) -> None:
        if is_main:
            print(f"\nSignal {signum}, saving checkpoint...")
            save_checkpoint(config.save_dir, global_step, model_raw, optimizer, best_eval_nll)
        raise SystemExit(0)

    signal.signal(signal.SIGUSR1, save_on_signal)

    # ── Training loop ──
    train_iter = iter(train_loader)
    vocab_size = config.vocab_size
    accum_steps = config.gradient_accumulation_steps

    if is_main:
        print(f"\nTraining starts at step {global_step}")
        print()

    t0 = time.time()
    model.train()

    while global_step < config.total_steps:
        # Set LR
        lr = get_lr(global_step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation (use no_sync for intermediate micro-batches)
        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            input_ids = batch["olmo_ids"].to(device)
            labels = batch["olmo_labels"].to(device)

            # Skip DDP gradient sync for all but the last micro-batch
            is_last_micro = (micro_step == accum_steps - 1)
            sync_ctx = contextlib.nullcontext() if is_last_micro or world_size == 1 \
                else model.no_sync()

            with sync_ctx:
                outputs = model(input_ids=input_ids)
                loss = F.cross_entropy(
                    outputs.logits.contiguous().view(-1, vocab_size),
                    labels.contiguous().view(-1),
                )
                loss = loss / accum_steps
                loss.backward()

            accum_loss += loss.item()

        # Gradient clipping
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        # Step
        optimizer.step()

        # Logging
        if is_main and global_step % config.log_every == 0:
            elapsed = time.time() - t0
            tokens_seen = (global_step + 1) * tokens_per_step
            tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0

            # Gradient norm
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5

            metrics = {
                "train/loss": accum_loss,
                "train/lr": lr,
                "train/tokens_per_sec": tok_per_sec,
                "train/tokens_seen_B": tokens_seen / 1e9,
                "train/grad_norm": grad_norm,
            }
            log_metrics(metrics, global_step, wandb_run)
            if csv_logger is not None:
                csv_logger.log(global_step, metrics)

        # Eval
        if is_main and global_step > 0 and global_step % config.eval_every == 0:
            model.eval()
            eval_loss_total = 0.0
            n_eval = 0

            with torch.no_grad():
                for eb in eval_batches:
                    eids = eb["olmo_ids"].to(device)
                    elabels = eb["olmo_labels"].to(device)
                    eout = model_raw(input_ids=eids)
                    eloss = F.cross_entropy(
                        eout.logits.contiguous().view(-1, vocab_size),
                        elabels.contiguous().view(-1),
                    )
                    eval_loss_total += eloss.item()
                    n_eval += 1

            eval_nll = eval_loss_total / max(n_eval, 1)
            eval_metrics = {"eval/nll": eval_nll}
            log_metrics(eval_metrics, global_step, wandb_run)
            if csv_logger is not None:
                csv_logger.log(global_step, eval_metrics)

            if eval_nll < best_eval_nll:
                best_eval_nll = eval_nll
                print(f"  New best eval NLL: {eval_nll:.4f}")

            model.train()

        # Checkpoint
        if is_main and global_step > 0 and global_step % config.save_every == 0:
            save_checkpoint(config.save_dir, global_step, model_raw, optimizer, best_eval_nll)

        global_step += 1

        if world_size > 1:
            dist.barrier()

    # ── Final eval & save ──
    if is_main:
        model.eval()
        eval_loss_total = 0.0
        n_eval = 0
        with torch.no_grad():
            for eb in eval_batches:
                eids = eb["olmo_ids"].to(device)
                elabels = eb["olmo_labels"].to(device)
                eout = model_raw(input_ids=eids)
                eloss = F.cross_entropy(
                    eout.logits.contiguous().view(-1, vocab_size),
                    elabels.contiguous().view(-1),
                )
                eval_loss_total += eloss.item()
                n_eval += 1

        final_nll = eval_loss_total / max(n_eval, 1)
        print(f"\nFinal eval NLL: {final_nll:.4f}")

        save_checkpoint(config.save_dir, global_step, model_raw, optimizer, best_eval_nll)

    finish_wandb(wandb_run)

    if world_size > 1:
        dist.destroy_process_group()

    if is_main:
        print("Training complete.")


if __name__ == "__main__":
    main()
