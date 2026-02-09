"""Training loop for DAGFormer Phase 1.

Pure PyTorch + DDP. Only the predictor MLP is trainable.
See CLAUDE.md §3.1 for training specification.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dolma import build_eval_dataloader, build_train_dataloader
from src.model.olmo_graph import DAGFormerOLMo, create_all_ones_A
from src.model.predictor import StructurePredictor
from src.training.checkpointing import load_checkpoint, save_checkpoint
from src.training.schedulers import lambda_schedule, tau_schedule
from src.utils.logging import finish_wandb, init_wandb, log_metrics
from src.utils.topology import compute_topology_metrics

import torch.nn.functional as F


@dataclass
class TrainConfig:
    """Training configuration. Parsed from YAML."""

    # Model
    olmo_model_id: str = "allenai/OLMo-2-0425-1B"
    qwen_model_id: str = "Qwen/Qwen3-Embedding-0.6B"

    # Predictor
    predictor_hidden_dim: int = 1024
    predictor_rank: int = 32
    cascading_gate_k: float = 5.0
    input_norm: str = "none"
    qwen_input_prefix: str = ""

    # Data
    dataset: str = "allenai/dolma"
    dataset_name: str = "v1_7"
    seq_len: int = 1024
    batch_size: int = 4
    micro_batch_size: int = 4

    # Eval
    eval_skip: int = 1_000_000
    eval_size: int = 1_000

    # Training
    total_steps: int = 1000
    lr: float = 3e-4
    weight_decay: float = 0.01
    optimizer: str = "adamw"

    # Schedules
    tau_init: float = 5.0
    tau_final: float = 0.2
    tau_schedule: str = "cosine"
    lambda_max: float = 0.0
    lambda_warmup_frac: float = 0.2

    # Logging
    wandb_project: str = "dagformer"
    wandb_run_name: str = "default"
    log_every: int = 10
    eval_every: int = 100

    # Checkpointing
    save_every: int = 500
    save_dir: str = "checkpoints/"
    resume_from: str = ""

    # Hardware
    num_gpus: int = 1

    @classmethod
    def from_yaml(cls, path: str) -> TrainConfig:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)

        known_keys = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(data.keys()) - known_keys
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")

        # Coerce types to match dataclass field annotations
        import dataclasses
        for f in dataclasses.fields(cls):
            if f.name in data:
                expected_type = f.type
                if expected_type == "float" or expected_type is float:
                    data[f.name] = float(data[f.name])
                elif expected_type == "int" or expected_type is int:
                    data[f.name] = int(data[f.name])

        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


class Trainer:
    """DAGFormer Phase 1 training loop."""

    def __init__(self, config: TrainConfig, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.is_main = (local_rank == 0)
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

        # Gradient accumulation
        assert config.batch_size % config.micro_batch_size == 0, \
            f"batch_size ({config.batch_size}) must be divisible by micro_batch_size ({config.micro_batch_size})"
        self.accum_steps = config.batch_size // config.micro_batch_size

        self._build_models()
        self._build_optimizer()
        self._build_data()
        self._setup_logging()

        self.global_step = 0
        self.best_eval_nll = float("inf")
        self.collapse_counter = 0  # consecutive steps with collapsed A

        # Resume from checkpoint if specified
        if config.resume_from:
            state = load_checkpoint(
                config.resume_from,
                self.predictor,
                self.optimizer,
                self.lr_scheduler,
                device=self.device,
            )
            self.global_step = state["step"]
            self.best_eval_nll = state["best_eval_nll"]
            if self.is_main:
                print(f"Resumed from step {self.global_step}")

    def _build_models(self) -> None:
        config = self.config

        # Load frozen OLMo2-1B
        if self.is_main:
            print(f"Loading {config.olmo_model_id}...")
        self.olmo = AutoModelForCausalLM.from_pretrained(
            config.olmo_model_id,
            torch_dtype=torch.bfloat16,
        ).to(self.device)
        self.olmo.eval()
        for p in self.olmo.parameters():
            p.requires_grad_(False)

        # Verify frozen
        assert all(not p.requires_grad for p in self.olmo.parameters()), \
            "OLMo parameters should be frozen"

        # OLMo tokenizer
        self.olmo_tokenizer = AutoTokenizer.from_pretrained(config.olmo_model_id)

        # DAGFormer OLMo wrapper
        self.olmo_wrapper = DAGFormerOLMo(
            model=self.olmo,
            input_norm=config.input_norm,
        ).to(self.device)

        # Structure predictor (includes frozen Qwen + trainable MLP)
        if self.is_main:
            print(f"Loading {config.qwen_model_id}...")
        self.predictor = StructurePredictor(
            qwen_model_id=config.qwen_model_id,
            hidden_dim=config.predictor_hidden_dim,
            rank=config.predictor_rank,
            cascading_gate_k=config.cascading_gate_k,
            qwen_input_prefix=config.qwen_input_prefix,
            device=self.device,
        )

        # DDP wrapping — only the predictor MLP (trainable component)
        if self.world_size > 1:
            self.predictor.mlp = DDP(
                self.predictor.mlp,
                device_ids=[self.local_rank],
            )

        if self.is_main:
            trainable = sum(p.numel() for p in self.predictor.get_trainable_parameters())
            norm_params = sum(p.numel() for p in self.olmo_wrapper.input_normalizer.parameters())
            print(f"Trainable params: predictor={trainable:,}, norm={norm_params:,}")

    def _build_optimizer(self) -> None:
        config = self.config

        # Collect all trainable parameters
        params = list(self.predictor.get_trainable_parameters())
        params.extend(self.olmo_wrapper.input_normalizer.parameters())

        assert config.optimizer == "adamw", f"Only adamw supported, got {config.optimizer}"
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.lr,
            betas=(0.9, 0.999),
            weight_decay=config.weight_decay,
        )
        self.lr_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.total_steps,
            eta_min=0.0,
        )

    def _build_data(self) -> None:
        config = self.config

        self.train_loader = build_train_dataloader(
            olmo_tokenizer=self.olmo_tokenizer,
            seq_len=config.seq_len,
            batch_size=config.micro_batch_size,
            dataset_name=config.dataset,
            dataset_version=config.dataset_name,
            rank=self.local_rank,
            world_size=self.world_size,
        )

        # Eval data: only on main rank
        if self.is_main:
            cache_path = os.path.join(config.save_dir, "eval_cache.pt")
            self.eval_batches = build_eval_dataloader(
                olmo_tokenizer=self.olmo_tokenizer,
                seq_len=config.seq_len,
                batch_size=config.micro_batch_size,
                dataset_name=config.dataset,
                dataset_version=config.dataset_name,
                eval_skip=config.eval_skip,
                eval_size=config.eval_size,
                cache_path=cache_path,
            )
        else:
            self.eval_batches = []

    def _setup_logging(self) -> None:
        if self.is_main:
            self.wandb_run = init_wandb(
                project=self.config.wandb_project,
                run_name=self.config.wandb_run_name,
                config=self.config.to_dict(),
            )
        else:
            self.wandb_run = None

    def train(self) -> None:
        """Main training loop."""
        config = self.config
        train_iter = iter(self.train_loader)

        if self.is_main:
            print(f"\nStarting training: {config.total_steps} steps")
            print(f"  batch_size={config.batch_size}, micro_batch={config.micro_batch_size}, accum={self.accum_steps}")
            print(f"  tau: {config.tau_init} → {config.tau_final}")
            print(f"  lambda: 0 → {config.lambda_max}")
            print()

        while self.global_step < config.total_steps:
            # Schedule values
            tau = tau_schedule(self.global_step, config.total_steps, config.tau_init, config.tau_final)
            lam = lambda_schedule(self.global_step, config.total_steps, config.lambda_max, config.lambda_warmup_frac)

            # Gradient accumulation
            self.optimizer.zero_grad()
            total_nll = 0.0
            total_sparsity = 0.0
            total_mean_A = 0.0

            for micro_step in range(self.accum_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                olmo_ids = batch["olmo_ids"].to(self.device)
                olmo_labels = batch["olmo_labels"].to(self.device)
                raw_texts = batch["raw_text"]

                # Forward: predictor → A → OLMo → loss
                A = self.predictor(raw_texts, tau=tau, mode="train")
                logits = self.olmo_wrapper(olmo_ids, A)

                # NLL loss
                nll = F.cross_entropy(
                    logits[:, :-1].contiguous().view(-1, self.olmo.config.vocab_size),
                    olmo_labels[:, 1:].contiguous().view(-1),
                )

                # Sparsity loss
                sparsity = lam * A.mean()
                loss = (nll + sparsity) / self.accum_steps

                loss.backward()

                total_nll += nll.item() / self.accum_steps
                total_sparsity += sparsity.item() / self.accum_steps
                total_mean_A += A.mean().item() / self.accum_steps

            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

            # Logging
            if self.is_main and self.global_step % config.log_every == 0:
                # Gradient norm
                grad_norm = 0.0
                for p in self.predictor.get_trainable_parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                for p in self.olmo_wrapper.input_normalizer.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5

                metrics = {
                    "train/nll": total_nll,
                    "train/sparsity_loss": total_sparsity,
                    "train/total_loss": total_nll + total_sparsity,
                    "topology/mean_A": total_mean_A,
                    "schedule/tau": tau,
                    "schedule/lambda": lam,
                    "grad/predictor_norm": grad_norm,
                }
                log_metrics(metrics, self.global_step, self.wandb_run)

                # Collapse alarm
                if total_mean_A < 0.01 or total_mean_A > 0.99:
                    self.collapse_counter += 1
                    if self.collapse_counter >= 100:
                        warnings.warn(
                            f"COLLAPSE ALARM: mean_A={total_mean_A:.4f} for {self.collapse_counter} steps"
                        )
                else:
                    self.collapse_counter = 0

            # Eval
            if self.is_main and self.global_step > 0 and self.global_step % config.eval_every == 0:
                self._run_eval(tau)

            # Checkpoint
            if self.is_main and self.global_step > 0 and self.global_step % config.save_every == 0:
                save_checkpoint(
                    config.save_dir,
                    self.global_step,
                    self.predictor,
                    self.optimizer,
                    self.lr_scheduler,
                    self.best_eval_nll,
                )

            self.global_step += 1

            # Barrier for multi-GPU sync
            if self.world_size > 1:
                dist.barrier()

        # Final eval and checkpoint
        if self.is_main:
            self._run_eval(tau_schedule(config.total_steps, config.total_steps, config.tau_init, config.tau_final))
            save_checkpoint(
                config.save_dir,
                self.global_step,
                self.predictor,
                self.optimizer,
                self.lr_scheduler,
                self.best_eval_nll,
            )

        finish_wandb(self.wandb_run)
        if self.is_main:
            print("\nTraining complete.")

    @torch.no_grad()
    def _run_eval(self, tau: float) -> None:
        """Run evaluation on held-out data (rank 0 only).

        Reports: eval/nll_soft, eval/nll_hard, eval/nll_baseline
        """
        if not self.eval_batches:
            return

        self.predictor.eval()

        nll_soft_total = 0.0
        nll_hard_total = 0.0
        nll_baseline_total = 0.0
        n_batches = 0
        topology_metrics_accum: dict[str, float] = {}

        for batch in self.eval_batches:
            olmo_ids = batch["olmo_ids"].to(self.device)
            olmo_labels = batch["olmo_labels"].to(self.device)
            raw_texts = batch["raw_text"]

            vocab_size = self.olmo.config.vocab_size

            # Eval soft
            A_soft = self.predictor(raw_texts, tau=tau, mode="eval_soft")
            logits_soft = self.olmo_wrapper(olmo_ids, A_soft)
            nll_soft = F.cross_entropy(
                logits_soft[:, :-1].contiguous().view(-1, vocab_size),
                olmo_labels[:, 1:].contiguous().view(-1),
            )
            nll_soft_total += nll_soft.item()

            # Eval hard
            A_hard = self.predictor(raw_texts, tau=tau, mode="eval_hard")
            logits_hard = self.olmo_wrapper(olmo_ids, A_hard)
            nll_hard = F.cross_entropy(
                logits_hard[:, :-1].contiguous().view(-1, vocab_size),
                olmo_labels[:, 1:].contiguous().view(-1),
            )
            nll_hard_total += nll_hard.item()

            # Baseline (A=1)
            A_ones = create_all_ones_A(olmo_ids.shape[0]).to(self.device)
            logits_base = self.olmo_wrapper(olmo_ids, A_ones)
            nll_base = F.cross_entropy(
                logits_base[:, :-1].contiguous().view(-1, vocab_size),
                olmo_labels[:, 1:].contiguous().view(-1),
            )
            nll_baseline_total += nll_base.item()

            # Topology metrics (from soft A)
            topo = compute_topology_metrics(A_soft)
            for k, v in topo.items():
                topology_metrics_accum[k] = topology_metrics_accum.get(k, 0.0) + v

            n_batches += 1

        # Average
        metrics = {
            "eval/nll_soft": nll_soft_total / n_batches,
            "eval/nll_hard": nll_hard_total / n_batches,
            "eval/nll_baseline": nll_baseline_total / n_batches,
        }
        for k, v in topology_metrics_accum.items():
            metrics[k] = v / n_batches

        log_metrics(metrics, self.global_step, self.wandb_run)

        # Track best
        eval_nll = metrics["eval/nll_soft"]
        if eval_nll < self.best_eval_nll:
            self.best_eval_nll = eval_nll
            print(f"  New best eval NLL: {eval_nll:.4f}")

        self.predictor.train()
