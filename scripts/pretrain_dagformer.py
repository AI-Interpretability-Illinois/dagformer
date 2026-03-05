"""DAGFormer 300M pretraining: OLMo-2 from scratch with learned topology.

Two predictor variants:
  - "static": Learnable global topology (same A for all inputs)
  - "self_embed": Input-dependent A from model's own embeddings

Uses the same data pipeline and base model architecture as pretrain_baseline.py.
The base model trains jointly with the predictor (both from random init).

Usage:
    # Single GPU
    python scripts/pretrain_dagformer.py --config configs/pretrain_300m_static.yaml

    # Multi-GPU (4× A40)
    torchrun --nproc_per_node=4 scripts/pretrain_dagformer.py --config configs/pretrain_300m_static.yaml
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
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, Olmo2Config, Olmo2ForCausalLM

from src.data.dolma import build_eval_dataloader, build_train_dataloader
from src.model.olmo_graph import DAGFormerOLMo, create_all_ones_A
from src.model.predictor import (
    ContextEmbedPredictor, MiniEncoderPredictor, SelfEmbedPredictor,
    StaticPredictor, StructurePredictor,
)
from src.utils.logging import finish_wandb, init_wandb, log_metrics


# ─── CSV Logger (same as baseline) ───────────────────────────────────────────

class CSVLogger:
    """Append-mode CSV logger. Auto-extends columns when new keys appear."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.columns: list[str] = ["step"]

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

        write_header = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        if write_header:
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.columns)
                writer.writeheader()
                writer.writerow(row)
        else:
            with open(self.path) as f:
                existing_header = next(csv.reader(f), [])
            if set(self.columns) != set(existing_header):
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


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class DAGFormerPretrainConfig:
    """Configuration for DAGFormer pretraining (base model + predictor)."""

    # Model architecture (same as baseline)
    hidden_size: int = 1024
    num_hidden_layers: int = 12
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    vocab_size: int = 100352
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 4096

    # Tokenizer
    tokenizer_id: str = "allenai/OLMo-2-0425-1B"

    # Predictor
    predictor_type: str = "static"       # "static", "self_embed", or "qwen"
    predictor_hidden_dim: int = 1024     # MLP hidden dim (self_embed/qwen)
    predictor_rank: int = 32
    predictor_lr: float = 3e-4           # separate LR for predictor
    cascading_gate_k: float = 5.0
    init_logit: float = 15.0             # A≈1 at init (dense start)
    input_norm: str = "none"
    qwen_model_id: str = "Qwen/Qwen3-Embedding-0.6B"  # for qwen predictor
    predictor_encoder_dim: int = 256      # encoder dim for mini_encoder / context_embed
    predictor_encoder_layers: int = 2     # transformer layers in predictor encoder
    predictor_encoder_heads: int = 4      # attention heads in predictor encoder

    # Schedules
    tau_init: float = 5.0
    tau_final: float = 0.2
    lambda_max: float = 0.01
    lambda_warmup_frac: float = 0.2

    # Data
    dataset: str = "allenai/dolma"
    dataset_name: str = "v1_7"
    seq_len: int = 1024
    seed: int = 42

    # Training
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 16
    total_steps: int = 10000
    lr: float = 5e-4                     # base model LR
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_steps: int = 300
    max_grad_norm: float = 1.0
    lr_schedule: str = "linear"

    # Eval
    eval_skip: int = 1_000_000
    eval_size: int = 50
    eval_every: int = 500

    # Logging
    wandb_project: str = "dagformer"
    wandb_run_name: str = "pretrain-300m-dagformer"
    log_every: int = 10

    # Staged / alternating training
    baseline_checkpoint: str = ""        # path to baseline checkpoint to init base model
    standard_steps_per_dag_step: int = 0 # 0 = always DAGFormer; N = N standard then 1 DAGFormer
    detach_predictor_input: bool = True  # False = let gradients flow from predictor into embedding

    # Checkpointing
    save_every: int = 2000
    save_dir: str = "checkpoints/pretrain_300m_dagformer"
    resume_from: str = ""

    @classmethod
    def from_yaml(cls, path: str) -> DAGFormerPretrainConfig:
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)

        known_keys = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - known_keys
        if unknown:
            raise ValueError(f"Unknown config keys: {unknown}")

        for f in fields(cls):
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


# ─── Model creation ─────────────────────────────────────────────────────────

def create_model(config: DAGFormerPretrainConfig) -> Olmo2ForCausalLM:
    """Create randomly initialized OLMo-2 model from config."""
    model_config = Olmo2Config(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        vocab_size=config.vocab_size,
        tie_word_embeddings=config.tie_word_embeddings,
        max_position_embeddings=config.max_position_embeddings,
    )
    model = Olmo2ForCausalLM(model_config)

    seen_data_ptrs: set[int] = set()
    unique_params = 0
    total_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.data_ptr() not in seen_data_ptrs:
            seen_data_ptrs.add(p.data_ptr())
            unique_params += p.numel()

    head_dim = config.hidden_size // config.num_attention_heads
    print(f"Base model: {unique_params:,} unique params ({total_params:,} total)")
    print(f"  hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"heads={config.num_attention_heads}, head_dim={head_dim}")
    if config.tie_word_embeddings:
        print(f"  weight tying: ON (saves {config.vocab_size * config.hidden_size:,} params)")

    return model


def create_predictor(
    config: DAGFormerPretrainConfig,
) -> nn.Module:
    """Create predictor (static or self_embed) from config."""
    num_nodes = config.num_hidden_layers * config.num_attention_heads
    heads_per_layer = config.num_attention_heads

    if config.predictor_type == "static":
        predictor = StaticPredictor(
            num_nodes=num_nodes,
            heads_per_layer=heads_per_layer,
            rank=config.predictor_rank,
            cascading_gate_k=config.cascading_gate_k,
            init_logit=config.init_logit,
        )
    elif config.predictor_type == "self_embed":
        predictor = SelfEmbedPredictor(
            embed_dim=config.hidden_size,
            hidden_dim=config.predictor_hidden_dim,
            num_nodes=num_nodes,
            heads_per_layer=heads_per_layer,
            rank=config.predictor_rank,
            cascading_gate_k=config.cascading_gate_k,
            init_logit=config.init_logit,
        )
    elif config.predictor_type == "mini_encoder":
        predictor = MiniEncoderPredictor(
            vocab_size=config.vocab_size,
            encoder_dim=config.predictor_encoder_dim,
            encoder_layers=config.predictor_encoder_layers,
            encoder_heads=config.predictor_encoder_heads,
            hidden_dim=config.predictor_hidden_dim,
            num_nodes=num_nodes,
            heads_per_layer=heads_per_layer,
            rank=config.predictor_rank,
            cascading_gate_k=config.cascading_gate_k,
            init_logit=config.init_logit,
        )
    elif config.predictor_type == "context_embed":
        predictor = ContextEmbedPredictor(
            embed_dim=config.hidden_size,
            encoder_dim=config.predictor_encoder_dim,
            encoder_layers=config.predictor_encoder_layers,
            encoder_heads=config.predictor_encoder_heads,
            hidden_dim=config.predictor_hidden_dim,
            num_nodes=num_nodes,
            heads_per_layer=heads_per_layer,
            rank=config.predictor_rank,
            cascading_gate_k=config.cascading_gate_k,
            init_logit=config.init_logit,
        )
    elif config.predictor_type == "qwen":
        predictor = StructurePredictor(
            qwen_model_id=config.qwen_model_id,
            hidden_dim=config.predictor_hidden_dim,
            rank=config.predictor_rank,
            cascading_gate_k=config.cascading_gate_k,
            init_logit=config.init_logit,
            num_nodes=num_nodes,
            heads_per_layer=heads_per_layer,
        )
    else:
        raise ValueError(f"Unknown predictor_type: {config.predictor_type}. "
                         f"Expected 'static', 'self_embed', 'mini_encoder', "
                         f"'context_embed', or 'qwen'.")

    pred_params = sum(p.numel() for p in predictor.parameters())
    trainable_params = sum(p.numel() for p in predictor.get_trainable_parameters())
    print(f"Predictor ({config.predictor_type}): {pred_params:,} total params, "
          f"{trainable_params:,} trainable")
    return predictor


def predict_A(
    predictor: nn.Module,
    predictor_type: str,
    batch_size: int,
    tau: float,
    mode: str,
    embedding: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.Tensor] = None,
    raw_texts: Optional[list[str]] = None,
    detach: bool = True,
) -> torch.Tensor:
    """Dispatch to the right predictor forward based on type."""
    if predictor_type == "static":
        return predictor(batch_size, tau, mode=mode)
    elif predictor_type == "self_embed":
        assert embedding is not None
        e = embedding.detach() if detach else embedding
        return predictor(e, tau, mode=mode)
    elif predictor_type == "context_embed":
        assert embedding is not None
        e = embedding.detach() if detach else embedding
        return predictor(e, tau, mode=mode)
    elif predictor_type == "mini_encoder":
        assert input_ids is not None
        return predictor(input_ids, tau, mode=mode)
    elif predictor_type == "qwen":
        assert raw_texts is not None
        return predictor(raw_texts, tau, mode=mode)
    else:
        raise ValueError(f"Unknown predictor_type: {predictor_type}")


# ─── Schedules ───────────────────────────────────────────────────────────────

def get_lr(step: int, config: DAGFormerPretrainConfig) -> float:
    """Compute learning rate with linear warmup + decay."""
    if step < config.warmup_steps:
        return config.lr * step / max(1, config.warmup_steps)

    progress = (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
    progress = min(progress, 1.0)

    if config.lr_schedule == "linear":
        return config.lr * max(0.0, 1.0 - progress)
    else:  # cosine
        return config.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def get_predictor_lr(step: int, config: DAGFormerPretrainConfig) -> float:
    """Compute predictor learning rate (same schedule shape, different base)."""
    if step < config.warmup_steps:
        return config.predictor_lr * step / max(1, config.warmup_steps)

    progress = (step - config.warmup_steps) / max(1, config.total_steps - config.warmup_steps)
    progress = min(progress, 1.0)

    if config.lr_schedule == "linear":
        return config.predictor_lr * max(0.0, 1.0 - progress)
    else:
        return config.predictor_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def get_tau(step: int, config: DAGFormerPretrainConfig) -> float:
    """Cosine annealing for Gumbel-Sigmoid temperature: τ_init → τ_final."""
    return config.tau_final + 0.5 * (config.tau_init - config.tau_final) * \
        (1 + math.cos(math.pi * step / max(1, config.total_steps)))


def get_lambda(step: int, config: DAGFormerPretrainConfig) -> float:
    """Linear ramp for sparsity coefficient: 0 → λ_max over first 20% of steps."""
    warmup_steps = int(config.lambda_warmup_frac * config.total_steps)
    if step >= warmup_steps:
        return config.lambda_max
    return config.lambda_max * step / max(1, warmup_steps)


# ─── Checkpointing ──────────────────────────────────────────────────────────

def save_checkpoint(
    save_dir: str,
    step: int,
    model: Olmo2ForCausalLM,
    predictor: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_eval_nll: float,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"checkpoint_step{step}.pt")

    # Save model state separately (large)
    model_path = path.replace(".pt", "_model.pt")
    torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)

    state = {
        "step": step,
        "predictor_state_dict": predictor.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_eval_nll": best_eval_nll,
        "model_state_path": model_path,
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
            s = int(m.group(1))
            if s > best_step:
                best_step = s
                best_path = f
    return best_path


# ─── Topology metrics ────────────────────────────────────────────────────────

def compute_topology_metrics(
    A: torch.Tensor,
    heads_per_layer: int = 16,
) -> dict[str, float]:
    """Compute topology metrics from adjacency matrix A.

    Args:
        A: [batch, num_nodes, num_nodes] — gate matrix
        heads_per_layer: heads per layer

    Returns:
        dict with topology metrics
    """
    num_nodes = A.shape[1]
    num_layers = num_nodes // heads_per_layer

    # Mean gate activation
    # Only count valid (cross-layer) entries
    layer_idx = torch.arange(num_nodes, device=A.device) // heads_per_layer
    valid_mask = (layer_idx.unsqueeze(1) < layer_idx.unsqueeze(0)).float()
    valid_count = valid_mask.sum().item()
    if valid_count > 0:
        mean_A = (A * valid_mask.unsqueeze(0)).sum().item() / (A.shape[0] * valid_count)
    else:
        mean_A = 0.0

    # Sequential (adjacent-layer) vs skip (hyperconnection) gate fractions
    adj_mask = ((layer_idx.unsqueeze(1) + 1) == layer_idx.unsqueeze(0)).float()
    skip_mask = valid_mask - adj_mask  # skip = valid but not adjacent

    adj_count = adj_mask.sum().item()
    skip_count = skip_mask.sum().item()

    if adj_count > 0:
        adj_gates = (A * adj_mask.unsqueeze(0)).view(A.shape[0], -1)
        seq_gate_frac = (adj_gates > 0.5).float().mean().item()
    else:
        seq_gate_frac = 0.0

    if skip_count > 0:
        skip_gates = (A * skip_mask.unsqueeze(0)).view(A.shape[0], -1)
        hyp_gate_frac = (skip_gates > 0.5).float().mean().item()
    else:
        hyp_gate_frac = 0.0

    return {
        "topology/mean_A": mean_A,
        "topology/seq_gate_frac": seq_gate_frac,
        "topology/hyp_gate_frac": hyp_gate_frac,
    }


# ─── Combined module for DDP ────────────────────────────────────────────────

class DAGFormerPretrainModule(nn.Module):
    """Wraps base model + predictor for DDP.

    DDP needs a single nn.Module to wrap. This combines both the base OLMo model
    and the predictor into one module so DDP can sync gradients for both.
    The DAGFormerOLMo wrapper is NOT an nn.Module member here — we use it
    as a functional helper to avoid double-wrapping the base model's parameters.
    """

    def __init__(
        self,
        base_model: Olmo2ForCausalLM,
        predictor: nn.Module,
    ):
        super().__init__()
        self.base_model = base_model
        self.predictor = predictor


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="DAGFormer 300M pretraining")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    config = DAGFormerPretrainConfig.from_yaml(args.config)

    # DDP setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = (local_rank == 0)

    # Build eval cache BEFORE DDP init to avoid NCCL timeout.
    # Skipping 1M docs in streaming Dolma takes ~30min, which exceeds
    # NCCL's default timeout. ALL ranks must reach init_process_group
    # roughly simultaneously, so we let rank 0 build and all ranks wait.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(config.save_dir, exist_ok=True)
    eval_cache_path = os.path.join(config.save_dir, "eval_cache.pt")
    if not os.path.exists(eval_cache_path):
        if is_main:
            print("Pre-building eval cache (before DDP init)...")
            _tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_id)
            build_eval_dataloader(
                olmo_tokenizer=_tokenizer,
                seq_len=config.seq_len,
                batch_size=config.micro_batch_size,
                dataset_name=config.dataset,
                dataset_version=config.dataset_name,
                eval_skip=config.eval_skip,
                eval_size=config.eval_size,
                cache_path=eval_cache_path,
            )
            del _tokenizer
            print("Eval cache ready.")
        else:
            # Non-main ranks: wait for rank 0 to create the cache file
            import time as _time
            print(f"[rank {local_rank}] Waiting for eval cache...")
            while not os.path.exists(eval_cache_path):
                _time.sleep(5)

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    torch.manual_seed(config.seed + local_rank)
    torch.cuda.manual_seed_all(config.seed + local_rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Compute effective batch info
    tokens_per_step = config.micro_batch_size * config.gradient_accumulation_steps * world_size * config.seq_len
    total_tokens = tokens_per_step * config.total_steps

    num_nodes = config.num_hidden_layers * config.num_attention_heads

    if is_main:
        print(f"{'=' * 60}")
        print(f"DAGFormer 300M Pretraining ({config.predictor_type})")
        print(f"{'=' * 60}")
        print(f"World size: {world_size}")
        print(f"Tokens/step: {tokens_per_step:,} "
              f"({config.micro_batch_size} × {config.gradient_accumulation_steps} accum "
              f"× {world_size} GPUs × {config.seq_len} seq)")
        print(f"Total tokens: {total_tokens / 1e9:.2f}B ({config.total_steps} steps)")
        print(f"A matrix: {num_nodes}×{num_nodes} ({config.num_hidden_layers}L × {config.num_attention_heads}H)")
        print()

    # Create base model
    base_model = create_model(config)
    base_model = base_model.to(device, dtype=torch.bfloat16)

    # Create predictor
    predictor = create_predictor(config)
    predictor = predictor.to(device)

    # Create DAGFormer wrapper (uses base_model's parameters, not a copy)
    dagformer = DAGFormerOLMo(
        model=base_model,
        input_norm=config.input_norm,
        num_layers=config.num_hidden_layers,
        num_heads=config.num_attention_heads,
    )

    # Combined module for DDP
    combined = DAGFormerPretrainModule(base_model, predictor)
    combined_raw = combined

    if world_size > 1:
        combined = DDP(combined, device_ids=[local_rank], find_unused_parameters=True)
        combined_raw = combined.module

    # Optimizer with two param groups (different LRs)
    # Deduplicate tied weights for base model
    base_params = list(combined_raw.base_model.parameters())
    predictor_params = list(combined_raw.predictor.get_trainable_parameters())

    if is_main:
        pred_param_count = sum(p.numel() for p in predictor_params)
        base_param_count = sum(p.numel() for p in base_params)
        print(f"Optimizer: base model {base_param_count:,} params (lr={config.lr}), "
              f"predictor {pred_param_count:,} params (lr={config.predictor_lr})")

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": config.lr},
            {"params": predictor_params, "lr": config.predictor_lr},
        ],
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )

    # Tokenizer
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

    # Eval data (rank 0 only — cache was built before DDP init)
    eval_batches: list[dict] = []
    if is_main:
        eval_batches = build_eval_dataloader(
            olmo_tokenizer=tokenizer,
            seq_len=config.seq_len,
            batch_size=config.micro_batch_size,
            dataset_name=config.dataset,
            dataset_version=config.dataset_name,
            eval_skip=config.eval_skip,
            eval_size=config.eval_size,
            cache_path=eval_cache_path,
        )

    # Load base model from baseline checkpoint (for staged/alternating training)
    # This only loads the base model weights; predictor starts fresh.
    # Optimizer is NOT restored (fresh optimizer for the new training phase).
    if config.baseline_checkpoint:
        if is_main:
            print(f"Loading base model from baseline checkpoint: {config.baseline_checkpoint}")
        bl_ckpt = torch.load(config.baseline_checkpoint, map_location=device)
        # Baseline checkpoints use "model_state_dict" key (from pretrain_baseline.py)
        combined_raw.base_model.load_state_dict(bl_ckpt["model_state_dict"])
        if is_main:
            bl_step = bl_ckpt.get("step", "?")
            bl_nll = bl_ckpt.get("best_eval_nll", "?")
            print(f"  Loaded baseline @ step {bl_step} (best_eval_nll={bl_nll})")
        del bl_ckpt

    # Resume from DAGFormer checkpoint (for continuing interrupted runs)
    global_step = 0
    best_eval_nll = float("inf")
    resume_path = config.resume_from or find_latest_checkpoint(config.save_dir)
    if resume_path:
        if is_main:
            print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)

        # Load predictor
        combined_raw.predictor.load_state_dict(ckpt["predictor_state_dict"])

        # Load base model
        if "model_state_path" in ckpt:
            model_state = torch.load(ckpt["model_state_path"], map_location=device)
            combined_raw.base_model.load_state_dict(model_state)
            del model_state
        elif "model_state_dict" in ckpt:
            combined_raw.base_model.load_state_dict(ckpt["model_state_dict"])

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

    # CSV logger
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
            save_checkpoint(
                config.save_dir, global_step,
                combined_raw.base_model, combined_raw.predictor,
                optimizer, best_eval_nll,
            )
        raise SystemExit(0)

    signal.signal(signal.SIGUSR1, save_on_signal)

    # ── Training loop ──
    train_iter = iter(train_loader)
    vocab_size = config.vocab_size
    accum_steps = config.gradient_accumulation_steps

    # Collapse alarm state
    collapse_counter = 0

    if is_main:
        if config.baseline_checkpoint:
            print(f"Mode: staged (initialized from baseline checkpoint)")
        if config.standard_steps_per_dag_step > 0:
            n = config.standard_steps_per_dag_step
            print(f"Mode: alternating ({n} standard + 1 DAGFormer, "
                  f"ratio={n}:{1})")
        if not config.detach_predictor_input:
            print(f"Mode: unfrozen embed (gradients flow through predictor input)")
        print(f"\nTraining starts at step {global_step}")
        print()

    t0 = time.time()
    combined.train()

    # Alternating mode: decide which steps use DAGFormer vs standard forward
    alt_n = config.standard_steps_per_dag_step  # 0 = always DAGFormer

    while global_step < config.total_steps:
        # Compute schedules
        base_lr = get_lr(global_step, config)
        pred_lr = get_predictor_lr(global_step, config)
        tau = get_tau(global_step, config)
        lambda_t = get_lambda(global_step, config)

        # Determine if this step uses DAGFormer or standard forward
        if alt_n > 0:
            # Cycle: N standard steps, then 1 DAGFormer step
            use_dagformer = (global_step % (alt_n + 1) == alt_n)
        else:
            use_dagformer = True

        # Set LRs for each param group
        optimizer.param_groups[0]["lr"] = base_lr   # base model
        optimizer.param_groups[1]["lr"] = pred_lr if use_dagformer else 0.0  # predictor

        # Gradient accumulation
        optimizer.zero_grad()
        accum_nll = 0.0
        accum_sparsity = 0.0
        accum_total = 0.0
        last_A: Optional[torch.Tensor] = None

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
                else combined.no_sync()

            with sync_ctx:
                if use_dagformer:
                    # DAGFormer step: per-head routing with predictor
                    embedding = combined_raw.base_model.model.embed_tokens(input_ids)

                    A = predict_A(
                        combined_raw.predictor, config.predictor_type,
                        batch_size=input_ids.shape[0], tau=tau, mode="train",
                        embedding=embedding, input_ids=input_ids,
                        raw_texts=batch["raw_text"],
                        detach=config.detach_predictor_input,
                    )

                    logits = dagformer(input_ids, A)

                    nll = F.cross_entropy(
                        logits.contiguous().view(-1, vocab_size),
                        labels.contiguous().view(-1),
                    )
                    sparsity_loss = lambda_t * A.mean()
                    total_loss = (nll + sparsity_loss) / accum_steps
                    total_loss.backward()
                    last_A = A.detach()
                else:
                    # Standard step: fast dense forward (no A, no predictor)
                    outputs = combined_raw.base_model(input_ids=input_ids)
                    nll = F.cross_entropy(
                        outputs.logits.contiguous().view(-1, vocab_size),
                        labels.contiguous().view(-1),
                    )
                    sparsity_loss = torch.tensor(0.0)
                    total_loss = nll / accum_steps
                    total_loss.backward()

            accum_nll += nll.item() / accum_steps
            accum_sparsity += sparsity_loss.item() / accum_steps
            accum_total += total_loss.item()

        # Gradient clipping (over all params)
        if config.max_grad_norm > 0:
            all_params = list(combined_raw.base_model.parameters()) + \
                list(combined_raw.predictor.parameters())
            torch.nn.utils.clip_grad_norm_(all_params, config.max_grad_norm)

        # Step
        optimizer.step()

        # Logging
        if is_main and global_step % config.log_every == 0:
            elapsed = time.time() - t0
            tokens_seen = (global_step + 1) * tokens_per_step
            tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0

            # Gradient norms
            base_grad_norm = 0.0
            for p in combined_raw.base_model.parameters():
                if p.grad is not None:
                    base_grad_norm += p.grad.data.norm(2).item() ** 2
            base_grad_norm = base_grad_norm ** 0.5

            pred_grad_norm = 0.0
            for p in combined_raw.predictor.parameters():
                if p.grad is not None:
                    pred_grad_norm += p.grad.data.norm(2).item() ** 2
            pred_grad_norm = pred_grad_norm ** 0.5

            metrics: dict[str, float] = {
                "train/nll": accum_nll,
                "train/sparsity_loss": accum_sparsity,
                "train/total_loss": accum_total,
                "train/lr": base_lr,
                "train/predictor_lr": pred_lr,
                "train/tokens_per_sec": tok_per_sec,
                "train/tokens_seen_B": tokens_seen / 1e9,
                "grad/base_norm": base_grad_norm,
                "grad/predictor_norm": pred_grad_norm,
                "schedule/tau": tau,
                "schedule/lambda": lambda_t,
                "train/use_dagformer": 1.0 if use_dagformer else 0.0,
            }

            # Topology metrics from last micro-batch's A
            if last_A is not None:
                topo = compute_topology_metrics(last_A, config.num_attention_heads)
                metrics.update(topo)

                # Collapse alarm
                mean_A = topo["topology/mean_A"]
                if mean_A < 0.01 or mean_A > 0.99:
                    collapse_counter += 1
                    if collapse_counter >= 100:
                        print(f"WARNING: topology collapse detected! mean_A={mean_A:.4f} "
                              f"for {collapse_counter} consecutive log steps")
                else:
                    collapse_counter = 0

            log_metrics(metrics, global_step, wandb_run)
            if csv_logger is not None:
                csv_logger.log(global_step, metrics)

        # Eval (rank 0 only)
        if is_main and global_step > 0 and global_step % config.eval_every == 0:
            combined.eval()
            eval_nll_soft_total = 0.0
            eval_nll_hard_total = 0.0
            eval_nll_baseline_total = 0.0
            n_eval = 0

            with torch.no_grad():
                for eb in eval_batches:
                    eids = eb["olmo_ids"].to(device)
                    elabels = eb["olmo_labels"].to(device)

                    # Soft eval
                    embedding_eval = combined_raw.base_model.model.embed_tokens(eids)
                    eval_raw = eb.get("raw_text")
                    A_soft = predict_A(
                        combined_raw.predictor, config.predictor_type,
                        batch_size=eids.shape[0], tau=tau, mode="eval_soft",
                        embedding=embedding_eval, input_ids=eids,
                        raw_texts=eval_raw,
                    )
                    logits_soft = dagformer(eids, A_soft)
                    nll_soft = F.cross_entropy(
                        logits_soft.contiguous().view(-1, vocab_size),
                        elabels.contiguous().view(-1),
                    )
                    eval_nll_soft_total += nll_soft.item()

                    # Hard eval
                    A_hard = predict_A(
                        combined_raw.predictor, config.predictor_type,
                        batch_size=eids.shape[0], tau=tau, mode="eval_hard",
                        embedding=embedding_eval, input_ids=eids,
                        raw_texts=eval_raw,
                    )
                    logits_hard = dagformer(eids, A_hard)
                    nll_hard = F.cross_entropy(
                        logits_hard.contiguous().view(-1, vocab_size),
                        elabels.contiguous().view(-1),
                    )
                    eval_nll_hard_total += nll_hard.item()

                    # Baseline (standard forward, no A)
                    eout_base = combined_raw.base_model(input_ids=eids)
                    nll_base = F.cross_entropy(
                        eout_base.logits.contiguous().view(-1, vocab_size),
                        elabels.contiguous().view(-1),
                    )
                    eval_nll_baseline_total += nll_base.item()

                    n_eval += 1

            eval_nll_soft = eval_nll_soft_total / max(n_eval, 1)
            eval_nll_hard = eval_nll_hard_total / max(n_eval, 1)
            eval_nll_baseline = eval_nll_baseline_total / max(n_eval, 1)

            eval_metrics = {
                "eval/nll_soft": eval_nll_soft,
                "eval/nll_hard": eval_nll_hard,
                "eval/nll_baseline": eval_nll_baseline,
            }
            log_metrics(eval_metrics, global_step, wandb_run)
            if csv_logger is not None:
                csv_logger.log(global_step, eval_metrics)

            print(f"  [eval @ step {global_step}] soft={eval_nll_soft:.4f} "
                  f"hard={eval_nll_hard:.4f} baseline={eval_nll_baseline:.4f}")

            if eval_nll_soft < best_eval_nll:
                best_eval_nll = eval_nll_soft
                print(f"  New best eval NLL (soft): {eval_nll_soft:.4f}")

            combined.train()

        # Checkpoint
        if is_main and global_step > 0 and global_step % config.save_every == 0:
            save_checkpoint(
                config.save_dir, global_step,
                combined_raw.base_model, combined_raw.predictor,
                optimizer, best_eval_nll,
            )

        global_step += 1

        if world_size > 1:
            dist.barrier()

    # ── Final eval & save ──
    if is_main:
        combined.eval()
        eval_nll_soft_total = 0.0
        eval_nll_hard_total = 0.0
        n_eval = 0
        with torch.no_grad():
            for eb in eval_batches:
                eids = eb["olmo_ids"].to(device)
                elabels = eb["olmo_labels"].to(device)

                embedding_eval = combined_raw.base_model.model.embed_tokens(eids)
                eval_raw = eb.get("raw_text")
                A_soft = predict_A(
                    combined_raw.predictor, config.predictor_type,
                    batch_size=eids.shape[0], tau=tau, mode="eval_soft",
                    embedding=embedding_eval, input_ids=eids,
                    raw_texts=eval_raw,
                )
                logits_soft = dagformer(eids, A_soft)
                nll_soft = F.cross_entropy(
                    logits_soft.contiguous().view(-1, vocab_size),
                    elabels.contiguous().view(-1),
                )
                eval_nll_soft_total += nll_soft.item()

                A_hard = predict_A(
                    combined_raw.predictor, config.predictor_type,
                    batch_size=eids.shape[0], tau=tau, mode="eval_hard",
                    embedding=embedding_eval, input_ids=eids,
                    raw_texts=eval_raw,
                )
                logits_hard = dagformer(eids, A_hard)
                nll_hard = F.cross_entropy(
                    logits_hard.contiguous().view(-1, vocab_size),
                    elabels.contiguous().view(-1),
                )
                eval_nll_hard_total += nll_hard.item()

                n_eval += 1

        final_nll_soft = eval_nll_soft_total / max(n_eval, 1)
        final_nll_hard = eval_nll_hard_total / max(n_eval, 1)
        print(f"\nFinal eval NLL: soft={final_nll_soft:.4f} hard={final_nll_hard:.4f}")

        save_checkpoint(
            config.save_dir, global_step,
            combined_raw.base_model, combined_raw.predictor,
            optimizer, best_eval_nll,
        )

    finish_wandb(wandb_run)

    if world_size > 1:
        dist.destroy_process_group()

    if is_main:
        print("Training complete.")


if __name__ == "__main__":
    main()
