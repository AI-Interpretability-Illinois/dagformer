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
from datetime import timedelta
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, Olmo2Config, Olmo2ForCausalLM

from src.data.dolma import build_eval_dataloader, build_train_dataloader
from src.data.mmap_dataset import build_mmap_train_dataloader
from src.model.olmo_graph import (
    DAGFormerOLMo, DynamicDenseHeadFormer, FourWayDAGFormer,
    LayerDWAGateFormer, create_all_ones_A,
)
from src.model.modular_routing import (
    build_modular_pair, column_group_penalty, edge_sparsity_penalty,
    edge_sparsity_stats, source_column_mass, threshold_routing,
)
from src.model.predictor import (
    ContextEmbedPredictor, FourWayAttentionBottleneckPredictor,
    FourWayPerLayerPredictor,
    FourWayPredictor, FourWayPositionalPredictor, FourWayStaticPredictor,
    MiniEncoderPredictor,
    PerTokenSeq2MatrixPredictor, SeqToMatrixPredictor, SelfEmbedPredictor,
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
    predictor_encoder_dim: int = 256      # encoder dim for mini_encoder / context_embed / seq2matrix
    predictor_encoder_layers: int = 2     # transformer layers in predictor encoder
    predictor_encoder_heads: int = 4      # attention heads in predictor encoder
    predictor_cross_attn_heads: int = 4   # cross-attention heads for seq2matrix
    predictor_max_seq_len: int = 4096     # max seq len for seq2matrix positional embedding

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
    # Data source: "stream" (HF streaming, legacy/fragile) or "mmap" (pretokenized
    # shards from scripts/pretokenize.py — zero network, exact resume, no stalls).
    data_source: str = "stream"
    mmap_index_path: str = ""            # index.json path (or its dir) for data_source=mmap
    data_block_size: int = 1024          # block-shuffle granularity for mmap
    data_num_workers: int = 2            # DataLoader workers for mmap prefetch

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
    lr_decay_steps: int = 0              # steps over which LR decays to 0; 0 => use total_steps

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
    freeze_base_model: bool = False      # True = predictor-only mode (base model frozen at random init)
    predictor_checkpoint: str = ""       # path to pretrained predictor checkpoint (loads predictor only)

    # Internal routing (no external predictor)
    routing_mode: str = ""               # "dynamic_head", "layer_dwa_gate", "fourway", "fourway_corrected"
    routing_rank: int = 16               # low-rank factorization rank for per-head routing
    routing_hidden: int = 256            # hidden dim for routing MLPs
    dwa_hidden: int = 256                # hidden dim for DWA MLPs (layer_dwa_gate)
    correction_hidden: int = 128         # hidden dim for local correction MLPs (fourway_corrected)
    fourway_hidden: int = 512            # hidden dim for FourWayPredictor trunk
    use_torch_compile: bool = False      # torch.compile the fourway forward for speed
    use_triton_kernel: bool = False      # use fused Triton kernel for routing+proj
    predictor_causal: bool = True        # causal mask in FourWayPredictor encoder
    fourway_predictor_variant: str = "encoder"  # "encoder", "per_layer" (independent encoder per routed layer), "static", or "attn_bottleneck"
    use_v_norm: bool = False             # add post-mix RMSNorm on V (symmetric with Q/K norm)
    freeze_predictor: bool = False       # freeze FourWayPredictor at identity init (local-only experiment)
    predictor_dropout: float = 0.0       # classical nn.Dropout inside FourWayPredictor encoder + trunk (regularizes input→α mapping)
    label_smoothing: float = 0.0         # cross_entropy label smoothing on TRAIN loss only (eval NLL uses 0)
    alpha_share_heads: bool = False      # DIAGNOSTIC: mean α over H dim then broadcast (reduces DoF 16x, for code-vs-arch test)
    route_q: bool = True                 # enable learned Q routing; if false force Q stream to identity
    route_k: bool = True                 # enable learned K routing; if false force K stream to identity
    route_v: bool = True                 # enable learned V routing; if false force V stream to identity
    route_r: bool = True                 # enable learned R routing; if false force R stream to identity
    correction_pool: str = "none"        # "none"=per-token, "mean"=seq-avg (reduces per-token memorization)
    freeze_predictor_embed: bool = False # freeze only predictor.embed + pos_embed (keep encoder/heads trainable)
    replace_rmsnorm: bool = False        # replace Olmo2RMSNorm (has f32 cast breaking gradients) with torch.nn.RMSNorm

    # Routing regularization (fourway modes)
    routing_l2_lambda: float = 0.0       # L2 decay on routing logits (deviation from init)
    routing_l1_lambda: float = 0.0       # L1 on routing logits (delayed sparsity)
    routing_l1_start_frac: float = 0.15  # fraction of training before L1 kicks in
    routing_l1_warmup_frac: float = 0.50 # fraction of training when L1 reaches full strength
    routing_clamp: float = 0.0           # clamp routing logits to [-clamp, clamp], 0=disabled
    routing_noise_std: float = 0.0       # add Gaussian noise to routing logits
    routing_noise_steps: int = 0         # stop noise after this many steps
    routing_dropout: float = 0.0         # fraction of routing weights reset to identity each step
    routing_entropy_lambda: float = 0.0  # entropy regularization on softmax'd routing weights
    routing_top_k: int = 0               # top-k routing (0=disabled, 2=keep top 2 sources)
    routing_temperature: float = 1.0     # divide routing weights by temperature before use
    routing_delayed_start: int = 0       # don't use routing for first N steps (pure dense)
    routing_normalize: str = "none"      # "none", "softmax", "sinkhorn", "row_col" — normalize routing weights
    routing_sinkhorn_iters: int = 20     # Sinkhorn-Knopp iterations for sinkhorn normalize
    routing_column_group_lambda: float = 0.0  # fourway_modular: group-lasso on source columns (module switch-off)
    # fourway_modular: sparsity regularisation on the connection matrix itself
    routing_sparsity_lambda: float = 0.0        # coefficient (0 = off)
    routing_sparsity_kind: str = "l1"           # l1 | sqrt (edge dropped for all tokens) | column (module switch-off)
    routing_sparsity_streams: str = "q,k,v,r,m,o"  # which reads are penalised
    routing_sparsity_hyper_only: bool = False   # leave the sequential edges of the vanilla transformer free
    routing_sparsity_start_frac: float = 0.0    # fraction of training before the penalty starts
    routing_sparsity_warmup_frac: float = 0.0   # ... and reaches full strength (linear ramp)
    routing_sparsity_eval_eps: float = 0.0      # >0: also eval with |alpha|<eps hard-zeroed (eval/nll_sparsified)

    # Checkpointing
    save_every: int = 2000
    save_dir: str = "checkpoints/pretrain_300m_dagformer"
    resume_from: str = ""
    # Keep at most this many most-recent checkpoints (older deleted after save).
    # 0 = keep all (legacy). Use 2-3 for long runs to bound disk usage.
    keep_last_n: int = 0

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


def replace_olmo_rmsnorm(model: nn.Module) -> int:
    """Replace all Olmo2RMSNorm modules with torch.nn.RMSNorm.

    Olmo2RMSNorm casts input to float32 internally, which breaks gradient
    flow when the input comes from mixed-precision computation (einsum etc).
    torch.nn.RMSNorm operates in native dtype → correct gradients.

    Returns: number of modules replaced.
    """
    count = 0
    for name, module in list(model.named_modules()):
        if type(module).__name__ == "Olmo2RMSNorm":
            hidden_size = module.weight.shape[0]
            eps = module.variance_epsilon
            new_norm = nn.RMSNorm(hidden_size, eps=eps).to(
                device=module.weight.device, dtype=module.weight.dtype
            )
            new_norm.weight = module.weight  # share the parameter (same nn.Parameter)
            # Navigate to parent and replace
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_norm)
            count += 1
    return count


# ─── Routing post-processing (shared between train and eval) ───────────────

def apply_deterministic_routing_transforms(
    rw: dict,
    config: "DAGFormerPretrainConfig",
    global_step: int,
) -> dict:
    """Apply deterministic routing post-processing that MUST be identical
    between train and eval paths to avoid distribution shift.

    Includes: share_heads, clamp, top_k, temperature, delayed_start, normalize.
    Excludes (train-only, intentional regularization):
        - routing_noise_std (Gaussian noise on logits)
        - routing_dropout (stochastic reset to identity)
    Excludes (not transforms, loss terms):
        - routing_l2_lambda, routing_l1_lambda, routing_entropy_lambda
    """
    # Diagnostic: collapse per-head α to layer-level by averaging over H dim
    # then broadcasting back. This cuts Q/K/V effective DoF by 16x (H=16).
    # R stream has no H dim so is untouched. Used to distinguish "too many
    # routing DoF" (code is fine, just too expressive) from "code bug".
    if getattr(config, "alpha_share_heads", False):
        for stream in ('q', 'k', 'v'):  # r has no head dim
            for i, α in enumerate(rw[stream]):
                # α shape: [B, T, H, L+1]
                avg = α.mean(dim=-2, keepdim=True)  # [B, T, 1, L+1]
                rw[stream][i] = avg.expand_as(α)     # [B, T, H, L+1]

    if config.routing_clamp > 0:
        for stream in ('q', 'k', 'v', 'r'):
            for i, α in enumerate(rw[stream]):
                rw[stream][i] = α.clamp(-config.routing_clamp, config.routing_clamp)

    # Top-k routing: keep only top k sources, zero rest, renormalize
    if config.routing_top_k > 0:
        for stream in ('q', 'k', 'v', 'r'):
            for i, α in enumerate(rw[stream]):
                k = min(config.routing_top_k, α.shape[-1])
                topk_vals, topk_idx = α.topk(k, dim=-1)
                sparse = torch.zeros_like(α)
                sparse.scatter_(-1, topk_idx, topk_vals)
                orig_sum = α.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                sparse_sum = sparse.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                rw[stream][i] = sparse * (orig_sum / sparse_sum)

    # Temperature scaling
    if config.routing_temperature != 1.0:
        for stream in ('q', 'k', 'v', 'r'):
            for i, α in enumerate(rw[stream]):
                rw[stream][i] = α / config.routing_temperature

    # Delayed start: use identity routing for first N steps
    if config.routing_delayed_start > 0 and global_step < config.routing_delayed_start:
        for stream in ('q', 'k', 'v', 'r'):
            for i, α in enumerate(rw[stream]):
                identity = torch.zeros_like(α)
                identity[..., -1] = 1.0
                rw[stream][i] = identity

    # Normalize routing weights
    if config.routing_normalize == "softmax":
        for stream in ('q', 'k', 'v', 'r'):
            for i, α in enumerate(rw[stream]):
                rw[stream][i] = F.softmax(α, dim=-1)
    elif config.routing_normalize == "softmax_rv":
        # Only normalize R and V streams (leave Q/K alone since they
        # have built-in q_norm/k_norm downstream).
        for stream in ('v', 'r'):
            for i, α in enumerate(rw[stream]):
                rw[stream][i] = F.softmax(α, dim=-1)
    elif config.routing_normalize == "sinkhorn":
        for stream in ('q', 'k', 'v', 'r'):
            for i, α in enumerate(rw[stream]):
                # Sinkhorn-Knopp: alternating row/col normalization
                a = α.exp()  # make positive
                for _ in range(config.routing_sinkhorn_iters):
                    a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                    a = a / a.sum(dim=-2, keepdim=True).clamp(min=1e-8)
                rw[stream][i] = a
    elif config.routing_normalize == "row_col":
        # mHC-lite inspired: row softmax + column rescaling to mean=1
        for stream in ('q', 'k', 'v', 'r'):
            for i, α in enumerate(rw[stream]):
                a = F.softmax(α, dim=-1)
                col_mean = a.mean(dim=-2, keepdim=True).clamp(min=1e-8)
                rw[stream][i] = a / col_mean  # rescale columns

    return rw


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


def predict_fourway_routing(
    predictor: nn.Module,
    base_model: Olmo2ForCausalLM,
    input_ids: torch.Tensor,
    config: DAGFormerPretrainConfig,
) -> dict[str, list[torch.Tensor]]:
    """Run the selected FourWay predictor variant.

    The bottleneck variant reads from a dense scout pass of the current base
    model under ``torch.no_grad()``. This keeps the predictor input
    task-aligned while avoiding a gradient path through the scout memory read.
    """
    if config.fourway_predictor_variant == "attn_bottleneck":
        with torch.no_grad():
            query_states = base_model.model.embed_tokens(input_ids)
            scout = base_model.model(
                inputs_embeds=query_states,
                use_cache=False,
                return_dict=True,
            )
            memory_states = scout.last_hidden_state
        return predictor(query_states=query_states, memory_states=memory_states)

    return predictor(input_ids)


def apply_fourway_stream_mask(
    rw: dict[str, list[torch.Tensor]],
    config: DAGFormerPretrainConfig,
) -> dict[str, list[torch.Tensor]]:
    """Force disabled FourWay streams to exact identity routing.

    This is used for clean QK-only / VR-only ablations. Disabled streams are
    replaced with constant identity tensors before any regularization terms are
    computed, so they do not receive gradients or contribute confounds.
    """
    stream_flags = {
        "q": config.route_q,
        "k": config.route_k,
        "v": config.route_v,
        "r": config.route_r,
    }
    for stream, enabled in stream_flags.items():
        if enabled:
            continue
        for i, α in enumerate(rw[stream]):
            identity = torch.zeros_like(α)
            identity[..., -1] = 1.0
            rw[stream][i] = identity
    return rw


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
    elif config.predictor_type == "seq2matrix":
        predictor = SeqToMatrixPredictor(
            vocab_size=config.vocab_size,
            encoder_dim=config.predictor_encoder_dim,
            encoder_layers=config.predictor_encoder_layers,
            encoder_heads=config.predictor_encoder_heads,
            cross_attn_heads=config.predictor_cross_attn_heads,
            max_seq_len=config.predictor_max_seq_len,
            num_nodes=num_nodes,
            heads_per_layer=heads_per_layer,
            rank=config.predictor_rank,
            cascading_gate_k=config.cascading_gate_k,
            init_logit=config.init_logit,
        )
    elif config.predictor_type == "per_token_seq2matrix":
        predictor = PerTokenSeq2MatrixPredictor(
            vocab_size=config.vocab_size,
            encoder_dim=config.predictor_encoder_dim,
            encoder_layers=config.predictor_encoder_layers,
            encoder_heads=config.predictor_encoder_heads,
            max_seq_len=config.predictor_max_seq_len,
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
                         f"'context_embed', 'seq2matrix', or 'qwen'.")

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
    elif predictor_type == "seq2matrix":
        assert input_ids is not None
        return predictor(input_ids, tau, mode=mode)
    elif predictor_type == "per_token_seq2matrix":
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

    decay_steps = config.lr_decay_steps if config.lr_decay_steps > 0 else config.total_steps
    progress = (step - config.warmup_steps) / max(1, decay_steps - config.warmup_steps)
    progress = min(progress, 1.0)

    if config.lr_schedule == "linear":
        return config.lr * max(0.0, 1.0 - progress)
    else:  # cosine
        return config.lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def get_predictor_lr(step: int, config: DAGFormerPretrainConfig) -> float:
    """Compute predictor learning rate (same schedule shape, different base)."""
    if step < config.warmup_steps:
        return config.predictor_lr * step / max(1, config.warmup_steps)

    decay_steps = config.lr_decay_steps if config.lr_decay_steps > 0 else config.total_steps
    progress = (step - config.warmup_steps) / max(1, decay_steps - config.warmup_steps)
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

def _atomic_torch_save(obj, path: str, **kwargs) -> None:
    """Write a torch checkpoint atomically: serialize to a .tmp file, fsync,
    then os.replace onto the final path (atomic on POSIX same-filesystem).
    Prevents a crash mid-save from leaving a truncated/corrupt checkpoint."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        torch.save(obj, f, **kwargs)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _strip_compile_prefix(state: dict) -> dict:
    """Drop the ``_orig_mod.`` prefixes torch.compile'd modules put on state-dict keys.

    Makes checkpoints independent of whether the run used ``use_torch_compile``
    (a compiled run can be resumed eagerly and vice versa), and lets the routing
    filter below recognise the ``olmo.`` base-model keys under a compiled wrapper.
    """
    return {k.replace("_orig_mod.", ""): v for k, v in state.items()}


def _uncompiled(module: nn.Module) -> nn.Module:
    """Return the module underneath a ``torch.compile`` wrapper (or the module itself)."""
    return getattr(module, "_orig_mod", module)


def save_checkpoint(
    save_dir: str,
    step: int,
    model: Olmo2ForCausalLM,
    predictor: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    best_eval_nll: float,
    routing_model: Optional[nn.Module] = None,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"checkpoint_step{step}.pt")

    # Save model state separately (large)
    model_path = path.replace(".pt", "_model.pt")
    _atomic_torch_save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)

    state: dict = {
        "step": step,
        "optimizer_state_dict": optimizer.state_dict(),
        "best_eval_nll": best_eval_nll,
        "model_state_path": model_path,
    }
    if predictor is not None:
        state["predictor_state_dict"] = _strip_compile_prefix(predictor.state_dict())
    if routing_model is not None:
        # Save only routing params (not base model, which is saved separately).
        # Strip the compile prefix first: under torch.compile the keys are
        # "_orig_mod.olmo...." and the filter used to miss them, silently adding a
        # second copy of the whole base model to every checkpoint.
        routing_state = {k: v for k, v in _strip_compile_prefix(routing_model.state_dict()).items()
                         if not k.startswith("olmo.")}
        state["routing_state_dict"] = routing_state
    _atomic_torch_save(state, path)
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


def cleanup_old_checkpoints(save_dir: str, keep_last_n: int) -> int:
    """Delete all but the keep_last_n most-recent checkpoint groups.

    Matches both 'checkpoint_step{N}.pt' (main) and '_model.pt' companion.
    Returns # of step groups deleted.
    """
    if keep_last_n <= 0 or not os.path.isdir(save_dir):
        return 0
    by_step: dict[int, list[str]] = {}
    for f in os.listdir(save_dir):
        m = re.search(r"checkpoint_step(\d+)(?:_model)?\.pt$", f)
        if not m:
            continue
        step = int(m.group(1))
        by_step.setdefault(step, []).append(os.path.join(save_dir, f))
    if len(by_step) <= keep_last_n:
        return 0
    sorted_steps = sorted(by_step.keys(), reverse=True)
    deleted = 0
    for step in sorted_steps[keep_last_n:]:
        for f in by_step[step]:
            try:
                os.remove(f)
            except OSError as e:
                print(f"  WARN: could not delete {f}: {e}")
        deleted += 1
    if deleted:
        print(f"Pruned {deleted} old checkpoint group(s); kept last {keep_last_n}")
    return deleted


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
    # Handle per-token A [B, T, N, N] by averaging over T first
    if A.dim() == 4:
        A = A.mean(dim=1)  # [B, T, N, N] → [B, N, N]

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

    DDP needs a single nn.Module to wrap AND its forward() must be called
    during training for gradient synchronization to work. Without calling
    forward(), DDP's reducer never calls prepare_for_backward() and
    gradient all-reduce may not fire — causing each GPU to train
    independently (confirmed by DDP CHECK: params DIVERGED at step 10).
    """

    def __init__(
        self,
        base_model,
        predictor: nn.Module,
    ):
        super().__init__()
        self.base_model = base_model
        self.predictor = predictor

    def forward(self, input_ids: torch.Tensor,
                routing_weights: dict | None = None):
        """Forward through base model.

        DDP traces the autograd graph from the output to find all "used"
        parameters. When routing_weights is passed, it was created by the
        predictor (outside this forward), so predictor params are reachable
        from the output through the rw tensors → DDP syncs all gradients.

        Args:
            input_ids: [B, T] token IDs
            routing_weights: pre-computed routing dict (from predictor)
        """
        if routing_weights is not None:
            # FourWay: use pre-computed routing
            return self.base_model(input_ids, routing_weights)
        # Standard forward (non-FourWay or routing_mode)
        return self.base_model(input_ids=input_ids)


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
    if config.eval_size <= 0:
        if is_main:
            print("eval_size=0, skipping eval set construction.")
    elif not os.path.exists(eval_cache_path):
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
        # 30-min PG timeout (vs ~10-min default) so a slow data/eval/save step
        # on one rank can't SIGABRT all ranks via NCCL collective timeout.
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
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
    if config.replace_rmsnorm:
        n = replace_olmo_rmsnorm(base_model)
        if is_main:
            print(f"Replaced {n} Olmo2RMSNorm → torch.nn.RMSNorm (fixes gradient bug)")
    base_model = base_model.to(device, dtype=torch.bfloat16)

    # ─── Routing mode ───
    use_routing_mode = config.routing_mode in ("dynamic_head", "layer_dwa_gate")
    use_modular = config.routing_mode in ("fourway_modular", "fourway_modular_corrected")
    use_fourway = config.routing_mode in ("fourway", "fourway_corrected") or use_modular
    routing_model = None
    fourway_model = None
    fourway_predictor = None
    predictor = None
    dagformer = None

    if use_modular:
        # Module-granular routing (src/model/modular_routing.py): sources are
        # attention / MLP block outputs, identity init is all-ones. The legacy
        # FourWay routing knobs assume a one-hot identity and are not supported.
        for knob, default in (("routing_clamp", 0.0), ("routing_top_k", 0),
                              ("routing_temperature", 1.0), ("routing_delayed_start", 0),
                              ("routing_normalize", "none"), ("routing_dropout", 0.0),
                              ("routing_l2_lambda", 0.0), ("routing_l1_lambda", 0.0),
                              ("routing_entropy_lambda", 0.0), ("alpha_share_heads", False),
                              ("routing_noise_std", 0.0)):
            assert getattr(config, knob) == default, (
                f"{knob} is not supported with routing_mode={config.routing_mode}")
        assert config.route_q and config.route_k and config.route_v and config.route_r
        assert config.fourway_predictor_variant == "encoder", config.fourway_predictor_variant
        fourway_model, fourway_predictor = build_modular_pair(config.to_dict(), base_model, device)
        if config.freeze_predictor:
            for p in fourway_predictor.parameters():
                p.requires_grad_(False)
            fourway_predictor.eval()
        if is_main:
            pred_params = sum(p.numel() for p in fourway_predictor.parameters())
            corr_params = sum(p.numel() for p in fourway_model.get_routing_parameters())
            print(f"Mode: {config.routing_mode} (module-granular per-token routing)")
            print(f"  Predictor [modular encoder]: {pred_params:,} params")
            if corr_params:
                print(f"  Correction MLPs / v_norms: {corr_params:,} params")
            if config.routing_column_group_lambda > 0:
                print(f"  column group penalty: lambda={config.routing_column_group_lambda}")
        if config.use_torch_compile:
            fourway_model = torch.compile(fourway_model)
            fourway_predictor = torch.compile(fourway_predictor)
    elif use_fourway:
        use_correction = (config.routing_mode == "fourway_corrected")
        fourway_model = FourWayDAGFormer(
            model=base_model,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            use_local_correction=use_correction,
            correction_hidden=config.correction_hidden,
            use_triton_kernel=config.use_triton_kernel,
            use_v_norm=config.use_v_norm,
            correction_pool=config.correction_pool,
        ).to(device)
        if config.fourway_predictor_variant == "attn_bottleneck":
            fourway_predictor = FourWayAttentionBottleneckPredictor(
                model_dim=config.hidden_size,
                encoder_dim=config.predictor_encoder_dim,
                attn_heads=config.predictor_encoder_heads,
                num_layers=config.num_hidden_layers,
                num_heads=config.num_attention_heads,
                hidden_dim=config.fourway_hidden,
                dropout=config.predictor_dropout,
            ).to(device)
        elif config.fourway_predictor_variant == "static":
            fourway_predictor = FourWayStaticPredictor(
                num_layers=config.num_hidden_layers,
                num_heads=config.num_attention_heads,
            ).to(device)
        elif config.fourway_predictor_variant == "pos_table":
            fourway_predictor = FourWayPositionalPredictor(
                max_seq_len=config.seq_len,
                num_layers=config.num_hidden_layers,
                num_heads=config.num_attention_heads,
            ).to(device)
        elif config.fourway_predictor_variant == "per_layer":
            fourway_predictor = FourWayPerLayerPredictor(
                vocab_size=config.vocab_size,
                encoder_dim=config.predictor_encoder_dim,
                encoder_layers=config.predictor_encoder_layers,
                encoder_heads=config.predictor_encoder_heads,
                max_seq_len=config.predictor_max_seq_len,
                num_layers=config.num_hidden_layers,
                num_heads=config.num_attention_heads,
                hidden_dim=config.fourway_hidden,
                causal=config.predictor_causal,
                dropout=config.predictor_dropout,
            ).to(device)
        elif config.fourway_predictor_variant == "encoder":
            fourway_predictor = FourWayPredictor(
                vocab_size=config.vocab_size,
                encoder_dim=config.predictor_encoder_dim,
                encoder_layers=config.predictor_encoder_layers,
                encoder_heads=config.predictor_encoder_heads,
                max_seq_len=config.predictor_max_seq_len,
                num_layers=config.num_hidden_layers,
                num_heads=config.num_attention_heads,
                hidden_dim=config.fourway_hidden,
                causal=config.predictor_causal,
                dropout=config.predictor_dropout,
            ).to(device)
        else:
            raise ValueError(
                f"Unknown fourway_predictor_variant: {config.fourway_predictor_variant}. "
                "Expected 'encoder', 'per_layer', 'static', 'pos_table', or 'attn_bottleneck'."
            )
        if config.freeze_predictor:
            # Freeze external predictor at identity. Only corrections learn.
            for p in fourway_predictor.parameters():
                p.requires_grad_(False)
            fourway_predictor.eval()
        elif config.freeze_predictor_embed and hasattr(fourway_predictor, "embed"):
            # Freeze only token + position embeddings (25M + 1M params).
            # Prevents memorization through the lookup table while keeping
            # encoder/trunk/heads trainable.
            for p in fourway_predictor.embed.parameters():
                p.requires_grad_(False)
            for p in fourway_predictor.pos_embed.parameters():
                p.requires_grad_(False)
        if is_main:
            pred_params = sum(p.numel() for p in fourway_predictor.parameters())
            corr_params = sum(p.numel() for p in fourway_model.get_routing_parameters())
            mode_str = "fourway_corrected" if use_correction else "fourway"
            print(f"Mode: {mode_str} (4-way per-head per-token)")
            print(f"  Predictor [{config.fourway_predictor_variant}]: {pred_params:,} params"
                  f"{' (FROZEN at identity)' if config.freeze_predictor else ''}")
            if use_correction:
                print(f"  Correction MLPs: {corr_params:,} params")
        if config.use_torch_compile:
            fourway_model = torch.compile(fourway_model)
            fourway_predictor = torch.compile(fourway_predictor)
            if is_main:
                print("  torch.compile enabled")

    elif config.routing_mode == "dynamic_head":
        routing_model = DynamicDenseHeadFormer(
            model=base_model,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            routing_rank=config.routing_rank,
            routing_hidden=config.routing_hidden,
        ).to(device)
        if is_main:
            routing_params = sum(p.numel() for p in routing_model.get_routing_parameters())
            print(f"Mode: dynamic_head (per-head per-token routing, {routing_params:,} routing params)")

    elif config.routing_mode == "layer_dwa_gate":
        routing_model = LayerDWAGateFormer(
            model=base_model,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
            dwa_hidden=config.dwa_hidden,
        ).to(device)
        if is_main:
            routing_params = sum(p.numel() for p in routing_model.get_routing_parameters())
            print(f"Mode: layer_dwa_gate (layer DWA + head gating, {routing_params:,} routing params)")

    else:
        # External predictor mode (existing behavior)
        predictor = create_predictor(config)
        predictor = predictor.to(device)
        dagformer = DAGFormerOLMo(
            model=base_model,
            input_norm=config.input_norm,
            num_layers=config.num_hidden_layers,
            num_heads=config.num_attention_heads,
        )

    # Freeze base model if predictor-only mode
    if config.freeze_base_model:
        for p in base_model.parameters():
            p.requires_grad_(False)
        if is_main:
            src = f"baseline {config.baseline_checkpoint}" if config.baseline_checkpoint else "random init"
            print(f"Mode: predictor_only (base model FROZEN from {src})")

    # ─── DDP wrapping ───
    if use_fourway:
        # Wrap fourway_model (contains base_model) and predictor separately
        combined = DAGFormerPretrainModule(fourway_model, fourway_predictor)
        combined_raw = combined
        if world_size > 1:
            combined = DDP(combined, device_ids=[local_rank], find_unused_parameters=True)
            combined_raw = combined.module
    elif use_routing_mode:
        combined = routing_model
        combined_raw = combined
        if world_size > 1:
            combined = DDP(combined, device_ids=[local_rank], find_unused_parameters=True)
            combined_raw = combined.module
    else:
        combined = DAGFormerPretrainModule(base_model, predictor)
        combined_raw = combined
        if world_size > 1:
            combined = DDP(combined, device_ids=[local_rank], find_unused_parameters=True)
            combined_raw = combined.module

    # ─── Optimizer ───
    if use_fourway:
        base_params = [p for p in base_model.parameters() if p.requires_grad]
        # Split predictor params: identity-init routing biases get NO weight decay.
        # Use 'layer_biases' (not 'bias') to avoid catching encoder/trunk biases
        # which should have normal weight decay. (Codex audit finding.)
        pred_bias_params = [p for n, p in fourway_predictor.named_parameters()
                            if 'layer_biases' in n and p.requires_grad]
        pred_other_params = [p for n, p in fourway_predictor.named_parameters()
                             if 'layer_biases' not in n and p.requires_grad]
        corr_params = list(fourway_model.get_routing_parameters())
        pred_other_params = pred_other_params + corr_params
        if is_main:
            bp_count = sum(p.numel() for p in base_params)
            pp_count = sum(p.numel() for p in pred_other_params) + sum(p.numel() for p in pred_bias_params)
            bias_count = sum(p.numel() for p in pred_bias_params)
            print(f"Optimizer: base {bp_count:,} (lr={config.lr}), "
                  f"predictor {pp_count:,} (lr={config.predictor_lr}), "
                  f"bias params {bias_count:,} (wd=0)")
        # Build param groups, skipping empty ones (handles freeze_predictor where
        # pred_bias_params/pred_other_params may be empty).
        param_groups = []
        if not config.freeze_base_model and len(base_params) > 0:
            param_groups.append({"params": base_params, "lr": config.lr,
                                 "weight_decay": config.weight_decay})
        if len(pred_other_params) > 0:
            param_groups.append({"params": pred_other_params, "lr": config.predictor_lr,
                                 "weight_decay": config.weight_decay})
        if len(pred_bias_params) > 0:
            param_groups.append({"params": pred_bias_params, "lr": config.predictor_lr,
                                 "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(param_groups, betas=(config.beta1, config.beta2))
    elif use_routing_mode:
        base_params = [p for p in base_model.parameters() if p.requires_grad]
        routing_params = list(combined_raw.get_routing_parameters())
        if is_main:
            bp_count = sum(p.numel() for p in base_params)
            rp_count = sum(p.numel() for p in routing_params)
            print(f"Optimizer: base {bp_count:,} (lr={config.lr}), "
                  f"routing {rp_count:,} (lr={config.predictor_lr})")
        optimizer = torch.optim.AdamW(
            [
                {"params": base_params, "lr": config.lr},
                {"params": routing_params, "lr": config.predictor_lr},
            ],
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
    elif config.freeze_base_model:
        predictor_params = list(combined_raw.predictor.get_trainable_parameters())
        if is_main:
            pred_param_count = sum(p.numel() for p in predictor_params)
            print(f"Optimizer: predictor only, {pred_param_count:,} params (lr={config.predictor_lr})")
        optimizer = torch.optim.AdamW(
            [{"params": predictor_params, "lr": config.predictor_lr}],
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )
    else:
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
    if config.data_source == "mmap":
        assert config.mmap_index_path, "data_source=mmap requires mmap_index_path"
        train_loader = build_mmap_train_dataloader(
            index_path=config.mmap_index_path,
            seq_len=config.seq_len,
            batch_size=config.micro_batch_size,
            rank=local_rank,
            world_size=world_size,
            num_workers=config.data_num_workers,
            skip_samples=0,
            seed=config.seed,
            block_size=config.data_block_size,
        )
    else:
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
    if is_main and config.eval_size > 0:
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
        # FourWay wraps OLMo under .olmo, so we need to load into the inner model.
        if use_fourway:
            combined_raw.base_model.olmo.load_state_dict(bl_ckpt["model_state_dict"])
        else:
            combined_raw.base_model.load_state_dict(bl_ckpt["model_state_dict"])
        if is_main:
            bl_step = bl_ckpt.get("step", "?")
            bl_nll = bl_ckpt.get("best_eval_nll", "?")
            print(f"  Loaded baseline @ step {bl_step} (best_eval_nll={bl_nll})")
        del bl_ckpt

    # Load from predictor-only checkpoint (for two-stage training: predictor-first → unfreeze LLM)
    # Loads BOTH predictor AND base model weights (the predictor was trained for this specific LLM).
    # Optimizer and step counter are NOT restored (fresh start for phase 2).
    if config.predictor_checkpoint:
        if is_main:
            print(f"Loading predictor + base model from: {config.predictor_checkpoint}")
        pred_ckpt = torch.load(config.predictor_checkpoint, map_location=device)
        combined_raw.predictor.load_state_dict(pred_ckpt["predictor_state_dict"])
        # Load base model from the companion _model.pt file
        if "model_state_path" in pred_ckpt:
            model_state = torch.load(pred_ckpt["model_state_path"], map_location=device)
            combined_raw.base_model.load_state_dict(model_state)
            del model_state
        elif "model_state_dict" in pred_ckpt:
            combined_raw.base_model.load_state_dict(pred_ckpt["model_state_dict"])
        if is_main:
            pred_step = pred_ckpt.get("step", "?")
            print(f"  Loaded checkpoint @ step {pred_step} (fresh optimizer for phase 2)")
        del pred_ckpt

    # Resume from DAGFormer checkpoint (for continuing interrupted runs)
    global_step = 0
    best_eval_nll = float("inf")
    resume_path = config.resume_from or find_latest_checkpoint(config.save_dir)
    if resume_path:
        if is_main:
            print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)

        # Load predictor (prefix-agnostic: works for checkpoints written with or
        # without torch.compile, into a compiled or eager module)
        if "predictor_state_dict" in ckpt:
            _uncompiled(combined_raw.predictor).load_state_dict(
                _strip_compile_prefix(ckpt["predictor_state_dict"]))

        # Load base model — FourWay wraps OLMo under .olmo, need to load into inner model
        model_state = None
        if "model_state_path" in ckpt:
            model_state = torch.load(ckpt["model_state_path"], map_location=device)
        elif "model_state_dict" in ckpt:
            model_state = ckpt["model_state_dict"]

        if model_state is not None:
            if use_fourway:
                # FourWay: combined_raw.base_model is FourWayDAGFormer, real OLMo at .olmo
                _uncompiled(combined_raw.base_model).olmo.load_state_dict(model_state)
            else:
                combined_raw.base_model.load_state_dict(model_state)
            del model_state

        # Load routing state (correction MLPs, v_norms — may be empty for pure fourway)
        if use_fourway and "routing_state_dict" in ckpt and len(ckpt["routing_state_dict"]) > 0:
            m, u = _uncompiled(combined_raw.base_model).load_state_dict(
                _strip_compile_prefix(ckpt["routing_state_dict"]), strict=False
            )
            if is_main:
                print(f"  Routing state loaded: {len(ckpt['routing_state_dict'])} keys "
                      f"(missing={len(m)}, unexpected={len(u)})")

        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if is_main:
                print("  Optimizer state restored")
        except (ValueError, RuntimeError) as e:
            if is_main:
                print(f"  WARNING: could not restore optimizer state ({e})")
                print("  Continuing with fresh optimizer momentum (model weights OK)")
        global_step = ckpt["step"] + 1
        best_eval_nll = ckpt.get("best_eval_nll", float("inf"))
        del ckpt
        if is_main:
            print(f"Resumed at step {global_step}")

        # Rebuild dataloader to skip already-seen data.
        # Without this, the stream restarts from the beginning of Dolma
        # on every resume, causing the model to retrain on the same prefix.
        samples_seen = global_step * config.gradient_accumulation_steps * config.micro_batch_size
        if is_main:
            print(f"  Rebuilding dataloader: skipping {samples_seen} samples to avoid data repetition")
        if config.data_source == "mmap":
            # mmap uses a GLOBAL sample offset; samples_seen is per-rank.
            mmap_skip = samples_seen * world_size
            train_loader = build_mmap_train_dataloader(
                index_path=config.mmap_index_path,
                seq_len=config.seq_len,
                batch_size=config.micro_batch_size,
                rank=local_rank,
                world_size=world_size,
                num_workers=config.data_num_workers,
                skip_samples=mmap_skip,
                seed=config.seed,
                block_size=config.data_block_size,
            )
        else:
            train_loader = build_train_dataloader(
                olmo_tokenizer=tokenizer,
                seq_len=config.seq_len,
                batch_size=config.micro_batch_size,
                dataset_name=config.dataset,
                dataset_version=config.dataset_name,
                rank=local_rank,
                world_size=world_size,
                skip_samples=samples_seen,
            )

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
            if use_fourway:
                save_checkpoint(
                    config.save_dir, global_step,
                    base_model, fourway_predictor,
                    optimizer, best_eval_nll,
                    routing_model=fourway_model,
                )
            else:
                save_checkpoint(
                    config.save_dir, global_step,
                    base_model, predictor,
                    optimizer, best_eval_nll,
                    routing_model=routing_model if use_routing_mode else None,
                )
        raise SystemExit(0)

    signal.signal(signal.SIGUSR1, save_on_signal)
    # SLURM sends SIGTERM on scancel/timeout by default (SIGUSR1 only if the batch
    # script configures --signal=USR1@...); trap both so preemption always saves.
    signal.signal(signal.SIGTERM, save_on_signal)

    # ── Training loop ──
    train_iter = iter(train_loader)
    vocab_size = config.vocab_size
    accum_steps = config.gradient_accumulation_steps

    # Collapse alarm state
    collapse_counter = 0

    # Force always-DAGFormer when base model is frozen (no point in standard forward)
    if config.freeze_base_model and config.standard_steps_per_dag_step > 0:
        if is_main:
            print("Warning: freeze_base_model=True, forcing standard_steps_per_dag_step=0")
        config.standard_steps_per_dag_step = 0

    if is_main:
        if config.freeze_base_model:
            print(f"Mode: predictor_only (base model frozen, random init)")
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

    # Consecutive steps with a non-finite gradient norm (see the guard below).
    nonfinite_steps = 0
    last_column_mass: Optional[torch.Tensor] = None  # fourway_modular column stats
    last_edge_stats: Optional[dict] = None           # fourway_modular edge sparsity stats

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
        if use_fourway:
            if config.freeze_base_model:
                optimizer.param_groups[0]["lr"] = pred_lr  # only predictor+correction
                if len(optimizer.param_groups) > 1:
                    optimizer.param_groups[1]["lr"] = pred_lr  # routing biases
            else:
                optimizer.param_groups[0]["lr"] = base_lr   # base model
                optimizer.param_groups[1]["lr"] = pred_lr    # predictor+correction
                if len(optimizer.param_groups) > 2:
                    optimizer.param_groups[2]["lr"] = pred_lr  # routing biases
        elif use_routing_mode:
            optimizer.param_groups[0]["lr"] = base_lr   # base model
            optimizer.param_groups[1]["lr"] = pred_lr    # routing params
        elif config.freeze_base_model:
            optimizer.param_groups[0]["lr"] = pred_lr  # only predictor
        else:
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
                if use_fourway:
                    # 4-way per-head per-token routing.
                    # Predictor is called raw (outside DDP forward) to get rw,
                    # then rw is passed to combined(input_ids, rw) which goes
                    # through DDP forward. DDP traces the autograd graph from
                    # the output logits back through rw → predictor params,
                    # so ALL params are found as "used" and gradients are synced.
                    rw = predict_fourway_routing(
                        fourway_predictor, base_model, input_ids, config,
                    )

                    # Apply routing regularization
                    reg_loss = torch.tensor(0.0, device=device)
                    if config.routing_noise_std > 0 and global_step < config.routing_noise_steps:
                        for stream in ('q', 'k', 'v', 'r'):
                            for i, α in enumerate(rw[stream]):
                                rw[stream][i] = α + torch.randn_like(α) * config.routing_noise_std

                    if config.routing_dropout > 0:
                        # Reset random fraction of routing weights to identity
                        # Identity = [0,...,0,1] (last source = 1)
                        for stream in ('q', 'k', 'v', 'r'):
                            for i, α in enumerate(rw[stream]):
                                mask = torch.rand(α.shape[:-1], device=α.device) < config.routing_dropout
                                identity = torch.zeros_like(α)
                                identity[..., -1] = 1.0
                                rw[stream][i] = torch.where(mask.unsqueeze(-1), identity, α)

                    # Clean stream-ablation controls: disabled streams are
                    # pinned to exact identity before any loss terms.
                    rw = apply_fourway_stream_mask(rw, config)

                    if config.routing_l2_lambda > 0:
                        # L2 on deviation from identity (bias is identity, so deviation = dynamic part)
                        for stream in ('q', 'k', 'v', 'r'):
                            for α in rw[stream]:
                                # At identity init, last source = 1, rest = 0
                                # Deviation = sum of squares of all logits
                                # (W=0 init means the predictor's output heads contribute the deviation)
                                reg_loss = reg_loss + config.routing_l2_lambda * α.pow(2).mean()

                    if config.routing_l1_lambda > 0:
                        frac = global_step / max(config.total_steps, 1)
                        if frac >= config.routing_l1_start_frac:
                            ramp = min(1.0, (frac - config.routing_l1_start_frac) /
                                       max(config.routing_l1_warmup_frac - config.routing_l1_start_frac, 1e-8))
                            l1_coeff = config.routing_l1_lambda * ramp
                            for stream in ('q', 'k', 'v', 'r'):
                                for α in rw[stream]:
                                    reg_loss = reg_loss + l1_coeff * α.abs().mean()

                    # Entropy regularization: encourage uniform routing
                    if config.routing_entropy_lambda > 0:
                        for stream in ('q', 'k', 'v', 'r'):
                            for α in rw[stream]:
                                p = F.softmax(α, dim=-1)
                                ent = -(p * (p + 1e-8).log()).sum(dim=-1).mean()
                                reg_loss = reg_loss - config.routing_entropy_lambda * ent

                    # Modular routing: group-lasso on source columns so the
                    # predictor can switch whole modules off (see modular_routing.py).
                    if use_modular and config.routing_column_group_lambda > 0:
                        reg_loss = reg_loss + config.routing_column_group_lambda * \
                            column_group_penalty(rw, config.num_hidden_layers)
                    if use_modular and config.routing_sparsity_lambda > 0:
                        frac = global_step / max(config.total_steps, 1)
                        ramp = 0.0
                        if frac >= config.routing_sparsity_start_frac:
                            span = max(config.routing_sparsity_warmup_frac - config.routing_sparsity_start_frac, 1e-8)
                            ramp = min(1.0, (frac - config.routing_sparsity_start_frac) / span) \
                                if config.routing_sparsity_warmup_frac > config.routing_sparsity_start_frac else 1.0
                        sparsity_coeff = config.routing_sparsity_lambda * ramp
                        if sparsity_coeff > 0:
                            reg_loss = reg_loss + sparsity_coeff * edge_sparsity_penalty(
                                rw, config.num_hidden_layers, kind=config.routing_sparsity_kind,
                                streams=tuple(config.routing_sparsity_streams.split(",")),
                                hyper_only=config.routing_sparsity_hyper_only)
                    if use_modular and is_last_micro:
                        with torch.no_grad():
                            last_column_mass = source_column_mass(rw, config.num_hidden_layers)
                            last_edge_stats = edge_sparsity_stats(rw, config.num_hidden_layers,
                                                                  eps=config.routing_sparsity_eval_eps or 1e-2)

                    # Apply deterministic transforms (clamp, top_k, temperature,
                    # delayed_start, normalize). Must match eval path exactly.
                    rw = apply_deterministic_routing_transforms(rw, config, global_step)

                    # Call through DDP wrapper (combined) to trigger gradient sync.
                    # Previously called fourway_model directly, bypassing DDP —
                    # causing each GPU to train independently (params DIVERGED).
                    logits = combined(input_ids, rw)
                    nll = F.cross_entropy(
                        logits.contiguous().view(-1, vocab_size),
                        labels.contiguous().view(-1),
                        label_smoothing=config.label_smoothing,
                    )
                    sparsity_loss = reg_loss
                    total_loss = (nll + reg_loss) / accum_steps
                    total_loss.backward()
                elif use_routing_mode:
                    # Internal routing mode: model computes routing internally
                    logits = combined(input_ids)
                    nll = F.cross_entropy(
                        logits.contiguous().view(-1, vocab_size),
                        labels.contiguous().view(-1),
                        label_smoothing=config.label_smoothing,
                    )
                    sparsity_loss = torch.tensor(0.0)
                    total_loss = nll / accum_steps
                    total_loss.backward()
                elif use_dagformer:
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
                        label_smoothing=config.label_smoothing,
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
                        label_smoothing=config.label_smoothing,
                    )
                    sparsity_loss = torch.tensor(0.0)
                    total_loss = nll / accum_steps
                    total_loss.backward()

            accum_nll += nll.item() / accum_steps
            accum_sparsity += sparsity_loss.item() / accum_steps
            accum_total += total_loss.item()

        # Gradient clipping
        if config.max_grad_norm > 0:
            if use_fourway:
                all_params = [p for p in base_model.parameters() if p.requires_grad]
                all_params += list(fourway_predictor.parameters())
                all_params += list(fourway_model.get_routing_parameters())
            elif use_routing_mode:
                all_params = list(base_model.parameters()) + \
                    list(combined_raw.get_routing_parameters())
            elif config.freeze_base_model:
                all_params = list(combined_raw.predictor.parameters())
            else:
                all_params = list(combined_raw.base_model.parameters()) + \
                    list(combined_raw.predictor.parameters())
            total_norm = torch.nn.utils.clip_grad_norm_(all_params, config.max_grad_norm)
            if os.environ.get("DDP_DEBUG") and global_step <= 10:
                print(f"[DDP_DEBUG rank {local_rank}] step {global_step} total_norm={float(total_norm):.6f} "
                      f"n_params_with_grad={sum(1 for p in all_params if p.grad is not None)}/{len(all_params)}", flush=True)

            # NaN/Inf guard. One non-finite grad element is fatal AND permanent:
            # clip_grad_norm_ turns total_norm=inf into clip_coef=0, so inf*0=NaN
            # poisons EVERY parameter, and AdamW then writes NaN into its state
            # (exp_avg/exp_avg_sq) even at lr=0. Skipping the step keeps the
            # optimizer clean; a persistent problem aborts instead of burning
            # hours training on NaN (this happened: 13h of nan losses).
            # total_norm comes from DDP-synced grads → identical on every rank,
            # so all ranks take the same branch (no collective, no deadlock).
            if not torch.isfinite(total_norm):
                nonfinite_steps += 1
                optimizer.zero_grad(set_to_none=True)
                if is_main:
                    print(f"[NONFINITE] step {global_step}: grad norm {total_norm} — "
                          f"step skipped ({nonfinite_steps} consecutive)", flush=True)
                if nonfinite_steps >= 5:
                    raise RuntimeError(
                        f"Aborting at step {global_step}: {nonfinite_steps} consecutive "
                        f"non-finite gradient norms (model has diverged)."
                    )
                global_step += 1
                continue
            nonfinite_steps = 0

        # Step
        optimizer.step()

        # DDP sync check: verify all GPUs have identical parameters
        if world_size > 1 and global_step == 10:
            param = next(base_model.parameters())
            # Accumulate in float64. With bf16 params a bf16 sum + bf16 all-reduce
            # rounds (e.g. 3 x -38.75 = -116.25 is not representable in bf16 and
            # becomes -116.0), reporting a spurious "divergence" while every rank
            # holds bit-identical params. Seen 2026-09-25 on a 3-GPU 75M run.
            local_sum = param.data.detach().double().sum()
            param_sum = local_sum.clone()
            dist.all_reduce(param_sum, op=dist.ReduceOp.SUM)
            mean_val = param_sum.item() / world_size
            local_val = local_sum.item()
            diff = abs(local_val - mean_val)
            if os.environ.get("DDP_DEBUG"):
                print(f"[DDP CHECK rank {local_rank}] local_sum={local_val:.6f} mean={mean_val:.6f}", flush=True)
            if local_rank == 0:
                if diff < 1e-6:
                    print(f"[DDP CHECK @ step 10] PASS — params identical across {world_size} GPUs (diff={diff:.2e})")
                else:
                    print(f"[DDP CHECK @ step 10] FAIL — params DIVERGED! local={local_val:.6f} mean={mean_val:.6f} diff={diff:.2e}")
                    if not math.isfinite(local_val):
                        # NaN params, not a sync bug — the old message misattributed
                        # this as "DDP sync is broken", which hid a real divergence
                        # for two full runs.
                        print("  → params are NON-FINITE (NaN/Inf): the model has diverged, "
                              "not a DDP-sync problem.")
                    else:
                        print("  → DDP gradient sync is NOT working.")
            # All ranks abort together (every rank computed the same diff).
            if not (diff < 1e-6):
                raise RuntimeError(
                    f"DDP check failed at step 10: local={local_val} mean={mean_val} "
                    f"diff={diff:.2e} — aborting instead of training on bad state."
                )

        # Logging
        if is_main and global_step % config.log_every == 0:
            elapsed = time.time() - t0
            tokens_seen = (global_step + 1) * tokens_per_step
            tok_per_sec = tokens_seen / elapsed if elapsed > 0 else 0

            # Gradient norms
            base_grad_norm = 0.0
            for p in base_model.parameters():
                if p.grad is not None:
                    base_grad_norm += p.grad.data.norm(2).item() ** 2
            base_grad_norm = base_grad_norm ** 0.5

            pred_grad_norm = 0.0
            if use_fourway:
                routing_params_iter = list(fourway_predictor.parameters()) + \
                    list(fourway_model.get_routing_parameters())
            elif use_routing_mode:
                routing_params_iter = combined_raw.get_routing_parameters()
            else:
                routing_params_iter = combined_raw.predictor.parameters()
            for p in routing_params_iter:
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

            if last_column_mass is not None:
                cm = last_column_mass[1:]          # skip the embedding column
                metrics["routing/column_mass_mean"] = cm.mean().item()
                metrics["routing/column_mass_min"] = cm.min().item()
                metrics["routing/dead_sources"] = float((cm < 1e-3).sum().item())
                metrics["routing/column_mass_attn_mean"] = cm[0::2].mean().item()
                metrics["routing/column_mass_mlp_mean"] = cm[1::2].mean().item()
            if last_edge_stats is not None:
                metrics.update(last_edge_stats)
                metrics["schedule/routing_sparsity_lambda"] = (
                    config.routing_sparsity_lambda * (
                        1.0 if config.routing_sparsity_warmup_frac <= config.routing_sparsity_start_frac
                        else min(1.0, max(0.0, (global_step / max(config.total_steps, 1)
                                                - config.routing_sparsity_start_frac)
                                          / max(config.routing_sparsity_warmup_frac
                                                - config.routing_sparsity_start_frac, 1e-8))))
                    if global_step / max(config.total_steps, 1) >= config.routing_sparsity_start_frac else 0.0)

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

        # Eval (rank 0 only, but all ranks must sync at boundary)
        do_eval = global_step > 0 and global_step % config.eval_every == 0
        if do_eval and is_main:
            combined.eval()
            eval_nll_routing_total = 0.0
            eval_nll_baseline_total = 0.0
            eval_nll_sparsified_total = 0.0
            n_eval = 0

            with torch.no_grad():
                for eb in eval_batches:
                    eids = eb["olmo_ids"].to(device)
                    elabels = eb["olmo_labels"].to(device)

                    if use_fourway:
                        rw_eval = predict_fourway_routing(
                            fourway_predictor,
                            base_model,
                            eids,
                            config,
                        )
                        rw_eval = apply_fourway_stream_mask(rw_eval, config)
                        # CRITICAL: apply the SAME deterministic transforms as training
                        # (clamp, top_k, temperature, delayed_start, normalize).
                        # Without this, eval sees raw α but base model was trained on
                        # transformed α → distribution shift. See Codex audit Finding 1.
                        rw_eval = apply_deterministic_routing_transforms(
                            rw_eval, config, global_step
                        )
                        logits_r = fourway_model(eids, rw_eval)
                        nll_r = F.cross_entropy(
                            logits_r.contiguous().view(-1, vocab_size),
                            elabels.contiguous().view(-1),
                        )
                        eval_nll_routing_total += nll_r.item()
                        if use_modular and config.routing_sparsity_eval_eps > 0:
                            logits_s = fourway_model(eids, threshold_routing(rw_eval, config.routing_sparsity_eval_eps))
                            eval_nll_sparsified_total += F.cross_entropy(
                                logits_s.contiguous().view(-1, vocab_size),
                                elabels.contiguous().view(-1),
                            ).item()
                    elif use_routing_mode:
                        logits_r = combined_raw(eids)
                        nll_r = F.cross_entropy(
                            logits_r.contiguous().view(-1, vocab_size),
                            elabels.contiguous().view(-1),
                        )
                        eval_nll_routing_total += nll_r.item()
                    else:
                        # Predictor mode: soft eval
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
                        eval_nll_routing_total += nll_soft.item()

                    # Baseline (standard forward, no routing)
                    eout_base = base_model(input_ids=eids)
                    nll_base = F.cross_entropy(
                        eout_base.logits.contiguous().view(-1, vocab_size),
                        elabels.contiguous().view(-1),
                    )
                    eval_nll_baseline_total += nll_base.item()

                    n_eval += 1

            eval_nll_routing = eval_nll_routing_total / max(n_eval, 1)
            eval_nll_baseline = eval_nll_baseline_total / max(n_eval, 1)

            eval_metrics = {
                "eval/nll_soft": eval_nll_routing,
                "eval/nll_hard": eval_nll_routing,  # same for routing mode
                "eval/nll_baseline": eval_nll_baseline,
            }
            if use_modular and config.routing_sparsity_eval_eps > 0:
                eval_metrics["eval/nll_sparsified"] = eval_nll_sparsified_total / max(n_eval, 1)
            log_metrics(eval_metrics, global_step, wandb_run)
            if csv_logger is not None:
                csv_logger.log(global_step, eval_metrics)

            print(f"  [eval @ step {global_step}] routing={eval_nll_routing:.4f} "
                  f"baseline={eval_nll_baseline:.4f}")

            if eval_nll_routing < best_eval_nll:
                best_eval_nll = eval_nll_routing
                print(f"  New best eval NLL: {eval_nll_routing:.4f}")

            combined.train()

        # Barrier: all ranks wait for eval to finish (prevents NCCL timeout)
        if do_eval and world_size > 1:
            torch.distributed.barrier()

        # Checkpoint
        if is_main and global_step > 0 and global_step % config.save_every == 0:
            if use_fourway:
                save_checkpoint(
                    config.save_dir, global_step,
                    base_model, fourway_predictor,
                    optimizer, best_eval_nll,
                    routing_model=fourway_model,
                )
            else:
                save_checkpoint(
                    config.save_dir, global_step,
                    base_model, predictor,
                    optimizer, best_eval_nll,
                    routing_model=routing_model if use_routing_mode else None,
                )
            cleanup_old_checkpoints(config.save_dir, config.keep_last_n)

        global_step += 1

        if world_size > 1:
            dist.barrier()

    # ── Final eval & save ──
    if is_main:
        combined.eval()
        eval_nll_final_total = 0.0
        n_eval = 0
        with torch.no_grad():
            for eb in eval_batches:
                eids = eb["olmo_ids"].to(device)
                elabels = eb["olmo_labels"].to(device)

                if use_fourway:
                    rw_eval = predict_fourway_routing(
                        fourway_predictor,
                        base_model,
                        eids,
                        config,
                    )
                    rw_eval = apply_fourway_stream_mask(rw_eval, config)
                    # Same deterministic transforms as training (final eval)
                    rw_eval = apply_deterministic_routing_transforms(
                        rw_eval, config, global_step
                    )
                    logits_eval = fourway_model(eids, rw_eval)
                elif use_routing_mode:
                    logits_eval = combined_raw(eids)
                else:
                    embedding_eval = combined_raw.base_model.model.embed_tokens(eids)
                    eval_raw = eb.get("raw_text")
                    A_soft = predict_A(
                        combined_raw.predictor, config.predictor_type,
                        batch_size=eids.shape[0], tau=tau, mode="eval_soft",
                        embedding=embedding_eval, input_ids=eids,
                        raw_texts=eval_raw,
                    )
                    logits_eval = dagformer(eids, A_soft)

                nll_eval = F.cross_entropy(
                    logits_eval.contiguous().view(-1, vocab_size),
                    elabels.contiguous().view(-1),
                )
                eval_nll_final_total += nll_eval.item()
                n_eval += 1

        final_nll = eval_nll_final_total / max(n_eval, 1)
        print(f"\nFinal eval NLL: {final_nll:.4f}")

        if use_fourway:
            save_checkpoint(
                config.save_dir, global_step,
                base_model, fourway_predictor,
                optimizer, best_eval_nll,
                routing_model=fourway_model,
            )
        else:
            save_checkpoint(
                config.save_dir, global_step,
                base_model, predictor,
                optimizer, best_eval_nll,
                routing_model=routing_model if use_routing_mode else None,
            )

    finish_wandb(wandb_run)

    if world_size > 1:
        dist.destroy_process_group()

    if is_main:
        print("Training complete.")


if __name__ == "__main__":
    main()
