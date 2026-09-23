"""Prune-during-finetune: gradual structured pruning of a pretrained checkpoint
while finetuning it on a domain / task corpus.

Runs identically for a dense OLMo-2 baseline and a FourWay DAGFormer (the
model type is read off ``routing_mode`` in the checkpoint's config.yaml), so
the two can be compared at matched sparsity. The experiment's question: does a
model with learned cross-layer hyperconnections re-route around pruned units
during finetuning and therefore keep more domain performance at the same
parameter budget?

Pipeline (Zhu & Gupta 2017 schedule; Michel et al. 2019 / Molchanov et al. 2019
importance; CoFi-style unit set):

    finetune ──► at steps t0, t0+dt, ..., t1: raise the target sparsity along
                 a cubic curve, rank alive units by accumulated first-order
                 Taylor importance |dL/dgate|, prune the least important, keep
                 finetuning ──► masks fixed after t1, recovery phase ──► final
                 eval + checkpoint with the pruned weights zeroed in place

Unit types: attention heads, MLP neurons, whole attention blocks, whole MLP
blocks (any subset). Sparsity targets are per unit type (fraction of units);
parameter sparsity of the backbone is reported alongside.

Usage:
    python scripts/prune_finetune.py --config configs/prune/150m_dagformer_math.yaml
    torchrun --nproc_per_node=4 scripts/prune_finetune.py --config ... \
        --override total_steps=3000 --override target_sparsity=0.6
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import signal
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import timedelta
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.eval_lm_harness import (  # noqa: E402
    build_base_model, load_dense, load_fourway, _replace_olmo_rmsnorm,
)
from scripts.pretrain_dagformer import (  # noqa: E402
    CSVLogger, DAGFormerPretrainConfig, _atomic_torch_save,
    apply_deterministic_routing_transforms, apply_fourway_stream_mask,
)
from src.data.mmap_dataset import build_mmap_train_dataloader  # noqa: E402
from src.pruning import StructuredMasker, cubic_sparsity, prune_steps  # noqa: E402
from src.pruning.masks import IMPORTANCE_CRITERIA, UNIT_TYPES, count_unique_params  # noqa: E402
from src.model.modular_routing import (  # noqa: E402
    build_modular_pair, module_importance_from_mass, source_column_mass,
)
from src.utils.logging import finish_wandb, init_wandb, log_metrics  # noqa: E402


# ─── Config ──────────────────────────────────────────────────────────────────

@dataclass
class PruneFinetuneConfig:
    # Model: either a shared-checkpoint directory (config.yaml + checkpoint.pt)
    # or an explicit config + checkpoint. checkpoint="" with model_config set
    # means random init (smoke tests only).
    model_dir: str = ""
    model_config: str = ""
    checkpoint: str = ""

    # Data (built by scripts/pretokenize_domain.py)
    train_index_path: str = ""
    eval_cache_path: str = ""
    general_eval_cache_path: str = ""      # optional out-of-domain control (wikitext)
    seq_len: int = 1024
    seed: int = 42
    data_num_workers: int = 2
    data_block_size: int = 256

    # Finetuning
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    total_steps: int = 2000
    lr: float = 1e-4
    predictor_lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    lr_schedule: str = "linear"            # linear | cosine | constant
    lr_decay_steps: int = 0                # 0 => total_steps
    freeze_predictor: bool = False         # DAGFormer only: no re-routing allowed
    freeze_base: bool = False              # DAGFormer only: recovery through routing alone
    label_smoothing: float = 0.0

    # Pruning
    prune_units: list[str] = field(default_factory=lambda: ["head", "neuron"])
    target_sparsity: float = 0.5           # fraction of units per type
    target_sparsity_by_type: dict[str, float] = field(default_factory=dict)
    initial_sparsity: float = 0.0
    prune_start_step: int = 200
    prune_end_step: int = 1400
    prune_every: int = 100
    importance: str = "taylor"             # taylor | fisher | magnitude | random | routing_column (modular: attn/mlp scored by routing column mass, other units by taylor)
    min_alive_per_layer: dict[str, int] = field(default_factory=dict)
    eval_after_prune: bool = True          # measure damage before any recovery

    # Eval / logging
    eval_every: int = 200
    eval_max_batches: int = 0              # 0 = whole cache
    log_every: int = 10
    wandb_project: str = ""                # "" disables wandb
    wandb_run_name: str = "prune-finetune"

    # Checkpointing
    save_dir: str = "checkpoints/prune_finetune"
    save_every: int = 0                    # 0 = final only
    resume_from: str = ""

    @classmethod
    def from_yaml(cls, path: str, overrides: Optional[list[str]] = None) -> "PruneFinetuneConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        for ov in overrides or []:
            key, _, val = ov.partition("=")
            assert key and _ == "=", f"override must be key=value, got {ov!r}"
            data[key] = yaml.safe_load(val)
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        for f in fields(cls):
            if f.name in data and data[f.name] is not None:
                t = f.type
                if t in ("float", float):
                    data[f.name] = float(data[f.name])
                elif t in ("int", int):
                    data[f.name] = int(data[f.name])
                elif t in ("bool", bool):
                    data[f.name] = bool(data[f.name])
        cfg = cls(**data)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        assert self.model_dir or self.model_config, "set model_dir or model_config"
        assert self.train_index_path, "train_index_path is required"
        assert self.eval_cache_path, "eval_cache_path is required"
        for u in self.prune_units:
            assert u in UNIT_TYPES, f"unknown prune unit {u!r}"
        assert self.importance in IMPORTANCE_CRITERIA + ("routing_column",), self.importance
        assert 0.0 <= self.initial_sparsity <= self.target_sparsity <= 1.0
        assert 0 <= self.prune_start_step <= self.prune_end_step <= self.total_steps, (
            self.prune_start_step, self.prune_end_step, self.total_steps)
        assert self.prune_every >= 1
        assert self.lr_schedule in ("linear", "cosine", "constant")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    # resolved paths
    @property
    def model_config_path(self) -> str:
        return self.model_config or os.path.join(self.model_dir, "config.yaml")

    @property
    def checkpoint_path(self) -> str:
        if self.checkpoint:
            return self.checkpoint
        return os.path.join(self.model_dir, "checkpoint.pt") if self.model_dir else ""


def get_lr(step: int, base: float, cfg: PruneFinetuneConfig) -> float:
    if step < cfg.warmup_steps:
        return base * (step + 1) / cfg.warmup_steps
    if cfg.lr_schedule == "constant":
        return base
    decay_steps = cfg.lr_decay_steps or cfg.total_steps
    frac = min(1.0, (step - cfg.warmup_steps) / max(decay_steps - cfg.warmup_steps, 1))
    if cfg.lr_schedule == "linear":
        return base * (1.0 - frac)
    return base * 0.5 * (1.0 + math.cos(math.pi * frac))


# ─── Model ───────────────────────────────────────────────────────────────────

class PruneModule(nn.Module):
    """One nn.Module for DDP: dense OLMo or (predictor + routed backbone)."""

    def __init__(self, model_cfg: dict, base: nn.Module,
                 fourway: Optional[nn.Module], predictor: Optional[nn.Module]):
        super().__init__()
        self.is_fourway = fourway is not None
        self.is_modular = str(model_cfg.get("routing_mode", "")).startswith("fourway_modular")
        self.base = base                # Olmo2ForCausalLM (shared with fourway.olmo)
        self.fourway = fourway
        self.predictor = predictor
        self.num_layers = int(model_cfg["num_hidden_layers"])
        self.last_column_mass: Optional[torch.Tensor] = None   # modular: [2L+1] from the last forward
        self.pretrain_cfg = DAGFormerPretrainConfig(**{
            k: v for k, v in model_cfg.items()
            if k in DAGFormerPretrainConfig.__dataclass_fields__})
        self._step = 0

    def set_step(self, step: int) -> None:
        self._step = step

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if not self.is_fourway:
            return self.base(input_ids=input_ids).logits
        rw = self.predictor(input_ids)
        if self.is_modular:
            with torch.no_grad():
                self.last_column_mass = source_column_mass(rw, self.num_layers)
            return self.fourway(input_ids, rw)
        rw = apply_fourway_stream_mask(rw, self.pretrain_cfg)
        rw = apply_deterministic_routing_transforms(rw, self.pretrain_cfg, self._step)
        return self.fourway(input_ids, rw)

    @property
    def olmo(self) -> nn.Module:
        return self.base


def build_model(cfg: PruneFinetuneConfig, device: torch.device, is_main: bool) -> tuple[PruneModule, dict]:
    with open(cfg.model_config_path) as f:
        model_cfg = yaml.safe_load(f)
    is_fourway = str(model_cfg.get("routing_mode", "")).startswith("fourway")
    if is_fourway:
        variant = model_cfg.get("fourway_predictor_variant", "encoder")
        assert variant == "encoder", f"only the encoder predictor is supported here (got {variant})"
    ckpt = cfg.checkpoint_path

    if ckpt:
        if is_main:
            print(f"Loading {'fourway' if is_fourway else 'dense'} model from {ckpt}")
        if is_fourway:
            fourway, predictor = load_fourway(ckpt, model_cfg, device)
            base = fourway.olmo
        else:
            base = load_dense(ckpt, model_cfg, device)
            fourway = predictor = None
    else:
        if is_main:
            print("No checkpoint: RANDOM INIT (smoke test mode)")
        base = build_base_model(model_cfg, device)
        if model_cfg.get("replace_rmsnorm", False):
            _replace_olmo_rmsnorm(base)
            base = base.to(device=device, dtype=torch.bfloat16)
        fourway = predictor = None
        if str(model_cfg.get("routing_mode", "")).startswith("fourway_modular"):
            fourway, predictor = build_modular_pair(model_cfg, base, device)
        elif is_fourway:
            from src.model.olmo_graph import FourWayDAGFormer
            from src.model.predictor import FourWayPredictor
            fourway = FourWayDAGFormer(
                model=base, num_layers=model_cfg["num_hidden_layers"],
                num_heads=model_cfg["num_attention_heads"],
                use_local_correction=(model_cfg.get("routing_mode") == "fourway_corrected"),
                correction_hidden=model_cfg.get("correction_hidden", 128),
                use_v_norm=model_cfg.get("use_v_norm", False),
                correction_pool=model_cfg.get("correction_pool", "none"),
            ).to(device)
            predictor = FourWayPredictor(
                vocab_size=model_cfg["vocab_size"],
                encoder_dim=model_cfg.get("predictor_encoder_dim", 256),
                encoder_layers=model_cfg.get("predictor_encoder_layers", 2),
                encoder_heads=model_cfg.get("predictor_encoder_heads", 4),
                max_seq_len=model_cfg.get("predictor_max_seq_len", 4096),
                num_layers=model_cfg["num_hidden_layers"],
                num_heads=model_cfg["num_attention_heads"],
                hidden_dim=model_cfg.get("fourway_hidden", 512),
                causal=model_cfg.get("predictor_causal", True),
            ).to(device)

    module = PruneModule(model_cfg, base, fourway, predictor)
    # loaders freeze everything; re-enable per the finetuning config
    for p in base.parameters():
        p.requires_grad_(not (is_fourway and cfg.freeze_base))
    if is_fourway:
        for p in fourway.get_routing_parameters():
            p.requires_grad_(not cfg.freeze_base)
        for p in predictor.parameters():
            p.requires_grad_(not cfg.freeze_predictor)
    module.train()
    if is_fourway and cfg.freeze_predictor:
        predictor.eval()
    return module, model_cfg


def build_optimizer(module: PruneModule, cfg: PruneFinetuneConfig, is_main: bool) -> torch.optim.AdamW:
    groups = []
    base_params = [p for p in module.base.parameters() if p.requires_grad]
    if base_params:
        groups.append({"params": base_params, "lr": cfg.lr, "weight_decay": cfg.weight_decay,
                       "base_lr": cfg.lr})
    if module.is_fourway:
        routing = [p for p in module.fourway.get_routing_parameters() if p.requires_grad]
        pred_bias = [p for n, p in module.predictor.named_parameters()
                     if "layer_biases" in n and p.requires_grad]
        pred_other = [p for n, p in module.predictor.named_parameters()
                      if "layer_biases" not in n and p.requires_grad] + routing
        if pred_other:
            groups.append({"params": pred_other, "lr": cfg.predictor_lr,
                           "weight_decay": cfg.weight_decay, "base_lr": cfg.predictor_lr})
        if pred_bias:
            groups.append({"params": pred_bias, "lr": cfg.predictor_lr,
                           "weight_decay": 0.0, "base_lr": cfg.predictor_lr})
    assert groups, "nothing to train (freeze_base and freeze_predictor both set?)"
    if is_main:
        for g in groups:
            print(f"  optimizer group: {sum(p.numel() for p in g['params']):,} params, "
                  f"lr={g['base_lr']}, wd={g['weight_decay']}")
    return torch.optim.AdamW(groups, betas=(cfg.beta1, cfg.beta2))


# ─── Eval ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(module: PruneModule, batches: list[dict], device: torch.device,
             max_batches: int = 0) -> float:
    module.eval()
    total, n = 0.0, 0
    for i, eb in enumerate(batches):
        if max_batches and i >= max_batches:
            break
        ids = eb["olmo_ids"].to(device)
        labels = eb["olmo_labels"].to(device)
        logits = module(ids)
        nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
        total += nll.item()
        n += 1
    module.train()
    if module.is_fourway and not any(p.requires_grad for p in module.predictor.parameters()):
        module.predictor.eval()
    return total / max(n, 1)


def load_eval_cache(path: str) -> list[dict]:
    return torch.load(path, map_location="cpu", weights_only=False)


# ─── Checkpoint ──────────────────────────────────────────────────────────────

def save_checkpoint(out_dir: str, module: PruneModule, masker: StructuredMasker,
                    optimizer: Optional[torch.optim.Optimizer], step: int,
                    model_cfg: dict, cfg: PruneFinetuneConfig, final: bool) -> str:
    """Write a checkpoint in the same layout as the shared model directories
    (config.yaml + checkpoint.pt [+ *_model.pt side file]) plus masks.pt, so
    the pruned model can be picked up by the lm-eval harness unchanged."""
    os.makedirs(out_dir, exist_ok=True)
    state: dict = {"step": step, "masker_state": masker.state_dict(),
                   "prune_config": cfg.to_dict()}
    if module.is_fourway:
        side = os.path.join(out_dir, f"checkpoint_step{step}_model.pt")
        _atomic_torch_save(module.base.state_dict(), side, _use_new_zipfile_serialization=False)
        state["model_state_path"] = side
        state["predictor_state_dict"] = module.predictor.state_dict()
        state["routing_state_dict"] = {k: v for k, v in module.fourway.state_dict().items()
                                       if not k.startswith("olmo.")}
    else:
        state["model_state_dict"] = module.base.state_dict()
    if optimizer is not None and not final:
        state["optimizer_state_dict"] = optimizer.state_dict()
    path = os.path.join(out_dir, "checkpoint.pt")
    _atomic_torch_save(state, path)
    _atomic_torch_save(masker.state_dict(), os.path.join(out_dir, "masks.pt"))
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(model_cfg, f, sort_keys=False)
    print(f"Checkpoint saved: {path}")
    return path


@torch.no_grad()
def bake_masks_into_weights(masker: StructuredMasker) -> None:
    """Zero the weights of pruned units so the saved model is standalone: the
    plain HF / FourWay forward without hooks reproduces the masked model."""
    layers = masker.olmo.model.layers
    hd = masker.hd
    for l, layer in enumerate(layers):
        if "head" in masker.masks:
            for h in range(masker.H):
                if masker.masks["head"][l, h] == 0:
                    layer.self_attn.o_proj.weight[:, h * hd:(h + 1) * hd].zero_()
        if "neuron" in masker.masks:
            dead = (masker.masks["neuron"][l] == 0).nonzero().flatten()
            if dead.numel():
                layer.mlp.down_proj.weight[:, dead] = 0.0
        if "attn" in masker.masks and masker.masks["attn"][l, 0] == 0:
            layer.self_attn.o_proj.weight.zero_()
        if "mlp" in masker.masks and masker.masks["mlp"][l, 0] == 0:
            layer.mlp.down_proj.weight.zero_()


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Gradual structured pruning during finetuning")
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[], help="key=value (yaml-parsed)")
    args = parser.parse_args()
    cfg = PruneFinetuneConfig.from_yaml(args.config, args.override)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_main = local_rank == 0
    if world_size > 1:
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed + local_rank)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(cfg.save_dir, exist_ok=True)

    tokens_per_step = cfg.micro_batch_size * cfg.gradient_accumulation_steps * world_size * cfg.seq_len
    if is_main:
        print("=" * 60)
        print("Prune-during-finetune")
        print("=" * 60)
        print(f"config: {args.config}  overrides: {args.override}")
        print(f"world={world_size}  tokens/step={tokens_per_step:,}  "
              f"total={tokens_per_step * cfg.total_steps / 1e6:.1f}M tokens over {cfg.total_steps} steps")
        print(f"pruning {cfg.prune_units} -> {cfg.target_sparsity} "
              f"(by type: {cfg.target_sparsity_by_type}) from step {cfg.prune_start_step} "
              f"to {cfg.prune_end_step} every {cfg.prune_every}, importance={cfg.importance}")

    # Model + masks
    module, model_cfg = build_model(cfg, device, is_main)
    if cfg.importance == "routing_column":
        assert module.is_modular, "importance=routing_column needs routing_mode=fourway_modular*"
    masker = StructuredMasker(module.olmo, unit_types=cfg.prune_units, device=device)
    column_mass_sum: Optional[torch.Tensor] = None       # routing_column accumulator
    base_total = count_unique_params(module.base)
    if is_main:
        extra = 0
        if module.is_fourway:
            extra = sum(p.numel() for p in module.predictor.parameters()) + \
                sum(p.numel() for p in module.fourway.get_routing_parameters())
        print(f"Model: {'fourway' if module.is_fourway else 'dense'}  base params {base_total:,}"
              + (f"  + predictor/routing {extra:,}" if extra else ""))
        print(f"Prunable units: {masker.counts()}")

    optimizer = build_optimizer(module, cfg, is_main)

    ddp_model: nn.Module = module
    if world_size > 1:
        ddp_model = DDP(module, device_ids=[local_rank], find_unused_parameters=True)

    # Data
    global_step = 0
    resume_path = cfg.resume_from or (
        os.path.join(cfg.save_dir, "latest", "checkpoint.pt")
        if os.path.exists(os.path.join(cfg.save_dir, "latest", "checkpoint.pt")) else "")
    if resume_path:
        if is_main:
            print(f"Resuming from {resume_path}")
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        if module.is_fourway:
            side = ck["model_state_path"]
            if not os.path.exists(side):
                side = os.path.join(os.path.dirname(resume_path), os.path.basename(side))
            module.base.load_state_dict(torch.load(side, map_location="cpu", weights_only=False))
            module.predictor.load_state_dict(ck["predictor_state_dict"])
            module.fourway.load_state_dict(ck["routing_state_dict"], strict=False)
        else:
            module.base.load_state_dict(ck["model_state_dict"])
        masker.load_state_dict(ck["masker_state"])
        if "optimizer_state_dict" in ck:
            optimizer.load_state_dict(ck["optimizer_state_dict"])
        global_step = int(ck["step"]) + 1
        del ck

    train_loader = build_mmap_train_dataloader(
        index_path=cfg.train_index_path, seq_len=cfg.seq_len, batch_size=cfg.micro_batch_size,
        rank=local_rank, world_size=world_size, num_workers=cfg.data_num_workers,
        skip_samples=global_step * cfg.gradient_accumulation_steps * cfg.micro_batch_size * world_size,
        seed=cfg.seed, block_size=cfg.data_block_size,
    )
    train_iter = iter(train_loader)
    eval_batches = load_eval_cache(cfg.eval_cache_path) if is_main else []
    general_batches = (load_eval_cache(cfg.general_eval_cache_path)
                       if is_main and cfg.general_eval_cache_path else [])
    if is_main:
        n_train = train_loader.dataset.n_samples
        print(f"Train windows: {n_train:,} ({n_train * cfg.seq_len / 1e6:.1f}M tokens; "
              f"{tokens_per_step * cfg.total_steps / max(n_train * cfg.seq_len, 1):.2f} epochs)")
        print(f"Eval: {sum(b['olmo_ids'].shape[0] for b in eval_batches)} domain windows"
              + (f", {sum(b['olmo_ids'].shape[0] for b in general_batches)} general windows"
                 if general_batches else ""))

    # Logging
    wandb_run = init_wandb(cfg.wandb_project, cfg.wandb_run_name, cfg.to_dict()) \
        if (is_main and cfg.wandb_project) else None
    csv_logger = CSVLogger(os.path.join(cfg.save_dir, "metrics.csv")) if is_main else None
    trajectory_path = os.path.join(cfg.save_dir, "trajectory.json")
    trajectory: list[dict] = []
    if is_main and os.path.exists(trajectory_path) and global_step > 0:
        with open(trajectory_path) as f:
            trajectory = json.load(f)

    def record_eval(step: int, tag: str) -> dict[str, float]:
        """Rank-0 eval on domain (+ general) caches; appends to the trajectory."""
        m = {"eval/domain_nll": evaluate(module, eval_batches, device, cfg.eval_max_batches)}
        if general_batches:
            m["eval/general_nll"] = evaluate(module, general_batches, device, cfg.eval_max_batches)
        m.update(masker.report())
        m["eval/tag"] = {"periodic": 0.0, "post_prune": 1.0, "final": 2.0, "initial": 3.0}[tag]
        log_metrics(m, step, wandb_run)
        csv_logger.log(step, m)
        entry = {"step": step, "tag": tag, **{k: v for k, v in m.items() if k != "eval/tag"},
                 "per_layer_pruned": masker.per_layer_summary()}
        trajectory.append(entry)
        with open(trajectory_path, "w") as f:
            json.dump(trajectory, f, indent=1)
        print(f"  [eval:{tag} @ {step}] domain={m['eval/domain_nll']:.4f}"
              + (f" general={m['eval/general_nll']:.4f}" if general_batches else "")
              + f" block_sparsity={m['prune/block_param_sparsity']:.3f}"
              + f" base_sparsity={m['prune/base_param_sparsity']:.3f}", flush=True)
        return m

    # Preemption handling
    def save_latest() -> None:
        if is_main:
            save_checkpoint(os.path.join(cfg.save_dir, "latest"), module, masker, optimizer,
                            global_step, model_cfg, cfg, final=False)

    def on_signal(signum: int, frame: Any) -> None:
        print(f"Signal {signum}: saving and exiting", flush=True)
        save_latest()
        raise SystemExit(0)

    signal.signal(signal.SIGUSR1, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    schedule = set(prune_steps(cfg.prune_start_step, cfg.prune_end_step, cfg.prune_every))
    accumulate = cfg.importance in ("taylor", "fisher", "routing_column")
    unit_criterion = "taylor" if cfg.importance == "routing_column" else cfg.importance
    vocab = int(model_cfg["vocab_size"])
    accum = cfg.gradient_accumulation_steps

    if is_main and global_step == 0:
        record_eval(0, "initial")
    if world_size > 1:
        dist.barrier()

    t0 = time.time()
    while global_step < cfg.total_steps:
        module.set_step(global_step)
        for g in optimizer.param_groups:
            g["lr"] = get_lr(global_step, g["base_lr"], cfg)

        optimizer.zero_grad(set_to_none=True)
        accum_nll = 0.0
        for micro in range(accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            ids = batch["olmo_ids"].to(device, non_blocking=True)
            labels = batch["olmo_labels"].to(device, non_blocking=True)
            last = micro == accum - 1
            ctx = contextlib.nullcontext() if (last or world_size == 1) else ddp_model.no_sync()
            with ctx:
                logits = ddp_model(ids)
                nll = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1),
                                      label_smoothing=cfg.label_smoothing)
                (nll / accum).backward()
            accum_nll += nll.item() / accum

        if accumulate:
            masker.accumulate_importance(unit_criterion)
        if cfg.importance == "routing_column":
            cm = module.last_column_mass.detach()
            column_mass_sum = cm.clone() if column_mass_sum is None else column_mass_sum + cm

        params = [p for g in optimizer.param_groups for p in g["params"]]
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm) \
            if cfg.max_grad_norm > 0 else torch.tensor(0.0)
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            if is_main:
                print(f"[NONFINITE] step {global_step}: grad norm {grad_norm}, step skipped", flush=True)
            global_step += 1
            continue
        optimizer.step()

        # ── pruning update ──
        pruned_now = None
        if global_step in schedule:
            masker.sync_importance()
            target = cubic_sparsity(global_step, cfg.initial_sparsity, cfg.target_sparsity,
                                    cfg.prune_start_step, cfg.prune_end_step)
            # per-type overrides ride the same cubic curve (scaled by their final value)
            frac = target / cfg.target_sparsity if cfg.target_sparsity > 0 else 1.0
            targets = {u: (float(cfg.target_sparsity_by_type[u]) * frac
                           if u in cfg.target_sparsity_by_type else target)
                       for u in cfg.prune_units}
            gen = torch.Generator().manual_seed(cfg.seed * 1000 + global_step)
            external = None
            if cfg.importance == "routing_column":
                if world_size > 1:
                    dist.all_reduce(column_mass_sum, op=dist.ReduceOp.SUM)
                scores = module_importance_from_mass(column_mass_sum, module.num_layers)
                external = {u: scores[u] for u in cfg.prune_units if u in scores}
                column_mass_sum = None
            pruned_now = masker.prune_to(targets, unit_criterion, cfg.min_alive_per_layer, gen,
                                         external_scores=external)
            masker.broadcast_masks()
            masker.reset_importance()
            if is_main:
                rep = masker.report()
                print(f"[prune @ {global_step}] target={target:.3f} newly={pruned_now} "
                      f"units={masker.counts()} base_param_sparsity={rep['prune/base_param_sparsity']:.3f}",
                      flush=True)
                if cfg.eval_after_prune:
                    record_eval(global_step, "post_prune")
            if world_size > 1:
                dist.barrier()

        # ── logging ──
        if is_main and global_step % cfg.log_every == 0:
            elapsed = time.time() - t0
            metrics = {
                "train/nll": accum_nll,
                "train/lr": optimizer.param_groups[0]["lr"],
                "train/grad_norm": float(grad_norm),
                "train/tokens_per_sec": tokens_per_step * (global_step + 1) / max(elapsed, 1e-9),
                "schedule/target_sparsity": cubic_sparsity(
                    global_step, cfg.initial_sparsity, cfg.target_sparsity,
                    cfg.prune_start_step, cfg.prune_end_step),
            }
            metrics.update(masker.report())
            log_metrics(metrics, global_step, wandb_run)
            csv_logger.log(global_step, metrics)

        # ── periodic eval / save ──
        do_eval = global_step > 0 and global_step % cfg.eval_every == 0 and pruned_now is None
        if do_eval and is_main:
            record_eval(global_step, "periodic")
        if cfg.save_every and global_step > 0 and global_step % cfg.save_every == 0:
            save_latest()
        if (do_eval or (cfg.save_every and global_step % cfg.save_every == 0)) and world_size > 1:
            dist.barrier()

        global_step += 1

    # ── final ──
    if is_main:
        final = record_eval(global_step, "final")
        bake_masks_into_weights(masker)
        masker.enabled = False
        baked = evaluate(module, eval_batches, device, cfg.eval_max_batches)
        masker.enabled = True
        assert abs(baked - final["eval/domain_nll"]) < 1e-2, (
            f"baked weights disagree with masks: {baked} vs {final['eval/domain_nll']}")
        out = os.path.join(cfg.save_dir, "final")
        save_checkpoint(out, module, masker, None, global_step, model_cfg, cfg, final=True)
        summary = {
            "model": "fourway" if module.is_fourway else "dense",
            "model_dir": cfg.model_dir or cfg.model_config_path,
            "base_params_total": base_total,
            "base_params_remaining": int(masker.report()["prune/base_params_remaining"]),
            "block_params_total": masker.block_params_total(),
            "block_params_remaining": int(masker.report()["prune/block_params_remaining"]),
            "unit_counts": masker.counts(),
            "final_domain_nll": final["eval/domain_nll"],
            "final_general_nll": final.get("eval/general_nll"),
            "initial_domain_nll": trajectory[0]["eval/domain_nll"] if trajectory else None,
            "steps": global_step,
            "tokens": tokens_per_step * global_step,
            "config": cfg.to_dict(),
        }
        with open(os.path.join(cfg.save_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))
        latest = os.path.join(cfg.save_dir, "latest")
        if os.path.isdir(latest):
            shutil.rmtree(latest, ignore_errors=True)
    finish_wandb(wandb_run)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
