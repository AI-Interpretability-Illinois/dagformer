"""Run lm-evaluation-harness on a dense baseline or FourWay DAGFormer checkpoint.

Usage:
    # Dense baseline:
    python scripts/eval_lm_harness.py \
        --config configs/pretrain_300m_baseline_5k.yaml \
        --ckpt checkpoints/pretrain_300m_baseline/checkpoint_step8000.pt \
        --tasks default \
        --output_path experiments/results/lmeval_300m_baseline_8k.json

    # FourWay DAGFormer:
    python scripts/eval_lm_harness.py \
        --config configs/fourway_vnorm_full_fix.yaml \
        --ckpt checkpoints/fourway_vnorm_full_fix/checkpoint_step5000.pt \
        --tasks default \
        --output_path experiments/results/lmeval_300m_dag_vnorm_5k.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer, Olmo2Config, Olmo2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Default benchmark suite (loglikelihood-style, size-appropriate for 75M-1B)
DEFAULT_TASKS = [
    "lambada_openai",
    "hellaswag",
    "piqa",
    "arc_easy",
    "winogrande",
    "openbookqa",
    "sciq",
    "boolq",
    "wikitext",
]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_base_model(cfg: dict, device, dtype=torch.bfloat16):
    mc = Olmo2Config(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_attention_heads"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        tie_word_embeddings=cfg.get("tie_word_embeddings", True),
        max_position_embeddings=cfg.get("max_position_embeddings", 4096),
    )
    model = Olmo2ForCausalLM(mc)
    return model.to(device=device, dtype=dtype)


def strip_prefixes(state: dict) -> dict:
    """Strip wrapper prefixes from checkpoint keys.

    Common cases:
    - "_orig_mod.": torch.compile wraps the module
    - "module.": DDP wraps the module
    - "base_model.olmo.", "olmo.", "base_model.": various pipeline wrappers
    Order matters: strip outermost wrappers first, and loop until stable to
    handle stacked prefixes (e.g. "module._orig_mod.foo" → "foo").
    """
    prefixes = (
        "module.base_model.", "module.predictor.", "module.fourway_model.",
        "_orig_mod.",  # torch.compile
        "module.",     # DDP
        "base_model.olmo.", "olmo.", "base_model.",
    )
    out = {}
    for k, v in state.items():
        nk = k
        # Loop: a key may have multiple stacked prefixes
        for _ in range(4):
            stripped = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p):]
                    stripped = True
                    break
            if not stripped:
                break
        out[nk] = v
    return out


def load_dense(ckpt_path: str, cfg: dict, device):
    model = build_base_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_path" in ckpt:
        side = ckpt["model_state_path"]
        if not Path(side).is_absolute():
            cand = Path(ckpt_path).parent / Path(side).name
            if cand.exists():
                side = str(cand)
        state = torch.load(side, map_location="cpu", weights_only=False)
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = strip_prefixes(state)
    m, u = model.load_state_dict(state, strict=False)
    print(f"[load_dense] missing={len(m)} unexpected={len(u)} (first 3 missing: {m[:3]})")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _replace_olmo_rmsnorm(model):
    """Inlined from scripts/pretrain_dagformer.py — avoids importing the trainer module."""
    import torch.nn as nn
    count = 0
    for name, module in list(model.named_modules()):
        if type(module).__name__ == "Olmo2RMSNorm":
            hidden_size = module.weight.shape[0]
            eps = module.variance_epsilon
            new_norm = nn.RMSNorm(hidden_size, eps=eps).to(
                device=module.weight.device, dtype=module.weight.dtype
            )
            new_norm.weight = module.weight
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_norm)
            count += 1
    return count


def load_fourway(ckpt_path: str, cfg: dict, device):
    """Load a FourWay DAGFormer (base OLMo + FourWayPredictor + FourWayDAGFormer wrapper).

    routing_mode "fourway_modular[_corrected]" loads the module-granular variant
    (src/model/modular_routing.py) instead; the checkpoint layout is the same.
    """
    from src.model.olmo_graph import FourWayDAGFormer
    from src.model.predictor import FourWayPredictor
    replace_olmo_rmsnorm = _replace_olmo_rmsnorm

    base = build_base_model(cfg, device)
    if cfg.get("replace_rmsnorm", False):
        replace_olmo_rmsnorm(base)
        base = base.to(device=device, dtype=torch.bfloat16)

    if str(cfg.get("routing_mode", "")).startswith("fourway_modular"):
        from src.model.modular_routing import build_modular_pair
        fourway_model, fourway_predictor = build_modular_pair(cfg, base, device)
        return _load_fourway_state(ckpt_path, base, fourway_model, fourway_predictor)

    fourway_model = FourWayDAGFormer(
        model=base,
        num_layers=cfg["num_hidden_layers"],
        num_heads=cfg["num_attention_heads"],
        use_local_correction=(cfg.get("routing_mode", "") == "fourway_corrected"),
        correction_hidden=cfg.get("correction_hidden", 128),
        use_triton_kernel=cfg.get("use_triton_kernel", False),
        use_v_norm=cfg.get("use_v_norm", False),
        correction_pool=cfg.get("correction_pool", "none"),
    ).to(device=device)
    if cfg.get("fourway_predictor_variant", "encoder") == "per_layer":
        from src.model.predictor import FourWayPerLayerPredictor as FourWayPredictor  # noqa: F811
    fourway_predictor = FourWayPredictor(
        vocab_size=cfg["vocab_size"],
        encoder_dim=cfg.get("predictor_encoder_dim", 256),
        encoder_layers=cfg.get("predictor_encoder_layers", 2),
        encoder_heads=cfg.get("predictor_encoder_heads", 4),
        max_seq_len=cfg.get("predictor_max_seq_len", 4096),
        num_layers=cfg["num_hidden_layers"],
        num_heads=cfg["num_attention_heads"],
        hidden_dim=cfg.get("fourway_hidden", 512),
        causal=cfg.get("predictor_causal", True),
        dropout=0.0,
    ).to(device=device)

    return _load_fourway_state(ckpt_path, base, fourway_model, fourway_predictor)


def _load_fourway_state(ckpt_path: str, base, fourway_model, fourway_predictor):
    """Load predictor / base / routing state into an already-built pair."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Predictor
    if "predictor_state_dict" in ckpt:
        ps = strip_prefixes(ckpt["predictor_state_dict"])
        m, u = fourway_predictor.load_state_dict(ps, strict=False)
        print(f"[load_fourway:predictor] missing={len(m)} unexpected={len(u)}")
    # Base model
    state = None
    if "model_state_path" in ckpt:
        side = ckpt["model_state_path"]
        if not Path(side).is_absolute():
            cand = Path(ckpt_path).parent / Path(side).name
            if cand.exists():
                side = str(cand)
        state = torch.load(side, map_location="cpu", weights_only=False)
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    if state is not None:
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = strip_prefixes(state)
        m, u = base.load_state_dict(state, strict=False)
        print(f"[load_fourway:base] missing={len(m)} unexpected={len(u)} (first 3 missing: {m[:3]})")
    # Routing state (v_norms, correction MLPs)
    if "routing_state_dict" in ckpt and len(ckpt["routing_state_dict"]) > 0:
        rs = strip_prefixes(ckpt["routing_state_dict"])
        m, u = fourway_model.load_state_dict(rs, strict=False)
        print(f"[load_fourway:routing] loaded {len(rs)} keys (missing={len(m)} unexpected={len(u)})")

    fourway_model.eval()
    fourway_predictor.eval()
    for p in fourway_model.parameters():
        p.requires_grad_(False)
    for p in fourway_predictor.parameters():
        p.requires_grad_(False)
    return fourway_model, fourway_predictor


def _cfg_to_namespace(cfg: dict):
    """Adapt the plain YAML dict into a PretrainConfig-shaped object.

    The reproduction-baseline builders (``build_denseformer`` /
    ``build_hyperconnection`` / ``build_muddformer``) read their fields via
    attribute access (``cfg.hidden_size``) and optional knobs via
    ``getattr(cfg, "...", default)`` — the same contract as the
    ``PretrainConfig`` produced by ``scripts/pretrain_baseline.py``. The eval
    script keeps ``cfg`` as a dict for the dense/fourway paths, so wrap it in a
    ``SimpleNamespace`` here to give the builders the attribute access they
    expect (missing optional keys correctly fall through to the builder
    defaults). This mirrors ``scripts/muddformer_lm_eval.py``.
    """
    import types
    return types.SimpleNamespace(**cfg)


def _load_method_checkpoint_state(ckpt_path: str) -> dict:
    """Pull the model state_dict out of a pretrain_baseline-style checkpoint.

    These reproduction baselines are trained by ``pretrain_baseline.main()``
    (via the ``pretrain_{denseformer,hyperconnection,muddformer}.py`` drop-ins),
    so ``save_checkpoint`` writes a single ``.pt`` whose ``model_state_dict`` is
    ``model.state_dict()`` — possibly DDP-/compile-wrapped. Strip wrapper
    prefixes so the keys line up with a freshly built method model.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    return strip_prefixes(state)


def _report_load(tag: str, model, state: dict, routing_substrings: tuple[str, ...]):
    """load_state_dict(strict=False) + print/verify routing keys actually loaded.

    A correct load of a method model must have ~0 missing keys. In particular
    the routing params (DWA alphas / hyper-connection alpha·beta / MuDD dense
    routing) MUST be present in the checkpoint and NOT show up as missing —
    otherwise we would be silently evaluating a plain OLMo-2 (the exact bug that
    using ``load_dense`` for these models would cause). Warn loudly if any
    routing key is missing.
    """
    m, u = model.load_state_dict(state, strict=False)
    missing_routing = [k for k in m if any(s in k for s in routing_substrings)]
    loaded_routing = [k for k in state if any(s in k for s in routing_substrings)]
    print(f"[{tag}] missing={len(m)} unexpected={len(u)} "
          f"(first 3 missing: {m[:3]})")
    print(f"[{tag}] routing keys loaded from checkpoint: {len(loaded_routing)} "
          f"(e.g. {loaded_routing[:2]})")
    if missing_routing:
        print(f"[{tag}] WARNING: {len(missing_routing)} routing keys MISSING "
              f"(silently dropped!): {missing_routing[:5]}")
    else:
        print(f"[{tag}] OK: all routing params loaded (0 routing keys missing)")
    if len(m) > 5:
        print(f"[{tag}] WARNING: {len(m)} total keys missing — expected ~0 for a "
              f"correct method-model load. Missing sample: {m[:10]}")
    return m, u


def load_denseformer(ckpt_path: str, cfg: dict, device):
    """Load a DenseFormer checkpoint into the ACTUAL DWA-augmented model.

    Builds ``DenseFormerForCausalLM`` via ``build_denseformer`` (NOT a plain
    Olmo2 — that would drop the DWA alpha params), loads the checkpoint's
    ``model_state_dict``, and confirms the DWA routing params are loaded.
    HF-CausalLM-compatible (``forward(input_ids=...) -> .logits``), so it wraps
    in the same ``make_lm``/HFLM path as ``load_dense``.
    """
    from src.model.denseformer.wrapper import build_denseformer

    model = build_denseformer(_cfg_to_namespace(cfg))
    model = model.to(device=device, dtype=torch.bfloat16)
    state = _load_method_checkpoint_state(ckpt_path)
    _report_load("load_denseformer", model, state, routing_substrings=("dwa_modules",))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_hyperconnection(ckpt_path: str, cfg: dict, device):
    """Load a Hyper-Connections checkpoint into the ACTUAL HC model.

    Builds ``HyperConnectionOlmo2ForCausalLM`` via ``build_hyperconnection``
    (NOT a plain Olmo2 — that would drop the per-layer alpha/beta routing),
    loads ``model_state_dict``, and confirms the hyper-connection routing params
    are loaded. HF-CausalLM-compatible, wraps in the same ``make_lm`` path.
    """
    from src.model.hyperconnection.wrapper import build_hyperconnection

    model = build_hyperconnection(_cfg_to_namespace(cfg))
    model = model.to(device=device, dtype=torch.bfloat16)
    state = _load_method_checkpoint_state(ckpt_path)
    _report_load("load_hyperconnection", model, state,
                 routing_substrings=("hyper_connections",))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_muddformer(ckpt_path: str, cfg: dict, device):
    """Load a MuDDFormer checkpoint into the ACTUAL dynamic-dense model.

    Builds ``MUDDFormerForCausalLM`` via ``build_muddformer`` (its native
    architecture — a plain Olmo2 could not represent the dynamic dense
    routing), loads ``model_state_dict``, and confirms the dense routing params
    are loaded. HF-CausalLM-compatible (``forward(input_ids=...) -> .logits``
    via CausalLMOutputWithPast), wraps in the same ``make_lm`` path.
    """
    from src.model.muddformer.wrapper import build_muddformer

    model = build_muddformer(_cfg_to_namespace(cfg))
    model = model.to(device=device, dtype=torch.bfloat16)
    state = _load_method_checkpoint_state(ckpt_path)
    # MuDD routing lives in dense_bs / dynamic_dense (+ per-branch dense_* proj).
    _report_load("load_muddformer", model, state,
                 routing_substrings=("dense_bs", "dynamic_dense"))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class FourWayLMWrapper(torch.nn.Module):
    """Wraps FourWay model + predictor into a single `forward(input_ids)->logits` interface."""

    def __init__(self, fourway_model, fourway_predictor, vocab_size):
        super().__init__()
        self.fourway_model = fourway_model
        self.fourway_predictor = fourway_predictor
        self.config = type("Cfg", (), {"vocab_size": vocab_size})()

    @torch.no_grad()
    def forward(self, input_ids, **kwargs):
        routing = self.fourway_predictor(input_ids)
        logits = self.fourway_model(input_ids, routing)
        # Match HF interface: return object with .logits. Use a real output
        # object, NOT `type("Out", (), {"logits": logits})()`: a class created
        # per call keeps the logits alive in a reference cycle until the cyclic
        # GC runs, so a full eval leaked ~1 batch of logits per step and OOM'd a
        # 48 GB GPU after ~7 batches (2026-09-25).
        return CausalLMOutputWithPast(logits=logits)


def make_lm(model, tokenizer, batch_size: int = 8, max_length: int = 1024):
    """Create an lm_eval LM wrapper around our model."""
    from lm_eval.models.huggingface import HFLM

    class CustomHFLM(HFLM):
        def __init__(self, our_model, tokenizer, batch_size, max_length):
            # Bypass HFLM.__init__ entirely — HFLM doesn't inherit from nn.Module
            # so calling Module.__init__(self) breaks. Set fields manually instead.
            self._model = our_model
            self.tokenizer = tokenizer
            self._batch_size = int(batch_size)
            self._max_length = int(max_length)
            self._device = next(our_model.parameters()).device
            self.backend = "causal"
            self.add_bos_token = False
            self.custom_prefix_token_id = None
            self.logits_cache = True
            self.truncation = False
            self.vocab_size = tokenizer.vocab_size
            self._rank = 0
            self._world_size = 1
            self.batch_schedule = 1
            self.batch_sizes = {}
            self.max_batch_size = 64
            self._prefix_token_id = (
                tokenizer.bos_token_id if tokenizer.bos_token_id is not None
                else tokenizer.eos_token_id
            )
            # LM base class attrs commonly accessed
            self._is_main_process = True
            self.cache_hook = type("NoCache", (), {"add_partial": lambda *a, **k: None})()
            # Misc HFLM internals accessed during eval
            self.softmax_dtype = None  # None → defaults to logits dtype
            self.mixed_precision_dtype = None
            self.think_end_token = None
            self.enable_thinking = False
            self.chat_template_args = None
            self.pretrained_name_or_path = "custom"
            # get_model_info() expects these
            self.revision = "main"
            self.peft = None
            self.delta = None

        @property
        def prefix_token_id(self): return self._prefix_token_id

        @property
        def rank(self): return self._rank

        @property
        def world_size(self): return self._world_size

        @property
        def device(self): return self._device

        @property
        def max_length(self): return self._max_length

        @property
        def max_gen_toks(self): return 256

        @property
        def batch_size(self): return self._batch_size

        @property
        def model(self): return self._model

        @property
        def eot_token_id(self):
            return self.tokenizer.eos_token_id

        def tok_encode(self, string, **kwargs):
            return self.tokenizer.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens, **kwargs):
            return self.tokenizer.decode(tokens, skip_special_tokens=True)

        def _model_call(self, inps, attn_mask=None, labels=None):
            # inps: [B, T] long tensor
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self._model(inps.to(self._device))
            return out.logits

        def _model_generate(self, context, max_length, stop, **kwargs):
            raise NotImplementedError("generate_until not implemented for these eval tasks")

        def get_model_info(self):
            return {
                "model_source": "custom",
                "model_name": "custom",
                "model_dtype": str(next(self._model.parameters()).dtype),
                "model_revision": self.revision,
                "model_num_parameters": sum(p.numel() for p in self._model.parameters()),
                "model_sha": "",
                "peft_sha": "",
                "delta_sha": "",
            }

    return CustomHFLM(model, tokenizer, batch_size, max_length)


def resolve_model_type(args, cfg: dict) -> str:
    """Decide which loader to dispatch to.

    Priority:
      1. Explicit ``--model-type`` (anything but the default 'auto').
      2. Explicit legacy ``--mode`` (dense/fourway) for back-compat.
      3. A ``model_type`` / ``method`` field in the config.
      4. Substring match on the config filename
         (denseformer / hyperconnection / muddformer / fourway).
      5. Fall back to fourway if ``routing_mode`` is a fourway variant, else dense.
    """
    known = {"dense", "fourway", "denseformer", "hyperconnection", "muddformer"}

    if getattr(args, "model_type", "auto") != "auto":
        return args.model_type

    # Legacy --mode still honored when set explicitly.
    if getattr(args, "mode", "auto") != "auto":
        return args.mode

    for field in ("model_type", "method"):
        val = str(cfg.get(field, "")).lower()
        if val in known:
            return val

    fname = os.path.basename(args.config or "").lower()
    # Order matters: check the more specific names first.
    for name in ("denseformer", "hyperconnection", "muddformer", "fourway"):
        if name in fname:
            return name

    if cfg.get("routing_mode", "") in ("fourway", "fourway_corrected"):
        return "fourway"
    return "dense"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                        help="YAML config (required for --ckpt mode, optional for --hf_model)")
    parser.add_argument("--ckpt", default=None,
                        help="Path to .pt checkpoint (mutually exclusive with --hf_model)")
    parser.add_argument("--hf_model", default=None,
                        help="HF model id (e.g. allenai/OLMo-2-0425-1B). Loads via AutoModelForCausalLM.")
    parser.add_argument("--hf_revision", default="main",
                        help="HF revision/branch (e.g. stage2-ingredient1-step2000-tokens5B)")
    parser.add_argument("--mode", choices=["auto", "dense", "fourway"], default="auto",
                        help="[legacy] auto: infer from config (presence of routing_mode). "
                             "Prefer --model-type for the reproduction baselines.")
    parser.add_argument(
        "--model-type", dest="model_type",
        choices=["auto", "dense", "fourway", "denseformer", "hyperconnection", "muddformer"],
        default="auto",
        help="Which model to build/load. 'auto' infers from the config: a "
             "'model_type'/'method' field if present, else the config filename, "
             "else routing_mode (fourway) / dense. Use e.g. "
             "'--model-type denseformer' to load a DWA DenseFormer checkpoint "
             "(NOT --mode, which only knows dense/fourway).")
    parser.add_argument("--tasks", default="default",
                        help="Comma-separated task names, or 'default'")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples per task (for debugging)")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.hf_model is None and args.ckpt is None:
        parser.error("Need either --ckpt or --hf_model")

    device = torch.device(args.device)

    if args.hf_model is not None:
        # Load HF model directly (e.g. allenai/OLMo-2-0425-1B at a given revision)
        from transformers import AutoModelForCausalLM
        print(f"[main] Loading HF model: {args.hf_model} @ {args.hf_revision}")
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            revision=args.hf_revision,
            torch_dtype=torch.bfloat16,
        ).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        wrapped = model
        tokenizer_id = args.hf_model
        mode = "dense"
    else:
        cfg = load_config(args.config)
        mode = resolve_model_type(args, cfg)
        print(f"[main] model-type: {mode}, ckpt={args.ckpt}")
        if mode == "dense":
            model = load_dense(args.ckpt, cfg, device)
            wrapped = model
        elif mode == "fourway":
            fourway_model, fourway_predictor = load_fourway(args.ckpt, cfg, device)
            wrapped = FourWayLMWrapper(fourway_model, fourway_predictor, cfg["vocab_size"])
        elif mode == "denseformer":
            wrapped = load_denseformer(args.ckpt, cfg, device)
        elif mode == "hyperconnection":
            wrapped = load_hyperconnection(args.ckpt, cfg, device)
        elif mode == "muddformer":
            wrapped = load_muddformer(args.ckpt, cfg, device)
        else:
            raise ValueError(f"Unhandled model-type: {mode}")
        tokenizer_id = cfg["tokenizer_id"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    # Some checkpoints lack pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tasks = DEFAULT_TASKS if args.tasks == "default" else args.tasks.split(",")
    print(f"[main] Tasks: {tasks}")

    lm = make_lm(wrapped, tokenizer, batch_size=args.batch_size, max_length=args.max_length)

    import lm_eval
    t0 = time.time()
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=0,
        batch_size=args.batch_size,
        device=str(device),
        limit=args.limit,
    )
    print(f"[main] Eval finished in {time.time()-t0:.1f}s")

    # Strip non-serializable bits before saving
    results_min = {
        "config": {
            "ckpt": args.ckpt,
            "config_file": args.config,
            "hf_model": args.hf_model,
            "hf_revision": args.hf_revision,
            "mode": mode,
            "tasks": tasks,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "limit": args.limit,
        },
        "results": results.get("results", {}),
        "n-samples": results.get("n-samples", {}),
        "versions": results.get("versions", {}),
    }
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results_min, f, indent=2, default=str)
    print(f"[main] Saved: {out_path}")

    # Pretty-print summary
    print("\n=== Summary ===")
    for task, m in results_min["results"].items():
        print(f"  {task}: {json.dumps({k: v for k, v in m.items() if not k.startswith('alias')}, default=str)}")


if __name__ == "__main__":
    main()
