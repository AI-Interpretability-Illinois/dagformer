"""Checkpoint registry and model loading for the lm-eval reasoning comparison.

Every model under ``/work/hdd/bfqt/shared/dagformer-models`` is a directory
holding ``config.yaml`` (the exact training config) plus the checkpoint(s).
Two kinds live there:

    dense     standard OLMo-2.  ``checkpoint.pt`` has ``model_state_dict``.
    fourway   DAGFormer: the same OLMo-2 backbone plus per-token, per-head
              routing over the Q/K/V/residual streams.  ``checkpoint.pt`` holds
              the predictor + routing params, the base weights sit in the
              ``checkpoint_step*_model.pt`` side file next to it.

The weight loading itself is *not* reimplemented here — ``scripts/eval_lm_harness.py``
is the project's loader and the one the shared checkpoints were verified with,
so :func:`load_model` dispatches to its ``load_dense`` / ``load_fourway``.  What
this module adds is (a) discovery/pairing of baseline-vs-DAGFormer checkpoints
and (b) :class:`FourWayCausalLM`, which gives the two-module DAGFormer the
minimal HF-model surface that lm-eval's ``HFLM`` expects from a pre-initialised
model (``.config``, ``.device``, ``tie_weights()``, ``forward -> .logits``).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from transformers.modeling_outputs import CausalLMOutputWithPast

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.eval_lm_harness import load_dense, load_fourway  # noqa: E402

MODELS_ROOT = Path("/work/hdd/bfqt/shared/dagformer-models")
DEFAULT_TOKENIZER = MODELS_ROOT / "tokenizer"

# Directory-name suffix -> the label used in the comparison tables.
FAMILIES = {"baseline": "baseline", "dagformer": "dagformer"}


@dataclass(frozen=True)
class ModelSpec:
    """One evaluable checkpoint directory."""

    name: str  # directory name, e.g. "300m-dagformer"
    path: Path
    kind: str  # "dense" | "fourway" — which loader to use
    family: str  # "baseline" | "dagformer" — which column of the comparison
    size: str  # "75m" | "150m" | ...
    config: dict

    @property
    def config_path(self) -> Path:
        return self.path / "config.yaml"

    @property
    def ckpt_path(self) -> Path:
        return self.path / "checkpoint.pt"

    @property
    def size_millions(self) -> float:
        """Parameter count implied by the directory name ('75m', '1b') in millions."""
        text = self.size.lower()
        scale = 1000.0 if text.endswith("b") else 1.0
        try:
            return float(text.rstrip("mb")) * scale
        except ValueError:  # unparseable name: sort it last, don't crash the sweep
            return float("inf")

    @property
    def train_seq_len(self) -> int:
        """Sequence length the model was trained at.

        Worth respecting: RoPE is configured for 4096 positions but none of
        these runs ever saw a context longer than ``seq_len`` (1024), so
        evaluating past it measures length extrapolation, not reasoning.
        """
        return int(self.config.get("seq_len", 1024))

    @property
    def sort_key(self) -> tuple:
        return (self.size_millions, self.family)


def _classify(cfg: dict) -> str:
    return "fourway" if str(cfg.get("routing_mode", "")).startswith("fourway") else "dense"


def discover_models(root: Path | str = MODELS_ROOT) -> dict[str, ModelSpec]:
    """Index every checkpoint directory under ``root``.

    A directory counts if it has both ``config.yaml`` and ``checkpoint.pt``;
    the kind is read off ``routing_mode`` in the config rather than the name,
    so a renamed or newly added checkpoint still loads with the right builder.
    """
    root = Path(root)
    specs: dict[str, ModelSpec] = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / "config.yaml").exists() or not (d / "checkpoint.pt").exists():
            continue
        cfg = yaml.safe_load((d / "config.yaml").read_text())
        size, _, suffix = d.name.partition("-")
        specs[d.name] = ModelSpec(
            name=d.name,
            path=d,
            kind=_classify(cfg),
            family=FAMILIES.get(suffix, suffix or "unknown"),
            size=size,
            config=cfg,
        )
    return dict(sorted(specs.items(), key=lambda kv: kv[1].sort_key))


def resolve_models(names: list[str] | None, root: Path | str = MODELS_ROOT) -> list[ModelSpec]:
    """Turn a list of ``--model`` arguments into specs.

    ``None`` or ``["all"]`` means every discovered checkpoint; ``"300m"`` means
    both families at that size, so ``--model 300m`` runs the pair you actually
    want to compare.
    """
    available = discover_models(root)
    if not names or names == ["all"]:
        return list(available.values())

    out: list[ModelSpec] = []
    for name in names:
        if name in available:
            out.append(available[name])
            continue
        matches = [s for s in available.values() if s.size == name or s.family == name]
        if not matches:
            raise SystemExit(
                f"unknown model {name!r}; available: {', '.join(available)} "
                f"(or a size like '300m', or a family like 'dagformer')"
            )
        out.extend(matches)
    # de-duplicate, keep size/family order
    seen: dict[str, ModelSpec] = {s.name: s for s in out}
    return sorted(seen.values(), key=lambda s: s.sort_key)


class FourWayCausalLM(nn.Module):
    """FourWay DAGFormer (routing predictor + routed backbone) as one causal LM.

    The predictor is a separate causal encoder that maps ``input_ids`` to the
    per-token routing weights the backbone consumes, so a forward pass is
    always two calls.  lm-eval only ever needs ``model(input_ids).logits``,
    plus the few attributes ``HFLM.__init__`` touches when it is handed an
    already-initialised model: ``.config`` (a real ``Olmo2Config``, so pad-token
    and backend detection behave), ``.device``, and ``tie_weights()``.

    Note there is no KV cache: the routed attention recomputes every layer's
    per-head inputs from all prior layer outputs, so generation has to re-run
    the whole prefix per step (see ``harness.CheckpointLM._generate_recompute``).
    """

    def __init__(self, fourway_model: nn.Module, predictor: nn.Module):
        super().__init__()
        self.fourway_model = fourway_model
        self.fourway_predictor = predictor
        self.config = fourway_model.olmo.config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def tie_weights(self) -> None:
        """No-op: weight tying lives inside the wrapped OLMo backbone."""

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, **kwargs) -> CausalLMOutputWithPast:
        routing = self.fourway_predictor(input_ids)
        logits = self.fourway_model(input_ids, routing)
        return CausalLMOutputWithPast(logits=logits)


def load_model(spec: ModelSpec, device: torch.device | str = "cuda") -> nn.Module:
    """Build the model described by ``spec`` and load its weights.

    Returns an eval-mode, gradient-free module whose ``forward(input_ids)``
    returns an object with ``.logits`` — the dense path is a plain
    ``Olmo2ForCausalLM`` (so HF generation with a KV cache works), the fourway
    path is the :class:`FourWayCausalLM` wrapper.
    """
    device = torch.device(device)
    if spec.kind == "dense":
        model = load_dense(str(spec.ckpt_path), spec.config, device)
    elif spec.kind == "fourway":
        fourway_model, predictor = load_fourway(str(spec.ckpt_path), spec.config, device)
        model = FourWayCausalLM(fourway_model, predictor)
    else:
        raise ValueError(f"unhandled model kind {spec.kind!r} for {spec.name}")

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def checkpoint_step(ckpt_path: Path | str) -> int | None:
    """Training step recorded in a checkpoint, or None if it has no ``step`` key.

    Worth carrying into the results: the shared checkpoints are not all
    step-matched (the 300M DAGFormer stopped at 9000 steps against the
    baseline's 12000), and a comparison that does not say so is misleading.
    ``mmap=True`` keeps this to a header read rather than a 1GB load.
    """
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False, mmap=True)
    except Exception:  # noqa: BLE001 — provenance is nice to have, never fatal
        return None
    step = ckpt.get("step", ckpt.get("global_step"))
    return int(step) if step is not None else None


def load_tokenizer(path: Path | str = DEFAULT_TOKENIZER):
    """All these runs share the OLMo-2 tokenizer; the local copy works offline."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
