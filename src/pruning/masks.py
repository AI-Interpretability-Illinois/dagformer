"""Structured unit masks for OLMo-2, shared by the dense baseline and DAGFormer.

Four kinds of prunable unit, all addressed per layer:

    head     one attention head: its q/k/v rows, its o_proj columns and its
             slice of q_norm / k_norm.       mask shape [L, H]
    neuron   one MLP intermediate channel: a gate_proj row, an up_proj row
             and a down_proj column.         mask shape [L, I]
    attn     the whole attention sub-block of a layer.   mask shape [L, 1]
    mlp      the whole MLP sub-block of a layer.         mask shape [L, 1]

The masks are applied with forward hooks on modules that BOTH forward passes
call by module (``HF Olmo2DecoderLayer.forward`` and
``FourWayDAGFormer.forward`` / ``FourWayModularDAGFormer.forward``):

    head    forward_pre_hook on  layer.self_attn.o_proj      (input [.., H*hd])
    neuron  forward_pre_hook on  layer.mlp.down_proj         (input [.., I])
    attn    forward_hook     on  layer.post_attention_layernorm   (output)
    mlp     forward_hook     on  layer.post_feedforward_layernorm (output)

so one ``StructuredMasker`` prunes a dense ``Olmo2ForCausalLM`` and the OLMo
backbone inside a DAGFormer wrapper identically -- the comparison the pruning
experiment is about.

Every unit also has a *gate*: a ones tensor with ``requires_grad=True`` that
multiplies the mask. It is never optimised; its gradient is the first-order
Taylor importance of the unit (Michel et al., 2019; Molchanov et al., 2019):

    dL/dg_u = < dL/dy_u , y_u >     with y_u the unit's output

which the trainer accumulates between pruning steps (``accumulate_importance``)
and the masker consumes when asked to reach a sparsity target (``prune_to``).
Masks are 0/1 and a pruned unit never comes back.

Parameter accounting (``pruned_params`` / ``report``) is structure-aware: a
layer whose whole attention block is pruned counts its full attention
parameters once, not once per head on top.
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

UNIT_TYPES = ("head", "neuron", "attn", "mlp")
IMPORTANCE_CRITERIA = ("taylor", "fisher", "magnitude", "random")


def unit_param_counts(config) -> dict[str, int]:
    """Parameters removed by pruning one unit of each type (OLMo-2 has no
    biases; q_norm / k_norm are over the full H*hd vector, post-norms over D)."""
    D = int(config.hidden_size)
    H = int(config.num_attention_heads)
    I = int(config.intermediate_size)
    hd = D // H
    return {
        "head": 4 * D * hd + 2 * hd,        # q/k/v rows + o cols + q_norm/k_norm slice
        "neuron": 3 * D,                    # gate row + up row + down col
        "attn": 4 * D * D + 2 * D + D,      # 4 projections + q_norm + k_norm + post_attn norm
        "mlp": 3 * D * I + D,               # gate/up/down + post_ff norm
    }


def count_unique_params(module: nn.Module) -> int:
    """Parameter count that does not double-count tied weights."""
    seen: set[int] = set()
    total = 0
    for p in module.parameters():
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        total += p.numel()
    return total


class StructuredMasker:
    """Multiplicative 0/1 masks over structured units of an ``Olmo2ForCausalLM``.

    Args:
        olmo: the HF model to prune (for DAGFormer pass ``fourway_model.olmo``).
        unit_types: subset of ``UNIT_TYPES`` to make prunable.
        device: where masks / gates live (defaults to the model's device).
    """

    def __init__(
        self,
        olmo: nn.Module,
        unit_types: Iterable[str] = ("head", "neuron"),
        device: Optional[torch.device] = None,
    ) -> None:
        self.olmo = olmo
        cfg = olmo.config
        self.L = int(cfg.num_hidden_layers)
        self.H = int(cfg.num_attention_heads)
        self.D = int(cfg.hidden_size)
        self.I = int(cfg.intermediate_size)
        self.hd = self.D // self.H
        self.unit_types = tuple(unit_types)
        for t in self.unit_types:
            assert t in UNIT_TYPES, f"unknown unit type {t!r}; expected one of {UNIT_TYPES}"
        assert len(set(self.unit_types)) == len(self.unit_types), "duplicate unit types"

        if device is None:
            device = next(olmo.parameters()).device
        self.device = torch.device(device)

        self.shapes = {
            "head": (self.L, self.H),
            "neuron": (self.L, self.I),
            "attn": (self.L, 1),
            "mlp": (self.L, 1),
        }
        self.masks: dict[str, torch.Tensor] = {}
        self.gates: dict[str, torch.Tensor] = {}
        self.importance: dict[str, torch.Tensor] = {}
        for t in self.unit_types:
            shape = self.shapes[t]
            self.masks[t] = torch.ones(shape, device=self.device)
            self.gates[t] = torch.ones(shape, device=self.device, requires_grad=True)
            self.importance[t] = torch.zeros(shape, device=self.device)
        self.n_accumulated = 0          # optimiser steps folded into `importance`
        self.n_prune_events = 0
        self.enabled = True
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._attach()

    # ---- hooks -------------------------------------------------------------
    def _attach(self) -> None:
        for l, layer in enumerate(self.olmo.model.layers):
            if "head" in self.masks:
                self._handles.append(
                    layer.self_attn.o_proj.register_forward_pre_hook(self._head_hook(l)))
            if "neuron" in self.masks:
                self._handles.append(
                    layer.mlp.down_proj.register_forward_pre_hook(self._neuron_hook(l)))
            if "attn" in self.masks:
                self._handles.append(
                    layer.post_attention_layernorm.register_forward_hook(
                        self._module_hook("attn", l)))
            if "mlp" in self.masks:
                self._handles.append(
                    layer.post_feedforward_layernorm.register_forward_hook(
                        self._module_hook("mlp", l)))

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _gate(self, t: str, l: int, dtype: torch.dtype) -> torch.Tensor:
        return (self.masks[t][l] * self.gates[t][l]).to(dtype)

    def _head_hook(self, l: int):
        def hook(module: nn.Module, inputs: tuple):
            if not self.enabled:
                return None
            x = inputs[0]
            assert x.shape[-1] == self.H * self.hd, (x.shape, self.H, self.hd)
            g = self._gate("head", l, x.dtype)                       # [H]
            shape = x.shape
            xh = x.reshape(*shape[:-1], self.H, self.hd)
            g = g.view(*([1] * (xh.dim() - 2)), self.H, 1)
            return (xh * g).reshape(shape), *inputs[1:]
        return hook

    def _neuron_hook(self, l: int):
        def hook(module: nn.Module, inputs: tuple):
            if not self.enabled:
                return None
            x = inputs[0]
            assert x.shape[-1] == self.I, (x.shape, self.I)
            g = self._gate("neuron", l, x.dtype)                     # [I]
            return (x * g, *inputs[1:])
        return hook

    def _module_hook(self, t: str, l: int):
        def hook(module: nn.Module, inputs: tuple, output: torch.Tensor):
            if not self.enabled:
                return None
            g = self._gate(t, l, output.dtype)                       # [1]
            return output * g
        return hook

    # ---- importance --------------------------------------------------------
    @torch.no_grad()
    def accumulate_importance(self, criterion: str = "taylor") -> None:
        """Fold the gate gradients of the last backward(s) into ``importance``
        and clear them. Call once per optimiser step, after backward."""
        assert criterion in ("taylor", "fisher"), criterion
        for t in self.unit_types:
            g = self.gates[t].grad
            if g is None:
                continue
            if criterion == "taylor":
                self.importance[t] += g.abs()
            else:
                self.importance[t] += g.pow(2)
            self.gates[t].grad = None
        self.n_accumulated += 1

    @torch.no_grad()
    def reset_importance(self) -> None:
        for t in self.unit_types:
            self.importance[t].zero_()
            self.gates[t].grad = None
        self.n_accumulated = 0

    @torch.no_grad()
    def sync_importance(self) -> None:
        """Sum importance over DDP ranks so every rank prunes identically."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        for t in self.unit_types:
            dist.all_reduce(self.importance[t], op=dist.ReduceOp.SUM)

    @torch.no_grad()
    def broadcast_masks(self, src: int = 0) -> None:
        if not (dist.is_available() and dist.is_initialized()):
            return
        for t in self.unit_types:
            dist.broadcast(self.masks[t], src=src)

    @torch.no_grad()
    def magnitude_scores(self) -> dict[str, torch.Tensor]:
        """Weight-norm importance, the classic data-free criterion."""
        out: dict[str, torch.Tensor] = {}
        layers = self.olmo.model.layers
        if "head" in self.masks:
            s = torch.zeros(self.L, self.H, device=self.device)
            for l, layer in enumerate(layers):
                a = layer.self_attn
                Wq = a.q_proj.weight.float().view(self.H, self.hd, self.D)
                Wk = a.k_proj.weight.float().view(self.H, self.hd, self.D)
                Wv = a.v_proj.weight.float().view(self.H, self.hd, self.D)
                Wo = a.o_proj.weight.float().view(self.D, self.H, self.hd).permute(1, 0, 2)
                s[l] = (Wq.flatten(1).norm(dim=1) + Wk.flatten(1).norm(dim=1)
                        + Wv.flatten(1).norm(dim=1) + Wo.flatten(1).norm(dim=1))
            out["head"] = s
        if "neuron" in self.masks:
            s = torch.zeros(self.L, self.I, device=self.device)
            for l, layer in enumerate(layers):
                m = layer.mlp
                s[l] = (m.gate_proj.weight.float().norm(dim=1)
                        * m.up_proj.weight.float().norm(dim=1)
                        * m.down_proj.weight.float().norm(dim=0))
            out["neuron"] = s
        if "attn" in self.masks:
            s = torch.zeros(self.L, 1, device=self.device)
            for l, layer in enumerate(layers):
                a = layer.self_attn
                s[l, 0] = sum(w.weight.float().norm() for w in (a.q_proj, a.k_proj, a.v_proj, a.o_proj))
            out["attn"] = s
        if "mlp" in self.masks:
            s = torch.zeros(self.L, 1, device=self.device)
            for l, layer in enumerate(layers):
                m = layer.mlp
                s[l, 0] = sum(w.weight.float().norm() for w in (m.gate_proj, m.up_proj, m.down_proj))
            out["mlp"] = s
        return out

    def scores(self, criterion: str, generator: Optional[torch.Generator] = None
               ) -> dict[str, torch.Tensor]:
        """Per-unit importance under ``criterion`` (higher = keep)."""
        assert criterion in IMPORTANCE_CRITERIA, criterion
        if criterion in ("taylor", "fisher"):
            assert self.n_accumulated > 0, (
                f"no importance accumulated for criterion={criterion}; call "
                "accumulate_importance() after backward before pruning")
            return {t: self.importance[t].clone() for t in self.unit_types}
        if criterion == "magnitude":
            return self.magnitude_scores()
        # random: draw on CPU with the given generator so all ranks agree
        out = {}
        for t in self.unit_types:
            out[t] = torch.rand(self.shapes[t], generator=generator).to(self.device)
        return out

    # ---- pruning -----------------------------------------------------------
    @torch.no_grad()
    def prune_to(
        self,
        targets: dict[str, float],
        criterion: str = "taylor",
        min_alive_per_layer: Optional[dict[str, int]] = None,
        generator: Optional[torch.Generator] = None,
        external_scores: Optional[dict[str, torch.Tensor]] = None,
    ) -> dict[str, int]:
        """Prune the least important alive units until each type reaches its
        target fraction of pruned units (global ranking across layers).

        ``external_scores`` (type -> tensor of the mask's shape, higher = keep)
        overrides the criterion for the types it contains, e.g. routing column
        mass for whole-module units of the modular DAGFormer.

        Returns the number of units newly pruned per type. Units already pruned
        are never revived; if the target is below the current sparsity nothing
        happens for that type.
        """
        min_alive_per_layer = min_alive_per_layer or {}
        external_scores = external_scores or {}
        need_criterion = [t for t in targets if t not in external_scores]
        scores = self.scores(criterion, generator=generator) if need_criterion else {}
        for t, ext in external_scores.items():
            assert tuple(ext.shape) == tuple(self.shapes[t]), (t, ext.shape, self.shapes[t])
            scores[t] = ext.to(self.device)
        newly: dict[str, int] = {}
        for t, target in targets.items():
            assert t in self.masks, f"type {t!r} not managed by this masker ({self.unit_types})"
            assert 0.0 <= target <= 1.0, (t, target)
            mask = self.masks[t]
            n_total = mask.numel()
            n_target = int(round(target * n_total))
            n_pruned = int((mask == 0).sum().item())
            k = n_target - n_pruned
            newly[t] = 0
            if k <= 0:
                continue
            per_layer = mask.shape[1]
            min_alive = int(min_alive_per_layer.get(t, 0))
            alive_per_layer = (mask != 0).sum(dim=1)

            s = scores[t].detach().float().reshape(-1).clone()
            s[mask.reshape(-1) == 0] = float("inf")   # never re-select pruned units
            order = torch.argsort(s, stable=True)
            flat = mask.reshape(-1)
            for idx in order.tolist():
                if k == 0:
                    break
                if not math.isfinite(s[idx].item()):
                    break
                l = idx // per_layer
                if alive_per_layer[l] <= min_alive:
                    continue
                flat[idx] = 0.0
                alive_per_layer[l] -= 1
                k -= 1
                newly[t] += 1
        self.n_prune_events += 1
        return newly

    # ---- accounting --------------------------------------------------------
    def counts(self) -> dict[str, tuple[int, int]]:
        """type -> (pruned, total) unit counts."""
        return {t: (int((m == 0).sum().item()), m.numel()) for t, m in self.masks.items()}

    def pruned_params(self) -> int:
        """Parameters removed by the current masks (structure-aware)."""
        c = unit_param_counts(self.olmo.config)
        total = 0
        for l in range(self.L):
            attn_gone = "attn" in self.masks and self.masks["attn"][l, 0].item() == 0
            if attn_gone:
                total += c["attn"]
            elif "head" in self.masks:
                n = int((self.masks["head"][l] == 0).sum().item())
                if n == self.H:
                    # every head gone: same effect as removing the block
                    total += c["attn"]
                else:
                    total += n * c["head"]
            mlp_gone = "mlp" in self.masks and self.masks["mlp"][l, 0].item() == 0
            if mlp_gone:
                total += c["mlp"]
            elif "neuron" in self.masks:
                n = int((self.masks["neuron"][l] == 0).sum().item())
                total += c["mlp"] if n == self.I else n * c["neuron"]
        return total

    def report(self, prefix: str = "prune/") -> dict[str, float]:
        base_total = count_unique_params(self.olmo)
        pruned = self.pruned_params()
        out: dict[str, float] = {
            f"{prefix}base_params_total": float(base_total),
            f"{prefix}base_params_pruned": float(pruned),
            f"{prefix}base_params_remaining": float(base_total - pruned),
            f"{prefix}base_param_sparsity": pruned / max(base_total, 1),
            f"{prefix}events": float(self.n_prune_events),
        }
        for t, (n_pruned, n_total) in self.counts().items():
            out[f"{prefix}{t}_pruned"] = float(n_pruned)
            out[f"{prefix}{t}_total"] = float(n_total)
            out[f"{prefix}{t}_sparsity"] = n_pruned / max(n_total, 1)
        return out

    def per_layer_summary(self) -> dict[str, list[int]]:
        """Pruned units per layer for logging / plotting."""
        return {t: (m == 0).sum(dim=1).tolist() for t, m in self.masks.items()}

    # ---- persistence -------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "unit_types": list(self.unit_types),
            "masks": {t: m.detach().cpu().clone() for t, m in self.masks.items()},
            "importance": {t: v.detach().cpu().clone() for t, v in self.importance.items()},
            "n_accumulated": self.n_accumulated,
            "n_prune_events": self.n_prune_events,
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict) -> None:
        assert list(state["unit_types"]) == list(self.unit_types), (
            state["unit_types"], self.unit_types)
        for t in self.unit_types:
            self.masks[t].copy_(state["masks"][t].to(self.device))
            if "importance" in state and t in state["importance"]:
                self.importance[t].copy_(state["importance"][t].to(self.device))
        self.n_accumulated = int(state.get("n_accumulated", 0))
        self.n_prune_events = int(state.get("n_prune_events", 0))
