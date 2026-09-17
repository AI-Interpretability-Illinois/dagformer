"""Shared machinery for behavioural circuit discovery in the routing graph.

Vocabulary used throughout this directory
-----------------------------------------
The FourWay model rewires itself per token.  For every target layer
l in 1..L-1 the routing vector alpha assigns a real weight to every
(stream, target head, source) triple:

    stream   q | k | v   (per head) or r (the residual, shared across heads)
    source   s in 0..l   s = 0 is the token embedding, s >= 1 is the output
                         of layer s-1

One such triple is an **edge** of the routing graph.  s == l is the ordinary
sequential path (bias-initialised to 1.0, i.e. "read the layer below"); s < l
is a genuine **hyperconnection** that skips over layers.  A **circuit** is a
set of edges.  For L=12, H=16 there are 3773 edges in total, 3311 of them
hyperconnections.

Two independently editable channels produce the weights actually used:

    pred   FourWayPredictor(input_ids) -> alpha.  An external causal encoder
           that never sees the base model's activations, so alpha is a pure
           function of the context.  This is the "structure predictor" whose
           interpretability is the point of the exercise.
    corr   per-layer correction MLPs inside FourWayDAGFormer that read the
           previous layer's hidden state and add a per-token delta.

    alpha_eff = alpha_pred + delta_corr

Flat layout, canonical order (shared with scripts/interp_common.py):

    for l in 1..L-1: [q (H*n_src), k (H*n_src), v (H*n_src), r (n_src)]
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from interp_common import (  # noqa: E402  (path set up above)
    STREAMS, alpha_layout, flatten_alpha, load_elh, save_json, unflatten_alpha,
)

__all__ = [
    "STREAMS", "Edge", "RoutingLayout", "AlignedPrompt", "PromptSet",
    "Edit", "RoutingRunner", "load_models", "load_tokenizer",
    "build_prompt_set", "score_behavior", "capability_nll", "save_json",
    "bh_fdr", "signflip_null",
]


# ---------------------------------------------------------------------------
# Edges and the flat layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class Edge:
    """One routing coordinate: source -> (stream, target head)."""
    stream: str
    layer: int
    head: int      # -1 for the shared 'r' stream
    src: int

    @property
    def is_hyper(self) -> bool:
        """True for skip connections, False for the sequential layer-below path."""
        return self.src < self.layer

    @property
    def source_name(self) -> str:
        return "emb" if self.src == 0 else f"L{self.src - 1}"

    @property
    def target_name(self) -> str:
        return f"L{self.layer}/r" if self.head < 0 else f"L{self.layer}/h{self.head}"

    @property
    def name(self) -> str:
        return f"{self.stream}:{self.source_name}->{self.target_name}"

    def to_json(self) -> dict:
        return {"stream": self.stream, "layer": self.layer, "head": self.head,
                "src": self.src, "name": self.name, "is_hyper": self.is_hyper}


class RoutingLayout:
    """Index algebra over the flat routing vector.

    Holds the canonical edge list plus vectorised views of it so that masks
    over streams / layers / heads / sources are one-liners.
    """

    def __init__(self, num_layers: int, num_heads: int):
        self.L = int(num_layers)
        self.H = int(num_heads)
        self.labels = alpha_layout(self.L, self.H)
        self.edges: list[Edge] = [Edge(**lab) for lab in self.labels]
        self.D = len(self.edges)
        self.stream_arr = np.array([e.stream for e in self.edges])
        self.layer_arr = np.array([e.layer for e in self.edges], dtype=np.int64)
        self.head_arr = np.array([e.head for e in self.edges], dtype=np.int64)
        self.src_arr = np.array([e.src for e in self.edges], dtype=np.int64)
        self.hyper_arr = self.src_arr < self.layer_arr
        self._index = {e: i for i, e in enumerate(self.edges)}

        # [(start, end)] per target layer l = 1..L-1, matching the output
        # layout of FourWayDAGFormer.correction_mlps[l-1].
        self.chunks: list[tuple[int, int]] = []
        off = 0
        for l in range(1, self.L):
            sz = (3 * self.H + 1) * (l + 1)
            self.chunks.append((off, off + sz))
            off += sz
        assert off == self.D, f"chunk layout {off} != D {self.D}"

    def __len__(self) -> int:
        return self.D

    def index_of(self, edge: Edge) -> int:
        return self._index[edge]

    def mask(self, streams: Optional[Sequence[str]] = None,
             layers: Optional[Sequence[int]] = None,
             heads: Optional[Sequence[int]] = None,
             srcs: Optional[Sequence[int]] = None,
             hyper_only: bool = False) -> np.ndarray:
        """Boolean [D] mask selecting edges matching every supplied filter."""
        m = np.ones(self.D, dtype=bool)
        if streams is not None:
            m &= np.isin(self.stream_arr, list(streams))
        if layers is not None:
            m &= np.isin(self.layer_arr, list(layers))
        if heads is not None:
            m &= np.isin(self.head_arr, list(heads))
        if srcs is not None:
            m &= np.isin(self.src_arr, list(srcs))
        if hyper_only:
            m &= self.hyper_arr
        return m

    def names(self, idx: Iterable[int]) -> list[str]:
        return [self.edges[i].name for i in idx]

    def group_sums(self, values: np.ndarray, by: str) -> dict:
        """Sum |values| within groups of edges. `by` in {stream, layer, head, src}."""
        assert values.shape == (self.D,), f"expected [{self.D}], got {values.shape}"
        key = {"stream": self.stream_arr, "layer": self.layer_arr,
               "head": self.head_arr, "src": self.src_arr}[by]
        out = {}
        for k in np.unique(key):
            out[k.item() if hasattr(k, "item") else k] = float(
                np.abs(values[key == k]).sum())
        return out


# ---------------------------------------------------------------------------
# Prompt construction with content-token alignment
# ---------------------------------------------------------------------------

@dataclass
class AlignedPrompt:
    """One tokenised (instruction, content) prompt.

    `span` is the half-open content range in token positions. Every prompt in
    a PromptSet shares the same span, which is the whole point: the routing
    difference between polarities is then read off identical content tokens
    sitting at identical absolute positions.
    """
    item: int
    polarity: str
    variant: int
    ids: list[int]
    span: tuple[int, int]


@dataclass
class PromptSet:
    prompts: list[AlignedPrompt]
    span: tuple[int, int]
    prefix_len: int
    n_items: int
    variants: dict[str, int]
    filler_id: int

    def by(self, polarity: str, variant: int | None = None) -> list[AlignedPrompt]:
        return [p for p in self.prompts
                if p.polarity == polarity and (variant is None or p.variant == variant)]

    def get(self, item: int, polarity: str, variant: int) -> AlignedPrompt:
        for p in self.prompts:
            if p.item == item and p.polarity == polarity and p.variant == variant:
                return p
        raise KeyError(f"no prompt for item={item} polarity={polarity} variant={variant}")


def load_tokenizer(cfg: dict, tokenizer_path: str | None = None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tokenizer_path or cfg["tokenizer_id"])


def build_prompt_set(spec, tokenizer, polarities: Sequence[str] = ("pos", "neg", "neutral"),
                     max_items: int | None = None) -> PromptSet:
    """Tokenise a BehaviorSpec into position-aligned prompts.

    Instruction and content are tokenised separately and concatenated, so the
    content token ids are bit-identical across polarities (joint tokenisation
    could merge the boundary differently for different instructions). Short
    instructions are front-padded with newlines until every prefix has the
    same length, which puts the content at the same absolute positions in
    every condition — the predictor is position-aware, so without this the
    contrast would be confounded by a position shift.
    """
    filler = tokenizer("\n", add_special_tokens=False)["input_ids"]
    assert filler, "tokenizer produced no id for '\\n'; pass an explicit filler"
    filler_id = filler[-1]

    prefixes: dict[tuple[str, int], list[int]] = {}
    for pol in polarities:
        for vi, text in enumerate(spec.instructions[pol]):
            prefixes[(pol, vi)] = tokenizer(text, add_special_tokens=False)["input_ids"]
    prefix_len = max(len(v) for v in prefixes.values())
    padded = {k: [filler_id] * (prefix_len - len(v)) + v for k, v in prefixes.items()}

    items = spec.items if max_items is None else spec.items[:max_items]
    contents = [tokenizer(it.content, add_special_tokens=False)["input_ids"]
                for it in items]
    n_content = max(len(c) for c in contents)
    if min(len(c) for c in contents) != n_content:
        # Right-pad content so a single [T] edit/read mask covers every item.
        # Padding is on the right and attention is causal, so the padding can
        # never influence the tokens before it.
        contents = [c + [filler_id] * (n_content - len(c)) for c in contents]

    span = (prefix_len, prefix_len + n_content)
    prompts = []
    for i in range(len(items)):
        for pol in polarities:
            for vi in range(len(spec.instructions[pol])):
                prompts.append(AlignedPrompt(
                    item=i, polarity=pol, variant=vi,
                    ids=padded[(pol, vi)] + contents[i], span=span))
    assert all(len(p.ids) == span[1] for p in prompts), "ragged prompt lengths"
    return PromptSet(prompts=prompts, span=span, prefix_len=prefix_len,
                     n_items=len(items),
                     variants={p: len(spec.instructions[p]) for p in polarities},
                     filler_id=filler_id)


# ---------------------------------------------------------------------------
# Edits to the routing weights
# ---------------------------------------------------------------------------

@dataclass
class Edit:
    """An intervention on one routing channel, restricted to a circuit.

    ops, applied only where `mask` is set (and, if given, only at positions
    where `pos_mask` is set):

        add           x += lam * delta                  additive steering
        scale         x *= (1 + lam)                    enlarge the wiring
        signed_scale  x += lam * |x| * sign(delta)      enlarge along the contrast
        patch         x  = target                       activation patching
        zero          x  = 0                            knockout

    `target` may be a tensor or a callable ids -> [B, T, D]; the callable form
    lets `patch` read its donor values from a rerun of the same tokens under
    the other instruction, resolved fresh for every batch.
    """
    op: str
    mask: torch.Tensor                       # [D] bool
    lam: float = 1.0
    delta: Optional[torch.Tensor] = None     # [D] float
    target: Union[torch.Tensor, Callable[[torch.Tensor], torch.Tensor], None] = None
    pos_mask: Optional[torch.Tensor] = None  # [T] bool

    VALID_OPS = ("add", "scale", "signed_scale", "patch", "zero")

    def __post_init__(self) -> None:
        assert self.op in self.VALID_OPS, f"unknown op {self.op!r}"
        assert self.mask.dtype == torch.bool, "mask must be bool"
        if self.op in ("add", "signed_scale"):
            assert self.delta is not None, f"op {self.op} needs a delta"
        if self.op == "patch":
            assert self.target is not None, "op patch needs a target"

    def apply(self, x: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
        """Apply to x = [B, T, hi-lo], the flat slice covering coords [lo, hi)."""
        m = self.mask[lo:hi]
        if not bool(m.any()):
            return x
        sel = m.to(x.device).view(1, 1, -1)
        if self.pos_mask is not None:
            T = x.shape[1]
            pm = self.pos_mask.to(x.device)
            assert pm.shape[0] >= T, f"pos_mask {pm.shape[0]} shorter than T {T}"
            sel = sel & pm[:T].view(1, -1, 1)
        self_ = sel.to(x.dtype)

        if self.op == "add":
            d = self.delta[lo:hi].to(x.device, x.dtype).view(1, 1, -1)
            return x + self.lam * d * self_
        if self.op == "scale":
            return x * (1.0 + self.lam * self_)
        if self.op == "signed_scale":
            d = self.delta[lo:hi].to(x.device, x.dtype).view(1, 1, -1)
            return x + self.lam * x.abs() * torch.sign(d) * self_
        if self.op == "patch":
            t = self.target
            t = t[..., lo:hi] if t.dim() == 3 else t[lo:hi].view(1, 1, -1)
            return torch.where(sel, t.to(x.device, x.dtype).expand_as(x), x)
        if self.op == "zero":
            return torch.where(sel, torch.zeros_like(x), x)
        raise AssertionError(f"unhandled op {self.op}")


# ---------------------------------------------------------------------------
# Model loading and the edited forward pass
# ---------------------------------------------------------------------------

def load_models(config: str, ckpt: str, device: torch.device,
                need_base: bool = True, dtype: torch.dtype | None = None):
    """Load cfg + (fourway | None) + predictor.

    With need_base=False only the predictor is built, which skips the 2 GB
    base-model load entirely — enough for everything that touches the `pred`
    channel alone.
    """
    elh = load_elh()
    cfg = elh.load_config(config)
    if not need_base:
        from interp_common import build_predictor_from_cfg, load_predictor_state
        predictor = build_predictor_from_cfg(cfg, device)
        load_predictor_state(predictor, ckpt, elh.strip_prefixes)
        for p in predictor.parameters():
            p.requires_grad_(False)
        return cfg, None, predictor

    fourway, predictor = elh.load_fourway(ckpt, cfg, device)
    if dtype is not None:
        fourway = fourway.to(dtype=dtype)
        predictor = predictor.to(dtype=dtype)
    return cfg, fourway, predictor


class RoutingRunner:
    """Runs the model with optional edits to either routing channel.

    `pred` edits are applied to the predictor's output before it reaches the
    model; `corr` edits are applied via forward hooks on the correction MLPs.
    Both are addressed in the same flat [D] coordinate system.
    """

    def __init__(self, layout: RoutingLayout, predictor, fourway=None,
                 device: torch.device | None = None):
        self.layout = layout
        self.predictor = predictor
        self.fourway = fourway
        self.device = device or next(predictor.parameters()).device
        self.has_corr = fourway is not None and getattr(fourway, "use_local_correction", False)
        self._edit: dict[str, Optional[Edit]] = {"pred": None, "corr": None}
        self._resolved: dict[str, Optional[Edit]] = {}
        self._suspend = False
        self._capture: Optional[dict[int, torch.Tensor]] = None
        self._hooks: list = []
        if self.has_corr:
            self._register_corr_hooks()

    # -- hooks ------------------------------------------------------------
    def _register_corr_hooks(self) -> None:
        for i, mlp in enumerate(self.fourway.correction_mlps):
            lo, hi = self.layout.chunks[i]

            def hook(_m, _inp, out, i=i, lo=lo, hi=hi):
                if self._capture is not None and not self._suspend:
                    self._capture[i] = out.detach().float().cpu()
                e = self._active("corr")
                return out if e is None else e.apply(out, lo, hi)

            self._hooks.append(mlp.register_forward_hook(hook))

    def _active(self, channel: str) -> Optional[Edit]:
        """The edit in force for `channel`, with any callable target resolved."""
        if self._suspend:
            return None
        return self._resolved.get(channel, self._edit[channel])

    def _resolve_channel(self, channel: str, ids: torch.Tensor) -> None:
        """Materialise a callable edit target for this batch of ids.

        The donor callback usually reruns the model, so edits are suspended
        while it runs — otherwise the donor would be computed under the very
        intervention it is supposed to supply a clean reference for.
        """
        if self._suspend:
            return
        e = self._edit[channel]
        if e is None or not callable(e.target):
            return
        self._suspend = True
        try:
            tgt = e.target(ids)
        finally:
            self._suspend = False
        self._resolved[channel] = replace(e, target=tgt)

    def _resolve(self, ids: torch.Tensor) -> None:
        self._resolved = {}
        for channel in ("pred", "corr"):
            self._resolve_channel(channel, ids)

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # -- edits ------------------------------------------------------------
    def set_edit(self, channel: str, edit: Optional[Edit]) -> None:
        assert channel in ("pred", "corr"), channel
        if channel == "corr" and edit is not None and not self.has_corr:
            raise RuntimeError("this checkpoint has no correction MLPs to edit")
        self._edit[channel] = edit
        self._resolved = {}

    def clear_edits(self) -> None:
        self._edit = {"pred": None, "corr": None}
        self._resolved = {}

    @contextmanager
    def edited(self, pred: Optional[Edit] = None, corr: Optional[Edit] = None):
        old = dict(self._edit)
        self.set_edit("pred", pred)
        if corr is not None or self.has_corr:
            self.set_edit("corr", corr)
        try:
            yield self
        finally:
            self._edit = old
            self._resolved = {}

    # -- forward ----------------------------------------------------------
    @torch.no_grad()
    def routing(self, ids: torch.Tensor) -> dict[str, list[torch.Tensor]]:
        """Predictor routing dict for [B, T] ids, with the `pred` edit applied."""
        ids = ids.to(self.device)
        routing = self.predictor(ids)
        pending = self._edit["pred"]
        if pending is not None and callable(pending.target) and "pred" not in self._resolved:
            self._resolve_channel("pred", ids)
        e = self._active("pred")
        if e is None:
            return routing
        flat = flatten_alpha(routing).float()
        flat = e.apply(flat, 0, self.layout.D)
        return {s: [t.to(self.device) for t in v]
                for s, v in unflatten_alpha(flat, self.layout.L, self.layout.H).items()}

    @torch.no_grad()
    def alpha_pred(self, ids: torch.Tensor) -> torch.Tensor:
        """Predicted routing weights as a flat [B, T, D] cpu float32 tensor.

        Always the raw predictor output — edits are bypassed, so this is safe
        to call from inside a donor callback.
        """
        return flatten_alpha(self.predictor(ids.to(self.device))).float().cpu()

    @torch.no_grad()
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Logits [B, T, V]. Requires the base model."""
        assert self.fourway is not None, "base model not loaded (need_base=False)"
        ids = ids.to(self.device)
        self._resolve(ids)
        return self.fourway(ids, self.routing(ids))

    @torch.no_grad()
    def forward_capture(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits plus the correction-channel output as a flat [B, T, D] tensor."""
        assert self.has_corr, "no correction MLPs in this checkpoint"
        self._capture = {}
        try:
            logits = self.forward(ids)
            corr = torch.cat([self._capture[i] for i in range(len(self.layout.chunks))],
                             dim=-1)
        finally:
            self._capture = None
        assert corr.shape[-1] == self.layout.D, f"corr dim {corr.shape[-1]} != {self.layout.D}"
        return logits, corr


# ---------------------------------------------------------------------------
# Behaviour scoring
# ---------------------------------------------------------------------------

def _pad_batch(seqs: list[list[int]], filler: int) -> torch.Tensor:
    """Right-pad to a rectangle. Safe because every forward here is causal."""
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [filler] * (n - len(s)) for s in seqs], dtype=torch.long)


@torch.no_grad()
def _continuation_logprobs(runner: RoutingRunner, prompt_ids: list[list[int]],
                           cont_ids: list[list[int]], filler: int,
                           batch_size: int = 8) -> np.ndarray:
    """Mean per-token logprob of each continuation given its prompt."""
    seqs = [p + c for p, c in zip(prompt_ids, cont_ids)]
    out = np.zeros(len(seqs), dtype=np.float64)
    for i in range(0, len(seqs), batch_size):
        chunk = slice(i, min(i + batch_size, len(seqs)))
        ids = _pad_batch(seqs[chunk], filler)
        logits = runner.forward(ids).float()
        logprobs = torch.log_softmax(logits, dim=-1).cpu()
        for j, (p, c) in enumerate(zip(prompt_ids[chunk], cont_ids[chunk])):
            start = len(p) - 1
            tgt = torch.tensor(c, dtype=torch.long)
            lp = logprobs[j, start:start + len(c)].gather(-1, tgt.view(-1, 1))
            out[i + j] = lp.mean().item()
    return out


@torch.no_grad()
def _next_token_probs(runner: RoutingRunner, prompt_ids: list[list[int]],
                      filler: int, batch_size: int = 8) -> torch.Tensor:
    """Next-token distribution after each prompt, [n, V] on cpu."""
    outs = []
    for i in range(0, len(prompt_ids), batch_size):
        chunk = prompt_ids[i:i + batch_size]
        ids = _pad_batch(chunk, filler)
        logits = runner.forward(ids).float().cpu()
        outs.append(torch.stack([torch.softmax(logits[j, len(p) - 1], dim=-1)
                                 for j, p in enumerate(chunk)]))
    return torch.cat(outs, dim=0)


def first_token_ids(tokenizer, words: Sequence[str]) -> list[int]:
    """Ids of the first token of each word, deduplicated."""
    ids = []
    for w in words:
        t = tokenizer(w, add_special_tokens=False)["input_ids"]
        if t:
            ids.append(t[0])
    return sorted(set(ids))


@torch.no_grad()
def score_behavior(runner: RoutingRunner, spec, prompts: Sequence[AlignedPrompt],
                   tokenizer, filler: int, batch_size: int = 8) -> dict:
    """Score a behaviour on a list of prompts. Higher = more pos-behaviour.

    Returns the scalar `score` (the headline number), its standard error, and
    scorer-specific extras. Per-item scores are returned too so that arms can
    be compared with a paired test.
    """
    kind = spec.scorer["kind"]
    prompt_ids = [p.ids for p in prompts]

    if kind == "candidate_margin":
        plus, minus = spec.scorer["plus"], spec.scorer["minus"]
        cont = {k: [tokenizer(spec.items[p.item].candidates[k],
                              add_special_tokens=False)["input_ids"]
                    for p in prompts] for k in (plus, minus)}
        for k, seqs in cont.items():
            assert all(len(s) > 0 for s in seqs), f"empty continuation for {k!r}"
        lp_plus = _continuation_logprobs(runner, prompt_ids, cont[plus], filler, batch_size)
        lp_minus = _continuation_logprobs(runner, prompt_ids, cont[minus], filler, batch_size)
        per_item = lp_plus - lp_minus
        return {
            "score": float(per_item.mean()),
            "sem": float(per_item.std(ddof=1) / max(1, np.sqrt(len(per_item)))),
            "per_item": per_item.tolist(),
            "plus_rate": float((per_item > 0).mean()),
            "logp_plus": float(lp_plus.mean()),
            "logp_minus": float(lp_minus.mean()),
            "metric": f"mean logp({spec.scorer.get('plus_label', plus)}) - "
                      f"logp({spec.scorer.get('minus_label', minus)}) per token",
        }

    if kind == "token_set_margin":
        plus_ids = first_token_ids(tokenizer, spec.scorer["plus"])
        minus_ids = first_token_ids(tokenizer, spec.scorer["minus"])
        assert plus_ids and minus_ids, "empty token set"
        probs = _next_token_probs(runner, prompt_ids, filler, batch_size)
        mp = probs[:, plus_ids].sum(-1).clamp_min(1e-12)
        mm = probs[:, minus_ids].sum(-1).clamp_min(1e-12)
        per_item = (mp.log() - mm.log()).numpy().astype(np.float64)
        return {
            "score": float(per_item.mean()),
            "sem": float(per_item.std(ddof=1) / max(1, np.sqrt(len(per_item)))),
            "per_item": per_item.tolist(),
            "mass_plus": float(mp.mean()),
            "mass_minus": float(mm.mean()),
            "metric": f"log mass({spec.scorer.get('plus_label', 'plus')}) - "
                      f"log mass({spec.scorer.get('minus_label', 'minus')}) on next token",
        }

    raise AssertionError(f"unknown scorer kind {kind!r}")


# ---------------------------------------------------------------------------
# Capability side-effect check
# ---------------------------------------------------------------------------

def capability_windows(tokenizer, seq_len: int = 256, n_windows: int = 8,
                       cache: str | None = None) -> torch.Tensor:
    """Held-out text windows [N, seq_len] for the "did we break the model" check.

    Uses a cached eval batch file when one is given, otherwise falls back to
    repo-local prose, which is fine for a *relative* comparison across arms.
    """
    if cache:
        from interp_common import load_eval_ids
        ids, _ = load_eval_ids(cache)
        return ids[:n_windows, :seq_len]
    text = "\n\n".join(
        p.read_text(errors="ignore")
        for p in (REPO / "readme.md",
                  REPO / "experiments" / "METHODOLOGY_flops_loss_pareto.md")
        if p.is_file())
    toks = tokenizer(text, add_special_tokens=False)["input_ids"]
    n = min(n_windows, len(toks) // seq_len)
    assert n > 0, f"not enough repo text for one {seq_len}-token window"
    return torch.tensor([toks[i * seq_len:(i + 1) * seq_len] for i in range(n)],
                        dtype=torch.long)


@torch.no_grad()
def capability_nll(runner: RoutingRunner, windows: torch.Tensor,
                   batch_size: int = 2) -> float:
    """Mean next-token NLL over the windows, under whatever edit is active."""
    tot, n = 0.0, 0
    for i in range(0, windows.shape[0], batch_size):
        ids = windows[i:i + batch_size]
        logits = runner.forward(ids).float()
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                               ids[:, 1:].reshape(-1).to(logits.device))
        tot += loss.item() * ids.shape[0]
        n += ids.shape[0]
    return tot / n


# ---------------------------------------------------------------------------
# Statistics (scipy is not available in this environment)
# ---------------------------------------------------------------------------

def paired_t(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-coordinate mean and one-sample t statistic of x = [n_items, D]."""
    n = x.shape[0]
    assert n >= 2, "need at least 2 paired observations"
    mean = x.mean(0)
    sd = x.std(0, ddof=1)
    t = np.divide(mean, sd / np.sqrt(n), out=np.zeros_like(mean),
                  where=sd > 0)
    return mean, t


def signflip_null(x: np.ndarray, n_perm: int = 10000, seed: int = 0,
                  chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Exchangeability test for a paired contrast, without scipy.

    Under the null that the pos and neg conditions are interchangeable, the
    sign of each item's difference is arbitrary. Flipping signs leaves the
    per-coordinate sum of squares invariant, so the permuted t statistic is
    available in closed form from the permuted mean alone — which makes the
    whole null cheap to compute exactly.

    Returns:
        p:      [D] two-sided per-coordinate p-value
        maxnull:[n_perm] the null distribution of max_d |t_d|, for a
                family-wise threshold that needs no independence assumption
    """
    n, D = x.shape
    _, t_obs = paired_t(x)
    abs_obs = np.abs(t_obs)
    sumsq = (x ** 2).sum(0)
    rng = np.random.default_rng(seed)
    ge = np.zeros(D, dtype=np.int64)
    maxnull = np.empty(n_perm, dtype=np.float64)
    done = 0
    while done < n_perm:
        p = min(chunk, n_perm - done)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(p, n))
        m = (signs @ x) / n                                  # [p, D]
        var = np.maximum(sumsq - n * m ** 2, 0.0) / (n - 1)
        se = np.sqrt(var / n)
        tp = np.divide(m, se, out=np.zeros_like(m), where=se > 0)
        atp = np.abs(tp)
        ge += (atp >= abs_obs[None, :]).sum(0)
        maxnull[done:done + p] = atp.max(1)
        done += p
    return (ge + 1) / (n_perm + 1), maxnull


def bh_fdr(p: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, float]:
    """Benjamini-Hochberg. Returns (selected mask, p-value cutoff used)."""
    D = p.shape[0]
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, D + 1) / D)
    passing = np.nonzero(ranked <= thresh)[0]
    if passing.size == 0:
        return np.zeros(D, dtype=bool), 0.0
    cut = ranked[passing[-1]]
    return p <= cut, float(cut)


def gini(x: np.ndarray) -> float:
    """Concentration of |x| in [0, 1]; 0 = uniform, 1 = all mass on one coord."""
    v = np.sort(np.abs(x))
    n = v.shape[0]
    s = v.sum()
    if s <= 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(v) / (n * s))
