"""Shared utilities for topology-predictor interpretability experiments.

The FourWay predictor is an *external* 2-layer causal encoder that maps
input_ids -> per-token routing weights (alpha). For a model with L layers and
H heads, each token gets, for every routed layer l in 1..L-1 (n_src = l+1
sources = embedding + l prior layer outputs):
    q/k/v: [H, n_src] each, r: [n_src]
Flattened canonical order (used everywhere in these scripts):
    for l in 1..L-1: [q (H*n_src), k (H*n_src), v (H*n_src), r (n_src)]
For L=12, H=16 this gives 49 * sum_{l=1}^{11}(l+1) = 49*77 = 3773 dims/token.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

STREAMS = ("q", "k", "v", "r")


def load_elh():
    """Import scripts/eval_lm_harness.py as a module (it is main-guarded)."""
    spec = importlib.util.spec_from_file_location(
        "eval_lm_harness", REPO / "scripts" / "eval_lm_harness.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Alpha dict <-> flat tensor
# ---------------------------------------------------------------------------

def alpha_layout(num_layers: int, num_heads: int) -> list[dict]:
    """Entry labels in canonical flatten order: one dict per flat dim."""
    labels = []
    for l in range(1, num_layers):
        n_src = l + 1
        for stream in ("q", "k", "v"):
            for h in range(num_heads):
                for s in range(n_src):
                    labels.append({"stream": stream, "layer": l, "head": h, "src": s})
        for s in range(n_src):
            labels.append({"stream": "r", "layer": l, "head": -1, "src": s})
    return labels


def alpha_flat_dim(num_layers: int, num_heads: int) -> int:
    return sum((3 * num_heads + 1) * (l + 1) for l in range(1, num_layers))


def flatten_alpha(routing: dict[str, list[torch.Tensor]]) -> torch.Tensor:
    """dict of per-layer tensors -> [B, T, D] in canonical order."""
    parts = []
    n_layers = len(routing["q"])
    for i in range(n_layers):
        B, T = routing["q"][i].shape[:2]
        for stream in ("q", "k", "v"):
            parts.append(routing[stream][i].reshape(B, T, -1))
        parts.append(routing["r"][i].reshape(B, T, -1))
    return torch.cat(parts, dim=-1)


def unflatten_alpha(flat: torch.Tensor, num_layers: int, num_heads: int
                    ) -> dict[str, list[torch.Tensor]]:
    """[B, T, D] -> dict of per-layer tensors (inverse of flatten_alpha)."""
    B, T, _ = flat.shape
    H = num_heads
    out: dict[str, list[torch.Tensor]] = {s: [] for s in STREAMS}
    off = 0
    for l in range(1, num_layers):
        n_src = l + 1
        for stream in ("q", "k", "v"):
            sz = H * n_src
            out[stream].append(flat[:, :, off:off + sz].reshape(B, T, H, n_src))
            off += sz
        out["r"].append(flat[:, :, off:off + n_src].reshape(B, T, n_src))
        off += n_src
    assert off == flat.shape[-1], f"layout mismatch: consumed {off} != {flat.shape[-1]}"
    return out


def identity_alpha(B: int, T: int, num_layers: int, num_heads: int,
                   device, dtype=torch.float32) -> dict[str, list[torch.Tensor]]:
    """Identity routing: [0,...,0,1] per stream (only most recent layer)."""
    out: dict[str, list[torch.Tensor]] = {s: [] for s in STREAMS}
    for l in range(1, num_layers):
        n_src = l + 1
        qkv = torch.zeros(B, T, num_heads, n_src, device=device, dtype=dtype)
        qkv[..., -1] = 1.0
        r = torch.zeros(B, T, n_src, device=device, dtype=dtype)
        r[..., -1] = 1.0
        for stream in ("q", "k", "v"):
            out[stream].append(qkv.clone())
        out["r"].append(r)
    return out


def mean_alpha(alpha_flats: list[torch.Tensor]) -> torch.Tensor:
    """Mean over all tokens of a list of [B,T,D] (or [T,D]) flats -> [D]."""
    total = None
    count = 0
    for a in alpha_flats:
        a2 = a.reshape(-1, a.shape[-1]).double()
        total = a2.sum(0) if total is None else total + a2.sum(0)
        count += a2.shape[0]
    return (total / count).float()


def broadcast_flat(mean_vec: torch.Tensor, B: int, T: int) -> torch.Tensor:
    return mean_vec.view(1, 1, -1).expand(B, T, -1).contiguous()


def replace_streams(dynamic: dict, replacement: dict,
                    streams: tuple[str, ...]) -> dict:
    """Return routing dict = dynamic with the given streams swapped out."""
    out: dict[str, list[torch.Tensor]] = {}
    for s in STREAMS:
        src = replacement if s in streams else dynamic
        out[s] = [t.clone() for t in src[s]]
    return out


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

def load_eval_ids(cache_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load cached held-out eval batches -> (ids [N,T], labels [N,T])."""
    batches = torch.load(cache_path, map_location="cpu", weights_only=False)
    ids = torch.cat([b["olmo_ids"] for b in batches], dim=0)
    labels = torch.cat([b["olmo_labels"] for b in batches], dim=0)
    return ids, labels


def synthetic_induction_ids(corpus_ids: torch.Tensor, n_seq: int, seq_len: int,
                            period: int, seed: int = 0
                            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat sequences + matched random controls.

    Token ids are sampled from the empirical marginal of corpus_ids so both
    conditions match natural token statistics. Returns (repeat_ids, random_ids),
    each [n_seq, seq_len].
    """
    g = torch.Generator().manual_seed(seed)
    flat = corpus_ids.reshape(-1)
    def sample(n):
        idx = torch.randint(0, flat.shape[0], (n,), generator=g)
        return flat[idx]
    reps = seq_len // period
    repeat = torch.stack([sample(period).repeat(reps)[:seq_len] for _ in range(n_seq)])
    random = torch.stack([sample(seq_len) for _ in range(n_seq)])
    return repeat, random


def domain_windows(tokenizer, seq_len: int, max_windows: int = 8
                   ) -> dict[str, torch.Tensor]:
    """Tokenize repo-local text of 3 distinct styles into fixed windows."""
    sources = {
        "code": sorted((REPO / "src").rglob("*.py")) + sorted((REPO / "scripts").glob("*.py")),
        # English-only prose files (results.md is mixed-language — excluded)
        "prose": [REPO / "readme.md",
                  REPO / "experiments" / "METHODOLOGY_flops_loss_pareto.md"],
        "latex": sorted((REPO / "experiments").glob("*.tex")),
    }
    out = {}
    for name, files in sources.items():
        text = "\n\n".join(p.read_text(errors="ignore") for p in files if p.is_file())
        toks = tokenizer(text, add_special_tokens=False)["input_ids"]
        n = min(max_windows, len(toks) // seq_len)
        if n == 0:
            print(f"[domain_windows] WARNING: not enough text for '{name}' "
                  f"({len(toks)} tokens < {seq_len}) — skipped")
            continue
        wins = [toks[i * seq_len:(i + 1) * seq_len] for i in range(n)]
        out[name] = torch.tensor(wins, dtype=torch.long)
        print(f"[domain_windows] {name}: {n} windows of {seq_len} tokens")
    return out


# ---------------------------------------------------------------------------
# Predictor loading (cheap path: no base model needed)
# ---------------------------------------------------------------------------

def build_predictor_from_cfg(cfg: dict, device):
    from src.model.predictor import FourWayPredictor
    pred = FourWayPredictor(
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
    ).to(device)
    pred.eval()
    return pred


def load_predictor_state(pred, ckpt_path: str, strip_prefixes_fn) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = strip_prefixes_fn(ckpt["predictor_state_dict"])
    missing, unexpected = pred.load_state_dict(state, strict=False)
    print(f"[load_predictor] {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")
    if len(missing) > 0:
        raise RuntimeError(f"predictor load missing keys: {missing[:5]}")
    del ckpt


@torch.no_grad()
def predict_alpha_flat(pred, ids: torch.Tensor, device, batch: int = 8) -> torch.Tensor:
    """Run predictor over [N,T] ids -> flat alpha [N,T,D] (cpu, fp16)."""
    outs = []
    for i in range(0, ids.shape[0], batch):
        chunk = ids[i:i + batch].to(device)
        routing = pred(chunk)
        outs.append(flatten_alpha(routing).half().cpu())
    return torch.cat(outs, dim=0)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"[saved] {path}")
