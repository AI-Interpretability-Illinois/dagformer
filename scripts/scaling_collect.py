"""Collect scaling-law inputs from the runs in experiments/scaling/runs.yaml.

For every manifest entry whose ``machine`` matches ``--machine`` and whose
save_dir exists locally, this records:

  * parameter counts: backbone total, backbone non-embedding, and routing
    extras (predictor + correction MLPs + v-norms), computed by instantiating
    the modules on the meta device from the run's config;
  * training FLOPs per token (forward+backward, 6x MACs) from the architecture,
    including the FourWay / modular source projections that dominate at scale;
  * the training-time held-out eval curve (step, tokens seen, eval NLL) from
    metrics.csv, and the final value; ``final_only`` entries (the shared
    models) get their curve from the common re-evaluation instead;
  * whether the run is finished (DONE marker or final checkpoint present).

Output: experiments/scaling/collected_<machine>.json, merged later by
scripts/scaling_fit.py. Run on each machine that holds runs:

    python scripts/scaling_collect.py --machine delta
    python scripts/scaling_collect.py --machine timan1      # on timan1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SEQ_LEN = 1024


def backbone_params(cfg: dict) -> dict:
    """OLMo-2 parameter counts (tied embeddings; no biases)."""
    V, D, L, I = cfg["vocab_size"], cfg["hidden_size"], cfg["num_hidden_layers"], cfg["intermediate_size"]
    per_layer = 4 * D * D + 2 * D + D + 3 * D * I + D      # attn + q/k norm + post-attn norm + mlp + post-ff norm
    non_embed = L * per_layer + D                            # + final norm
    return {"embed": V * D, "non_embed": non_embed, "total": V * D + non_embed}


def routing_params(cfg: dict) -> int | None:
    """Predictor + correction / v-norm parameters, instantiated on the meta device."""
    mode = str(cfg.get("routing_mode", ""))
    if not mode.startswith("fourway"):
        return 0
    L, H, V = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["vocab_size"]
    total = 0
    try:
        return _routing_params_meta(cfg, mode, L, H, V)
    except Exception as e:  # e.g. no transformers in this environment: routing params are informational only
        print(f"  (routing_params unavailable here: {type(e).__name__}: {str(e)[:80]})")
        return None


def _routing_params_meta(cfg: dict, mode: str, L: int, H: int, V: int) -> int:
    total = 0
    with torch.device("meta"):
        if mode.startswith("fourway_modular"):
            from src.model.modular_routing import FourWayModularPredictor
            pred = FourWayModularPredictor(
                vocab_size=V, encoder_dim=cfg.get("predictor_encoder_dim", 256),
                encoder_layers=cfg.get("predictor_encoder_layers", 2), encoder_heads=cfg.get("predictor_encoder_heads", 4),
                max_seq_len=cfg.get("predictor_max_seq_len", 4096), num_layers=L, num_heads=H,
                hidden_dim=cfg.get("fourway_hidden", 512))
        else:
            variant = cfg.get("fourway_predictor_variant", "encoder")
            from src.model.predictor import FourWayPerLayerPredictor, FourWayPredictor
            cls = FourWayPerLayerPredictor if variant == "per_layer" else FourWayPredictor
            pred = cls(vocab_size=V, encoder_dim=cfg.get("predictor_encoder_dim", 256),
                       encoder_layers=cfg.get("predictor_encoder_layers", 2), encoder_heads=cfg.get("predictor_encoder_heads", 4),
                       max_seq_len=cfg.get("predictor_max_seq_len", 4096), num_layers=L, num_heads=H,
                       hidden_dim=cfg.get("fourway_hidden", 512))
        total += sum(p.numel() for p in pred.parameters())
    D = cfg["hidden_size"]
    if mode.endswith("corrected"):
        hid = cfg.get("correction_hidden", 128)
        for l in range(1, L):
            n_src = (2 * l + 1) if mode.startswith("fourway_modular") else (l + 1)
            out = 3 * H * n_src + n_src + ((n_src + 1) if mode.startswith("fourway_modular") else 0)
            total += D * hid + hid * out
    if cfg.get("use_v_norm", False):
        total += (L - 1) * D
    return total


def train_flops_per_token(cfg: dict, seq_len: int = SEQ_LEN) -> dict:
    """6 x forward MACs per token: dense backbone + routing extras."""
    V, D, L, I, H = cfg["vocab_size"], cfg["hidden_size"], cfg["num_hidden_layers"], cfg["intermediate_size"], cfg["num_attention_heads"]
    attn_proj = 4 * D * D
    attn_scores = 2 * seq_len * D          # QK^T and AV, causal average ~T/2 each -> T total
    mlp = 3 * D * I
    dense = L * (attn_proj + attn_scores + mlp) + V * D
    mode = str(cfg.get("routing_mode", ""))
    extra = 0
    if mode.startswith("fourway_modular"):
        extra += sum(2 * l * 3 * D * D for l in range(1, L))          # (2l+1) source projections instead of 1
        extra += sum((2 * l + 1) * D * H * 3 + (2 * l + 2) * D + 0 for l in range(1, L))  # mixing einsums (q/k/v per head, m)
        extra += (2 * L + 1) * D                                        # read-out
    elif mode.startswith("fourway"):
        extra += sum(l * 3 * D * D for l in range(1, L))              # (l+1) source projections instead of 1
        extra += sum((l + 1) * (3 * H * (D // H) + D) for l in range(1, L))   # mixing
    if mode.startswith("fourway"):
        E = cfg.get("predictor_encoder_dim", 256)
        enc_layers = cfg.get("predictor_encoder_layers", 2)
        hid = cfg.get("fourway_hidden", 512)
        extra += enc_layers * (4 * E * E + 2 * E * 4 * E + 2 * seq_len * E) + E * hid   # predictor encoder + trunk
        n_out = sum((3 * H + 1) * (l + 1) for l in range(1, L))
        if mode.startswith("fourway_modular"):
            n_out = sum((3 * H + 1) * (2 * l + 1) + (2 * l + 2) for l in range(1, L)) + 2 + (2 * L + 1)
        extra += hid * n_out
        if mode.endswith("corrected"):
            ch = cfg.get("correction_hidden", 128)
            extra += sum(D * ch + ch * (3 * H + 1) * (l + 1) for l in range(1, L))
    return {"dense_macs": dense, "routing_macs": extra, "train_flops_per_token": 6 * (dense + extra)}


def read_curve(save_dir: str, eval_col: str, tokens_per_step: int) -> list[dict]:
    path = os.path.join(save_dir, "metrics.csv")
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row.get(eval_col, "")
            if v not in ("", None):
                step = int(row["step"])
                seen = row.get("train/tokens_seen_B", "")
                tokens = int(float(seen) * 1e9) if seen not in ("", None) else step * tokens_per_step
                out.append({"step": step, "tokens": tokens, "eval_nll": float(v)})
    return out


def is_done(save_dir: str, total_steps: int) -> bool:
    if os.path.exists(os.path.join(save_dir, "DONE")):
        return True
    return os.path.exists(os.path.join(save_dir, f"checkpoint_step{total_steps}.pt")) or \
        os.path.exists(os.path.join(save_dir, "checkpoint.pt"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="experiments/scaling/runs.yaml")
    p.add_argument("--machine", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    runs = yaml.safe_load(open(args.manifest))["runs"]
    collected = []
    for r in runs:
        if r["machine"] != args.machine:
            continue
        if not os.path.isdir(r["save_dir"]):
            print(f"skip (missing): {r['name']}")
            continue
        cfg_path = r["config"] if os.path.isabs(r["config"]) else os.path.join(os.path.dirname(__file__), "..", r["config"])
        if not os.path.exists(cfg_path):
            cfg_path = os.path.join(r["save_dir"], "config.yaml")
        cfg = yaml.safe_load(open(cfg_path))
        total_steps = int(r.get("steps_override") or cfg["total_steps"])
        entry = {**{k: v for k, v in r.items() if k != "config"},
                 "hidden_size": cfg["hidden_size"], "num_hidden_layers": cfg["num_hidden_layers"],
                 "routing_mode": cfg.get("routing_mode", "dense"), "total_steps": total_steps,
                 "total_tokens": total_steps * r["tokens_per_step"],
                 "params": backbone_params(cfg), "routing_params": routing_params(cfg),
                 "flops": train_flops_per_token(cfg), "done": is_done(r["save_dir"], total_steps)}
        entry["train_flops_total"] = float(entry["flops"]["train_flops_per_token"]) * entry["total_tokens"]  # float: exceeds int64 at 1B
        curve = read_curve(r["save_dir"], r["eval_col"], r["tokens_per_step"]) if r.get("eval_col") else []
        entry["curve"] = curve
        entry["final_eval_nll"] = curve[-1]["eval_nll"] if curve else None
        collected.append(entry)
        print(f"{r['name']:28s} N={entry['params']['total']/1e6:7.1f}M (non-embed {entry['params']['non_embed']/1e6:6.1f}M, "
              f"routing {(entry['routing_params'] or 0)/1e6:5.1f}M) flops/tok={entry['flops']['train_flops_per_token']/1e9:6.2f}G "
              f"tokens={entry['total_tokens']/1e9:5.2f}B done={entry['done']} evals={len(curve)} final={entry['final_eval_nll']}")
    out = args.out or f"experiments/scaling/collected_{args.machine}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"machine": args.machine, "runs": collected}, open(out, "w"), indent=1)
    print(f"wrote {out} ({len(collected)} runs)")


if __name__ == "__main__":
    main()
