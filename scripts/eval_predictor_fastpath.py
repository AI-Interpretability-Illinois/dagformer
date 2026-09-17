"""Measure the old MHA-fastpath patch's numerical effect on a trained model."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

from eval_context_fidelity import paired_summary
from interp_common import flatten_alpha, load_elh, load_eval_ids


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-sequences", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    device = torch.device(args.device)
    elh = load_elh()
    model, predictor = elh.load_fourway(args.ckpt, elh.load_config(args.config), device)
    ids, labels = load_eval_ids(args.eval_cache)
    ids, labels = ids[:args.n_sequences, :args.seq_len], labels[:args.n_sequences, :args.seq_len]
    losses = {"enabled": [], "disabled": []}
    calls = {"enabled": 0, "disabled": 0}
    differences = []
    original_setting = torch.backends.mha.get_fastpath_enabled()
    original_fused = torch._transformer_encoder_layer_fwd
    current = "enabled"

    def count_fused(*inputs, **kwargs):
        calls[current] += 1
        return original_fused(*inputs, **kwargs)

    # Count actual fused-encoder calls so a disabled fastpath is observable.
    torch._transformer_encoder_layer_fwd = count_fused
    try:
        with torch.inference_mode():
            for row, target in zip(ids, labels):
                values = {}
                for enabled in (True, False):
                    current = "enabled" if enabled else "disabled"
                    torch.backends.mha.set_fastpath_enabled(enabled)
                    routing = predictor(row[None].to(device))
                    logits = model(row[None].to(device), routing).float()
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.to(device))
                    losses[current].append(float(loss))
                    values[current] = (flatten_alpha(routing).float(), logits)
                a, x = values["enabled"]
                b, y = values["disabled"]
                differences.append({"alpha_max_abs": float((b - a).abs().max()),
                                    "alpha_relative_l2": float((b - a).norm() / a.norm().clamp_min(1e-12)),
                                    "logits_max_abs": float((y - x).abs().max())})
    finally:
        torch._transformer_encoder_layer_fwd = original_fused
        torch.backends.mha.set_fastpath_enabled(original_setting)
    result = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "git_commit": commit,
              "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers")},
              "n_sequences": len(ids), "actual_fused_encoder_calls": calls,
              "nll": losses, "per_sequence_differences": differences,
              "disable_minus_enable_nll": paired_summary(losses["disabled"], losses["enabled"]),
              "protocol": "same weights and token windows; inference mode; only torch.backends.mha fastpath setting changes; no model defaults are changed"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(result["disable_minus_enable_nll"], calls, flush=True)


if __name__ == "__main__":
    main()
