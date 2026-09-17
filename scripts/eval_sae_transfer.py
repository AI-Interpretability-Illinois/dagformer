"""Retest fixed historical SAE directions on a new natural-text corpus."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

from interp_common import load_elh, load_eval_ids
from interp_editing import layer_chunks, stream_slices


def permute_within_streams(direction, chunks, heads, seed):
    generator = torch.Generator().manual_seed(seed)
    result = direction.clone()
    for layer, (start, _) in enumerate(chunks, 1):
        slices, _ = stream_slices(layer, heads)
        for lo, hi in slices.values():
            values = direction[start + lo:start + hi]
            result[start + lo:start + hi] = values[torch.randperm(len(values), generator=generator)]
    return result


def permute_whole_heads(direction, chunks, heads, seed):
    """Swap Q/K/V head profiles together, preserving source identity and shared R."""
    generator = torch.Generator().manual_seed(seed)
    result = direction.clone()
    for layer, (start, _) in enumerate(chunks, 1):
        slices, n_sources = stream_slices(layer, heads)
        permutation = torch.randperm(heads, generator=generator)
        for stream in ("q", "k", "v"):
            lo, hi = slices[stream]
            profiles = direction[start + lo:start + hi].reshape(heads, n_sources)
            result[start + lo:start + hi] = profiles[permutation].reshape(-1)
    return result


def ratio_summary(delta, reference, mask, indexes):
    counts = mask.sum(1).astype(np.int64)
    sums = (delta * mask).sum(1)
    n = int(counts.sum())
    if n == 0:
        return {"n_tokens": 0, "n_windows": 0, "delta": None,
                "paired_bootstrap_95ci": None, "window_delta_sum": sums.tolist(),
                "window_token_count": counts.tolist()}
    boot_den = counts[indexes].sum(1)
    valid = boot_den > 0
    draws = sums[indexes[valid]].sum(1) / boot_den[valid]
    base = float((reference * mask).sum() / n)
    value = float(sums.sum() / n)
    return {"n_tokens": n, "n_windows": int((counts > 0).sum()),
            "reference_nll": base, "edited_nll": base + value, "delta": value,
            "paired_bootstrap_95ci": np.quantile(draws, [.025, .975]).tolist(),
            "valid_bootstrap_draws": int(valid.sum()),
            "window_delta_sum": sums.tolist(), "window_token_count": counts.tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--directions", required=True)
    ap.add_argument("--eval-cache", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-sequences", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--controls", type=int, default=5)
    ap.add_argument("--control-mode", choices=["coordinates", "whole-heads"], default="coordinates")
    ap.add_argument("--stream-decomposition", action="store_true", help="also evaluate shared R and Q/K/V separately")
    ap.add_argument("--alphas", nargs="+", type=float, default=[-4., 4.])
    ap.add_argument("--feature-ids", nargs="+", type=int)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    device = torch.device(args.device)
    elh = load_elh()
    cfg = elh.load_config(args.config)
    model, predictor = elh.load_fourway(args.ckpt, cfg, device)
    chunks = layer_chunks(cfg["num_hidden_layers"], cfg["num_attention_heads"])
    bundle = torch.load(args.directions, map_location="cpu", weights_only=False)
    features = [f for f in bundle["features"] if args.feature_ids is None or f["feature"] in args.feature_ids]
    ids, labels = load_eval_ids(args.eval_cache)
    ids, labels = ids[:args.n_sequences, :args.seq_len], labels[:args.n_sequences, :args.seq_len]
    label_array = labels.numpy()
    # The eight preselected historical directions all use explicit token sets.
    if any(f["rule"][0] != "tokset" for f in features):
        raise ValueError("This transfer run expects the saved explicit-token-set rules")
    state = {"parts": None}
    handles = []
    for index, mlp in enumerate(model.correction_mlps):
        def hook(module, inputs, output, layer=index):
            return output if state["parts"] is None else output + state["parts"][layer].to(output.dtype)
        handles.append(mlp.register_forward_hook(hook))

    @torch.inference_mode()
    def losses():
        result = []
        for start in range(0, len(ids), args.batch_size):
            rows = ids[start:start + args.batch_size].to(device)
            logits = model(rows, predictor(rows)).float()
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   labels[start:start + args.batch_size].to(device).reshape(-1),
                                   reduction="none")
            result.append(loss.reshape(rows.shape).cpu())
        return torch.cat(result).numpy().astype(np.float64)

    def write(payload):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(args.out)

    try:
        reference = losses()
        indexes = np.random.default_rng(args.seed).integers(len(ids), size=(args.draws, len(ids)))
        control_protocol = (
            f"{args.controls} coordinate permutations within each layer and Q/K/V/R stream; each preserves the coefficient multiset and direction norm"
            if args.control_mode == "coordinates" else
            f"{args.controls} whole-head permutations within each layer; the same permutation acts on Q/K/V, source positions are preserved, and shared R remains unchanged; each preserves direction norm and per-stream/source head means")
        result = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  "git_commit": commit, "versions": {k: importlib.metadata.version(k) for k in ("torch", "transformers", "numpy")},
                  "source_sae": bundle["source"], "selection_source": bundle["selection_source"],
                  "reference_mean_nll": float(reference.mean()),
                  "reference_window_mean_nll": reference.mean(1).tolist(),
                  "protocol": {"selection": f"{len(features)} fixed features from the historical screen_generation.json; no new selection",
                               "edit": bundle["direction_units"] + "; additive correction-channel edit on all tokens",
                               "controls": control_protocol,
                               "stream_decomposition": "R-only and Q/K/V-only additive edits are evaluated separately" if args.stream_decomposition else None,
                               "uncertainty": "paired window bootstrap of token-weighted loss differences; fixed checkpoint; unadjusted intervals",
                               "corpus": "new WikiText test windows, separate from the original Dolma feature-discovery corpus"},
                  "features": {}, "complete": False}
        for feature in features:
            direction = feature["direction"].float()
            if direction.numel() != chunks[-1][1]:
                raise ValueError("SAE direction and model routing dimensions differ")
            mask = np.isin(label_array, feature["rule"][1])
            rec = {k: v for k, v in feature.items() if k != "direction"}
            rec["arms"] = {}
            variants = [("feature", direction)]
            permute = permute_within_streams if args.control_mode == "coordinates" else permute_whole_heads
            variants += [(f"random{i}", permute(direction, chunks, cfg["num_attention_heads"],
                                                args.seed + 100 * feature["feature"] + i))
                         for i in range(args.controls)]
            if args.stream_decomposition:
                residual = torch.zeros_like(direction)
                for layer, (start, _) in enumerate(chunks, 1):
                    slices, _ = stream_slices(layer, cfg["num_attention_heads"])
                    lo, hi = slices["r"]
                    residual[start + lo:start + hi] = direction[start + lo:start + hi]
                variants += [("r_only", residual), ("qkv_only", direction - residual)]
            for name, vector in variants:
                for alpha in args.alphas:
                    delta = (alpha * vector).to(device)
                    state["parts"] = [delta[a:b] for a, b in chunks]
                    difference = losses() - reference
                    arm = {"alpha": alpha, "direction_l2": float(vector.norm()),
                           "on_rule": ratio_summary(difference, reference, mask, indexes),
                           "off_rule": ratio_summary(difference, reference, ~mask, indexes),
                           "overall": ratio_summary(difference, reference, np.ones_like(mask), indexes)}
                    if name in ("feature", "r_only", "qkv_only"):
                        arm["by_token"] = {}
                        for token_id in feature["rule"][1]:
                            token_stats = ratio_summary(difference, reference, label_array == token_id, indexes)
                            arm["by_token"][str(token_id)] = {
                                k: v for k, v in token_stats.items()
                                if k not in ("window_delta_sum", "window_token_count")}
                    rec["arms"][f"{name}/alpha{alpha:g}"] = arm
                    print(f"f{feature['feature']}/{name}/alpha{alpha:g}",
                          {k: arm[k]["delta"] for k in ("on_rule", "off_rule", "overall")}, flush=True)
            result["features"][str(feature["feature"])] = rec
            write(result)
        result["complete"] = True
        write(result)
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
