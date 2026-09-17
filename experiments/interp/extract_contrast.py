"""Step 1 — read the routing weights under a RepE-style behavioural contrast.

The same content tokens are run once per instruction polarity, and the
routing weights are averaged over the (shared, position-aligned) content
span.  The output is a per-prompt matrix of routing vectors that step 2
differences.

Also — and this runs first, because it can invalidate everything downstream —
the behaviour itself is measured under each polarity with no intervention.
That pos-minus-neg gap is the headroom: a circuit can only be *causal for the
behaviour* to the extent the instruction moves the behaviour in the first
place.  If the gap is ~0 the model is not following the instruction, and any
difference found in the routing weights is a signature of reading the
instruction, not of doing the thing.

Usage:
    python experiments/interp/extract_contrast.py \
        --config /work/hdd/bfqt/shared/dagformer-models/300m-dagformer/config.yaml \
        --ckpt   /work/hdd/bfqt/shared/dagformer-models/300m-dagformer/checkpoint.pt \
        --behavior honesty --channel both
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from behaviors import get_behavior
from circuit_common import (RoutingLayout, RoutingRunner, build_prompt_set,
                            load_models, load_tokenizer, save_json, score_behavior)

DEFAULT_TOKENIZER = "/work/hdd/bfqt/shared/dagformer-models/tokenizer"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="training config yaml")
    ap.add_argument("--ckpt", required=True, help="checkpoint.pt")
    ap.add_argument("--behavior", default="honesty",
                    help="registry name or path to a behaviour JSON file")
    ap.add_argument("--channel", default="both", choices=("pred", "corr", "both"),
                    help="which routing channel to record")
    ap.add_argument("--out-dir", default="experiments/results/interp/circuits")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help="local tokenizer dir; falls back to the config's id")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp32"))
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--read-last-k", type=int, default=0,
                    help="average alpha over the last k content tokens "
                         "(0 = the whole content span)")
    ap.add_argument("--no-behavior-score", action="store_true",
                    help="skip the instruction-sensitivity check (no base model "
                         "is loaded when the channel is 'pred')")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = {"auto": None, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    if args.dtype == "auto" and device.type == "cpu":
        dtype = torch.float32

    spec = get_behavior(args.behavior)
    want_corr = args.channel in ("corr", "both")
    need_base = want_corr or not args.no_behavior_score
    print(f"[extract] behaviour={spec.name} channel={args.channel} "
          f"base_model={'yes' if need_base else 'no'} device={device}")

    cfg, fourway, predictor = load_models(args.config, args.ckpt, device,
                                          need_base=need_base, dtype=dtype)
    tokenizer = load_tokenizer(cfg, args.tokenizer if Path(args.tokenizer).is_dir() else None)
    layout = RoutingLayout(cfg["num_hidden_layers"], cfg["num_attention_heads"])
    runner = RoutingRunner(layout, predictor, fourway, device)
    if want_corr and not runner.has_corr:
        raise SystemExit("--channel asks for 'corr' but this checkpoint has no "
                         "correction MLPs; rerun with --channel pred")

    ps = build_prompt_set(spec, tokenizer, max_items=args.max_items)
    lo, hi = ps.span
    read_lo = hi - args.read_last_k if args.read_last_k else lo
    print(f"[extract] {len(ps.prompts)} prompts, T={hi}, prefix={ps.prefix_len}, "
          f"content span=[{lo},{hi}), alpha read over [{read_lo},{hi}) "
          f"({ps.n_items} items x {ps.variants})")

    ids = torch.tensor([p.ids for p in ps.prompts], dtype=torch.long)
    meta = [{"item": p.item, "polarity": p.polarity, "variant": p.variant}
            for p in ps.prompts]

    # --- routing weights, averaged over the aligned content span ---------
    t0 = time.time()
    alpha: dict[str, torch.Tensor] = {}
    if args.channel in ("pred", "both"):
        rows = []
        for i in range(0, ids.shape[0], args.batch_size):
            flat = runner.alpha_pred(ids[i:i + args.batch_size])       # [B, T, D]
            rows.append(flat[:, read_lo:hi].mean(1))
        alpha["pred"] = torch.cat(rows, 0)
        print(f"[extract] alpha_pred {tuple(alpha['pred'].shape)} "
              f"({time.time() - t0:.1f}s)")
    if want_corr:
        rows = []
        for i in range(0, ids.shape[0], args.batch_size):
            _, corr = runner.forward_capture(ids[i:i + args.batch_size])
            rows.append(corr[:, read_lo:hi].mean(1))
        alpha["corr"] = torch.cat(rows, 0)
        print(f"[extract] alpha_corr {tuple(alpha['corr'].shape)} "
              f"({time.time() - t0:.1f}s)")
    if "pred" in alpha and "corr" in alpha:
        alpha["eff"] = alpha["pred"] + alpha["corr"]

    # --- instruction sensitivity: does the behaviour actually move? ------
    scores: dict[str, dict] = {}
    if not args.no_behavior_score:
        runner.clear_edits()
        for pol in ("pos", "neg", "neutral"):
            sub = ps.by(pol)
            scores[pol] = score_behavior(runner, spec, sub, tokenizer,
                                         ps.filler_id, args.batch_size)
            print(f"[extract] behaviour[{pol:7s}] = {scores[pol]['score']:+.4f} "
                  f"(sem {scores[pol]['sem']:.4f})")
        gap = scores["pos"]["score"] - scores["neg"]["score"]
        # Paired over items: prompts are emitted item-major, so reshaping to
        # [n_items, n_variants] and averaging over variants gives one score
        # per item in the same order for every polarity.
        a = np.array(scores["pos"]["per_item"]).reshape(-1, ps.variants["pos"]).mean(1)
        b = np.array(scores["neg"]["per_item"]).reshape(-1, ps.variants["neg"]).mean(1)
        d = a - b
        tstat = float(d.mean() / (d.std(ddof=1) / np.sqrt(d.shape[0]))) if d.std() > 0 else 0.0
        scores["gap"] = {"pos_minus_neg": float(gap), "paired_t": tstat,
                         "n_items": int(d.shape[0]),
                         "metric": scores["pos"]["metric"]}
        verdict = ("USABLE" if abs(tstat) >= 2.0 else
                   "WEAK — the instruction barely moves the behaviour; treat any "
                   "circuit below as a circuit for reading the instruction")
        print(f"[extract] behaviour gap pos-neg = {gap:+.4f} (paired t={tstat:+.2f}) "
              f"-> {verdict}")
        scores["gap"]["verdict"] = verdict

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"contrast_{spec.name}"
    payload = {
        "behavior": spec.to_json(),
        "channel": args.channel,
        "layout": {"L": layout.L, "H": layout.H, "D": layout.D},
        "span": [lo, hi], "read_span": [read_lo, hi], "prefix_len": ps.prefix_len,
        "variants": ps.variants, "n_items": ps.n_items,
        "filler_id": ps.filler_id,
        "meta": meta, "ids": ids, "alpha": alpha,
        "behavior_scores": scores,
        "ckpt": args.ckpt, "config": args.config,
    }
    torch.save(payload, out_dir / f"{stem}.pt")
    print(f"[saved] {out_dir / f'{stem}.pt'}")
    save_json({k: v for k, v in payload.items()
               if k not in ("meta", "ids", "alpha", "behavior")},
              out_dir / f"{stem}_summary.json")
    runner.close()
    print("[extract] DONE")


if __name__ == "__main__":
    main()
