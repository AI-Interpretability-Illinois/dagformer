"""Dump per-token routing weights (alpha) from FourWay predictor checkpoints.

The predictor is external and only sees input_ids, so alpha extraction does not
require the base model. Dumps:
  - main steps (default 9000, 10500): full fp16 alpha for eval / synthetic /
    domain corpora
  - all other steps: eval-corpus alpha subsampled (stride 2) for emergence
    curves, plus summary stats (identity distance, per-entry std)

Usage:
  python3 scripts/interp_dump_alpha.py --config configs/fourway_300m_dagformer_mmap.yaml \
      --ckpt-dir checkpoints/fourway_300m_dagformer_mmap --out experiments/results/interp
"""
from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import torch

from interp_common import (REPO, alpha_flat_dim, alpha_layout,
                           build_predictor_from_cfg, domain_windows,
                           identity_alpha, flatten_alpha, load_elh,
                           load_eval_ids, load_predictor_state,
                           predict_alpha_flat, save_json,
                           synthetic_induction_ids)


def summary_stats(alpha: torch.Tensor, ident: torch.Tensor) -> dict:
    """alpha [N,T,D] fp16, ident [D] -> summary dict."""
    a = alpha.reshape(-1, alpha.shape[-1]).float()
    dist = (a - ident.view(1, -1)).abs().mean().item()
    per_entry_std = a.std(dim=0)
    return {
        "mean_abs_dist_to_identity": dist,
        "mean_entry_std_over_tokens": per_entry_std.mean().item(),
        "max_entry_std_over_tokens": per_entry_std.max().item(),
        "frac_entries_std_gt_0.05": (per_entry_std > 0.05).float().mean().item(),
        "frac_entries_std_gt_0.2": (per_entry_std > 0.2).float().mean().item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--out", default="experiments/results/interp")
    ap.add_argument("--main-steps", default="9000,10500")
    ap.add_argument("--eval-cache", default=None,
                    help="defaults to <ckpt-dir>/eval_cache.pt")
    ap.add_argument("--seq-len", type=int, default=1024)
    args = ap.parse_args()

    elh = load_elh()
    cfg = elh.load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    D = alpha_flat_dim(L, H)
    out_dir = Path(args.out) / "alpha_dumps"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[dump] L={L} H={H} alpha_dim={D} device={device}")

    # Layout labels (once)
    save_json(alpha_layout(L, H), out_dir / "layout.json")

    # Corpora
    cache = args.eval_cache or str(Path(args.ckpt_dir) / "eval_cache.pt")
    eval_ids, eval_labels = load_eval_ids(cache)
    print(f"[dump] eval corpus: {tuple(eval_ids.shape)}")
    rep_ids, rand_ids = synthetic_induction_ids(eval_ids, n_seq=32,
                                                seq_len=args.seq_len, period=128)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])
    domains = domain_windows(tok, args.seq_len)

    torch.save({"eval_ids": eval_ids, "eval_labels": eval_labels,
                "repeat_ids": rep_ids, "random_ids": rand_ids,
                "domains": domains}, out_dir / "corpora.pt")

    # Identity reference vector [D]
    ident = flatten_alpha(identity_alpha(1, 1, L, H, "cpu"))[0, 0]

    # Discover checkpoints
    ckpts = {}
    for p in sorted(Path(args.ckpt_dir).glob("checkpoint_step*.pt")):
        m = re.match(r"checkpoint_step(\d+)\.pt$", p.name)
        if m:
            ckpts[int(m.group(1))] = p
    main_steps = [int(s) for s in args.main_steps.split(",") if int(s) in ckpts]
    print(f"[dump] checkpoints found: {sorted(ckpts)}; main={main_steps}")

    pred = build_predictor_from_cfg(cfg, device)
    all_summaries = {}
    for step in sorted(ckpts):
        load_predictor_state(pred, str(ckpts[step]), elh.strip_prefixes)
        a_eval = predict_alpha_flat(pred, eval_ids, device)
        all_summaries[step] = summary_stats(a_eval, ident)
        print(f"[dump] step {step}: {all_summaries[step]}")

        if step in main_steps:
            torch.save(a_eval, out_dir / f"alpha_eval_step{step}.pt")
            torch.save(predict_alpha_flat(pred, rep_ids, device),
                       out_dir / f"alpha_repeat_step{step}.pt")
            torch.save(predict_alpha_flat(pred, rand_ids, device),
                       out_dir / f"alpha_random_step{step}.pt")
            for name, ids in domains.items():
                torch.save(predict_alpha_flat(pred, ids, device),
                           out_dir / f"alpha_domain_{name}_step{step}.pt")
        else:
            torch.save(a_eval[:, ::2].clone(), out_dir / f"alpha_eval_sub_step{step}.pt")
        del a_eval
        gc.collect()

    save_json(all_summaries, out_dir / "summaries_by_step.json")
    print("[dump] DONE")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
