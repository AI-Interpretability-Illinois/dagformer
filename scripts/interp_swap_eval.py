"""Causal necessity ladder: evaluate a trained FourWay model under routing
interventions, decomposing the NLL gain into (avg rewiring) + (per-token
dynamism) + (local corrections) + (content alignment).

Modes (all on the same trained weights, eval-only):
  dynamic            alpha_pred(context) + correction MLPs (as trained)
  dynamic_no_corr    external alpha only, corrections zeroed
  static_mean        alpha_pred replaced by calib-set mean (constant over
                     tokens & contexts), corrections ON
  static_mean_no_corr  ... and corrections zeroed
  identity           alpha=[0..0,1] (vanilla wiring), corrections ON
  identity_no_corr   vanilla wiring, corrections zeroed
  shuffle_time       per-seq random permutation of alpha along T
  swap_context       alpha computed from a different sequence
  static_q/k/v/r     one stream staticized to calib mean, rest dynamic

Also: synthetic induction copy-accuracy (repeat vs random sequences) under
dynamic / static_mean / identity_no_corr, plus a dense-baseline reference.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from interp_common import (STREAMS, alpha_layout, broadcast_flat,
                           identity_alpha, flatten_alpha, load_elh,
                           load_eval_ids, mean_alpha, save_json,
                           synthetic_induction_ids, unflatten_alpha)


@torch.no_grad()
def model_nll(fourway, ids_row: torch.Tensor, labels_row: torch.Tensor,
              routing: dict, use_corr: bool, device) -> torch.Tensor:
    """Per-token NLL [T] for one sequence under a routing dict."""
    fourway.use_local_correction = use_corr and fourway_has_corr(fourway)
    routing_dev = {s: [t.to(device) for t in routing[s]] for s in STREAMS}
    logits = fourway(ids_row.to(device), routing_dev)
    nll = F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                          labels_row.to(device).view(-1), reduction="none")
    return nll.view(-1).cpu()


def fourway_has_corr(fourway) -> bool:
    return hasattr(fourway, "correction_mlps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dense-config", default=None)
    ap.add_argument("--dense-ckpt", default=None)
    ap.add_argument("--eval-cache", default=None)
    ap.add_argument("--out", default="experiments/results/interp")
    ap.add_argument("--n-calib", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    elh = load_elh()
    cfg = elh.load_config(args.config)
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fourway, predictor = elh.load_fourway(args.ckpt, cfg, device)
    orig_corr = fourway.use_local_correction

    cache = args.eval_cache or str(Path(args.ckpt).parent / "eval_cache.pt")
    ids, labels = load_eval_ids(cache)
    n = ids.shape[0]
    n_calib = min(args.n_calib, n // 2)
    calib_ids, test_ids = ids[:n_calib], ids[n_calib:]
    test_labels = labels[n_calib:]
    n_test, T = test_ids.shape
    print(f"[swap] calib={n_calib} test={n_test} T={T} corr_trained={orig_corr}")

    # --- Predictor alpha for calib + test (per sequence, flat fp32 on CPU) ---
    @torch.no_grad()
    def alpha_flat_rows(rows: torch.Tensor) -> list[torch.Tensor]:
        return [flatten_alpha(predictor(rows[i:i + 1].to(device))).float().cpu()
                for i in range(rows.shape[0])]

    calib_alpha = alpha_flat_rows(calib_ids)
    test_alpha = alpha_flat_rows(test_ids)
    mean_vec = mean_alpha(calib_alpha)          # [D]
    ident_vec = flatten_alpha(identity_alpha(1, 1, L, H, "cpu"))[0, 0]
    print(f"[swap] mean|alpha_mean - identity| = "
          f"{(mean_vec - ident_vec).abs().mean().item():.4f}")

    layout = alpha_layout(L, H)
    stream_masks = {s: torch.tensor([lab["stream"] == s for lab in layout])
                    for s in STREAMS}

    # --- Mode definitions: flat-alpha transform + corrections flag ---
    g = torch.Generator().manual_seed(args.seed)
    perms = [torch.randperm(T, generator=g) for _ in range(n_test)]

    def make_modes():
        modes = {}
        modes["dynamic"] = (lambda i: test_alpha[i], True)
        modes["dynamic_no_corr"] = (lambda i: test_alpha[i], False)
        modes["static_mean"] = (lambda i: broadcast_flat(mean_vec, 1, T), True)
        modes["static_mean_no_corr"] = (lambda i: broadcast_flat(mean_vec, 1, T), False)
        modes["identity"] = (lambda i: broadcast_flat(ident_vec, 1, T), True)
        modes["identity_no_corr"] = (lambda i: broadcast_flat(ident_vec, 1, T), False)
        modes["shuffle_time"] = (lambda i: test_alpha[i][:, perms[i], :], True)
        modes["swap_context"] = (lambda i: test_alpha[(i + 1) % n_test], True)
        for s in STREAMS:
            def repl(i, s=s):
                a = test_alpha[i].clone()
                a[:, :, stream_masks[s]] = broadcast_flat(
                    mean_vec, 1, T)[:, :, stream_masks[s]]
                return a
            modes[f"static_{s}"] = (repl, True)
        return modes

    results = {}
    per_token_saved = {}
    for name, (fn, use_corr) in make_modes().items():
        nlls = []
        for i in range(n_test):
            routing = unflatten_alpha(fn(i), L, H)
            nlls.append(model_nll(fourway, test_ids[i:i + 1],
                                  test_labels[i:i + 1], routing, use_corr, device))
        nll_all = torch.stack(nlls)              # [n_test, T]
        results[name] = nll_all.mean().item()
        if name in ("dynamic", "static_mean", "identity_no_corr"):
            per_token_saved[name] = nll_all
        print(f"[swap] {name:22s} NLL = {results[name]:.4f}")
    fourway.use_local_correction = orig_corr

    # --- Dense reference on the same test sequences ---
    if args.dense_ckpt and args.dense_config:
        dense_cfg = elh.load_config(args.dense_config)
        dense = elh.load_dense(args.dense_ckpt, dense_cfg, device)
        dense.eval()
        with torch.no_grad():
            nlls = []
            for i in range(n_test):
                lg = dense(test_ids[i:i + 1].to(device)).logits
                nlls.append(F.cross_entropy(
                    lg.float().view(-1, lg.shape[-1]),
                    test_labels[i:i + 1].to(device).view(-1),
                    reduction="none").cpu())
            dense_nll = torch.stack(nlls)
        results["dense_baseline"] = dense_nll.mean().item()
        per_token_saved["dense_baseline"] = dense_nll
        print(f"[swap] {'dense_baseline':22s} NLL = {results['dense_baseline']:.4f}")

    # --- Synthetic induction: copy accuracy under interventions ---
    rep_ids, rand_ids = synthetic_induction_ids(ids, n_seq=32, seq_len=T,
                                                period=128, seed=args.seed)
    copy_mask = torch.zeros(T - 1, dtype=torch.bool)
    copy_mask[127:] = True  # label position t predicts ids[t+1]; copyable iff t+1>=128

    @torch.no_grad()
    def synth_metrics(rows: torch.Tensor, mode: str) -> dict:
        accs, nlls = [], []
        for i in range(rows.shape[0]):
            row = rows[i:i + 1]
            lab = row[:, 1:]
            if mode == "dense":
                lg = dense(row.to(device)).logits[:, :-1]
            else:
                a = flatten_alpha(predictor(row.to(device))).float().cpu()
                if mode == "static_mean":
                    a = broadcast_flat(mean_vec, 1, T)
                elif mode == "identity_no_corr":
                    a = broadcast_flat(ident_vec, 1, T)
                use_corr = mode != "identity_no_corr"
                fourway.use_local_correction = use_corr and fourway_has_corr(fourway)
                routing = {s: [t.to(device) for t in v] for s, v in
                           unflatten_alpha(a, L, H).items()}
                lg = fourway(row.to(device), routing)[:, :-1]
            pred_tok = lg.argmax(-1).cpu()
            nll = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]),
                                  lab.to(device).reshape(-1), reduction="none"
                                  ).cpu().view(-1)
            accs.append((pred_tok.view(-1) == lab.view(-1))[copy_mask].float().mean())
            nlls.append(nll[copy_mask].mean())
        return {"copy_acc": torch.stack(accs).mean().item(),
                "copy_nll": torch.stack(nlls).mean().item()}

    synth = {}
    synth_modes = ["dynamic", "static_mean", "identity_no_corr"]
    if args.dense_ckpt and args.dense_config:
        synth_modes.append("dense")
    for m in synth_modes:
        synth[f"repeat/{m}"] = synth_metrics(rep_ids, m)
        synth[f"random/{m}"] = synth_metrics(rand_ids, m)
        print(f"[synth] {m:18s} repeat: {synth[f'repeat/{m}']}  "
              f"random: {synth[f'random/{m}']}")
    fourway.use_local_correction = orig_corr

    step_tag = Path(args.ckpt).stem.replace("checkpoint_", "")
    save_json({"modes_nll": results, "synthetic": synth,
               "n_calib": n_calib, "n_test": n_test,
               "mean_alpha_dist_to_identity":
                   (mean_vec - ident_vec).abs().mean().item()},
              out_dir / f"swap_eval_{step_tag}.json")
    torch.save({k: v for k, v in per_token_saved.items()},
               out_dir / f"per_token_nll_{step_tag}.pt")
    print("[swap] DONE")


if __name__ == "__main__":
    main()
