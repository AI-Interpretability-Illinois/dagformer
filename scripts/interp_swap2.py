"""Round-2 routing interventions, all from saved alpha dumps + step-9000 ckpt.

Closes three gaps left by interp_swap_eval.py:
 1. pos_table: replace alpha with a per-POSITION mean (content-free positional
    schedule, calib-estimated). If ~= dynamic, the external predictor reduces
    to a position lookup table.
 2. *_no_corr variants of swap_context / shuffle / pos_table: correction MLPs
    see the hidden state and can compensate content mismatches, shadowing the
    external predictor's content channel. Removing them unmasks it.
 3. Cross-domain swaps: code/prose windows evaluated under own-domain-mean vs
    other-domain-mean vs dolma-mean alpha — tests coarse content conditioning
    that homogeneous in-distribution swaps cannot detect.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from interp_common import (STREAMS, broadcast_flat, load_elh, save_json,
                           unflatten_alpha)


@torch.no_grad()
def nll_under(fourway, ids_row, labels_row, flat_alpha, use_corr, L, H, device):
    fourway.use_local_correction = use_corr and hasattr(fourway, "correction_mlps")
    routing = {s: [t.to(device) for t in v] for s, v in
               unflatten_alpha(flat_alpha, L, H).items()}
    logits = fourway(ids_row.to(device), routing)
    return F.cross_entropy(logits.float().view(-1, logits.shape[-1]),
                           labels_row.to(device).view(-1)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump-dir", default="experiments/results/interp/alpha_dumps")
    ap.add_argument("--out", default="experiments/results/interp")
    ap.add_argument("--n-calib", type=int, default=25)
    args = ap.parse_args()

    elh = load_elh()
    cfg = elh.load_config(args.config)
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    dump = Path(args.dump_dir)

    fourway, _pred = elh.load_fourway(args.ckpt, cfg, device)
    orig_corr = fourway.use_local_correction

    corp = torch.load(dump / "corpora.pt", map_location="cpu", weights_only=False)
    ids, labels = corp["eval_ids"], corp["eval_labels"]
    a_eval = torch.load(dump / "alpha_eval_step9000.pt", map_location="cpu").float()
    nc = args.n_calib
    T = ids.shape[1]
    n_test = ids.shape[0] - nc

    mean_vec = a_eval[:nc].reshape(-1, a_eval.shape[-1]).mean(0)        # [D]
    pos_table = a_eval[:nc].mean(0)                                     # [T, D]
    print(f"[swap2] pos_table var vs mean_vec: "
          f"{(pos_table - mean_vec).abs().mean().item():.4f}")

    g = torch.Generator().manual_seed(0)
    perms = [torch.randperm(T, generator=g) for _ in range(n_test)]

    def alpha_for(mode, i):
        if mode == "dynamic":
            return a_eval[nc + i:nc + i + 1]
        if mode == "pos_table":
            return pos_table.unsqueeze(0)
        if mode == "static_mean":
            return broadcast_flat(mean_vec, 1, T)
        if mode == "swap_context":
            return a_eval[nc + (i + 1) % n_test: nc + (i + 1) % n_test + 1]
        if mode == "shuffle_time":
            return a_eval[nc + i:nc + i + 1][:, perms[i], :]
        raise ValueError(mode)

    results = {}
    for mode in ("dynamic", "pos_table", "static_mean", "swap_context", "shuffle_time"):
        for corr in (True, False):
            tag = mode + ("" if corr else "_no_corr")
            vals = [nll_under(fourway, ids[nc + i:nc + i + 1],
                              labels[nc + i:nc + i + 1], alpha_for(mode, i),
                              corr, L, H, device) for i in range(n_test)]
            results[tag] = sum(vals) / len(vals)
            print(f"[swap2] {tag:24s} NLL = {results[tag]:.4f}")

    # ---- Cross-domain (corr ON and OFF) ----
    dom_alpha = {}
    dom_mean = {}
    for name in corp["domains"]:
        p = dump / f"alpha_domain_{name}_step9000.pt"
        if p.exists():
            dom_alpha[name] = torch.load(p, map_location="cpu").float()
            dom_mean[name] = dom_alpha[name].reshape(-1, a_eval.shape[-1]).mean(0)
    cross = {}
    for name, a_dom in dom_alpha.items():
        wins = corp["domains"][name]
        others = [n for n in dom_mean if n != name]
        for corr in (True, False):
            suf = "" if corr else "_no_corr"
            for tag, af in (
                ("dynamic", lambda w, i: a_dom[i:i + 1]),
                ("own_mean", lambda w, i: broadcast_flat(dom_mean[name], 1, w.shape[1])),
                ("dolma_mean", lambda w, i: broadcast_flat(mean_vec, 1, w.shape[1])),
                ("pos_table", lambda w, i: pos_table[:w.shape[1]].unsqueeze(0)),
                *[(f"{o}_mean", lambda w, i, o=o: broadcast_flat(dom_mean[o], 1, w.shape[1]))
                  for o in others],
            ):
                vals = []
                for i in range(wins.shape[0]):
                    w = wins[i:i + 1]
                    pad = torch.full((1, 1), -100, dtype=torch.long)
                    lab = torch.cat([w[:, 1:], pad], dim=1)  # -100 = CE ignore_index
                    n = nll_under(fourway, w, lab, af(w, i), corr, L, H, device)
                    vals.append(n)
                cross[f"{name}/{tag}{suf}"] = sum(vals) / len(vals)
                print(f"[swap2] {name}/{tag}{suf:12s} NLL = {cross[f'{name}/{tag}{suf}']:.4f}")

    fourway.use_local_correction = orig_corr
    save_json({"eval_modes": results, "cross_domain": cross},
              Path(args.out) / "swap2_step9000.json")
    print("[swap2] DONE")


if __name__ == "__main__":
    main()
