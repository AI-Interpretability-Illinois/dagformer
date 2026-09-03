"""Per-token causal effect of every head, on a large held-out corpus — the
raw material for discovery-style head naming (no pre-specified functions).

For each head in [head_start, head_end): ablate its Q/K/V routing (replace by
layer mean, applied to predictor + corrections) and record the change in
per-token NLL relative to the unablated model, on n_seq held-out windows read
directly from the pretokenized mmap corpus at permuted positions the model
never consumed during training.

Shard 0 additionally writes: the corpus ids, the baseline per-token NLL, and
each head's per-token routing-correction magnitude (L2 norm of that head's
Q/K/V correction entries) — a free "activation-like" signal per head.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interp_common import REPO, load_elh, unflatten_alpha, flatten_alpha
from interp_editing import layer_chunks, stream_slices
from interp_localize import apply_spec_to_chunk


def build_corpus(index_path, seq_len, n_seq, seed, block_size, skip_positions, offset_seq=0):
    import sys
    sys.path.insert(0, str(REPO))
    from src.data.mmap_dataset import MmapPackedDataset, _epoch_positions
    ds = MmapPackedDataset(index_path, seq_len=seq_len)
    it = _epoch_positions(ds.n_samples, seed, block_size, 0, 1, skip_positions)
    for _ in range(offset_seq):
        next(it)
    idxs = [next(it) for _ in range(n_seq)]
    wins = np.stack([ds._read_window(i) for i in idxs]).astype(np.int64)
    return torch.from_numpy(wins[:, :seq_len]), torch.from_numpy(wins[:, 1:seq_len + 1]), idxs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--index-path", default="/work/hdd/bfqt/data/pretok/dolma_v1_7_12b")
    ap.add_argument("--n-seq", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--skip-positions", type=int, default=5_376_000)
    ap.add_argument("--head-start", type=int, default=0)
    ap.add_argument("--head-end", type=int, default=176)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--offset-seq", type=int, default=0,
                    help="skip this many unseen windows before taking n_seq (for fresh corpora)")
    ap.add_argument("--heads", default=None,
                    help="comma-separated explicit head list like L3/h1,L5/h2 (overrides head range)")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--save-corpus", action="store_true")
    ap.add_argument("--out", default="experiments/results/interp/token_effects")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    elh = load_elh()
    cfg = elh.load_config(args.config)
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    T = cfg["seq_len"]
    chunks = layer_chunks(L, H)
    fourway, predictor = elh.load_fourway(args.ckpt, cfg, device)

    ids, labels, idxs = build_corpus(args.index_path, T, args.n_seq, args.seed,
                                     args.block_size, args.skip_positions, args.offset_seq)
    shard0 = args.save_corpus or (args.head_start == 0 and args.heads is None)
    sfx = args.tag
    if shard0:
        torch.save({"ids": ids, "labels": labels, "sample_idxs": idxs}, out / f"corpus{sfx}.pt")
    print(f"[tokfx] corpus {tuple(ids.shape)} from mmap positions >= {args.skip_positions}")

    state = {"spec": {}}
    cap = {}
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            def hook(m, inp, o):
                cap[i] = o.detach()
                return apply_spec_to_chunk(o, i + 1, H, state["spec"]) if state["spec"] else o
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))

    @torch.no_grad()
    def run(rows):
        routing = predictor(rows.to(device))
        if state["spec"]:
            flat = flatten_alpha(routing).float()
            parts = [apply_spec_to_chunk(flat[:, :, a:b], li + 1, H, state["spec"])
                     for li, (a, b) in enumerate(chunks)]
            routing = {s: [t.to(device) for t in v] for s, v in
                       unflatten_alpha(torch.cat(parts, dim=-1), L, H).items()}
        return fourway(rows.to(device), routing)

    @torch.no_grad()
    def per_token_nll(collect_corr=False):
        outs, corr_mag = [], []
        for i in range(0, ids.shape[0], args.batch):
            rows, labs = ids[i:i + args.batch], labels[i:i + args.batch]
            cap.clear()
            lg = run(rows)
            nll = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]),
                                  labs.to(device).reshape(-1), reduction="none")
            outs.append(nll.view(rows.shape[0], T).half().cpu())
            if collect_corr:
                per_head = []
                for l in range(1, L):
                    o = cap[l - 1].float()                       # [B,T,chunk]
                    sl, n = stream_slices(l, H)
                    sq = o[:, :, sl["q"][0]:sl["q"][1]].view(o.shape[0], T, H, n)
                    sk = o[:, :, sl["k"][0]:sl["k"][1]].view(o.shape[0], T, H, n)
                    sv = o[:, :, sl["v"][0]:sl["v"][1]].view(o.shape[0], T, H, n)
                    mag = torch.sqrt(sq.pow(2).sum(-1) + sk.pow(2).sum(-1) + sv.pow(2).sum(-1))  # [B,T,H]
                    per_head.append(mag.permute(2, 0, 1))        # [H,B,T]
                corr_mag.append(torch.cat(per_head, 0).half().cpu())  # [(L-1)H, B, T]
        nll_all = torch.cat(outs, 0)
        return nll_all, (torch.cat(corr_mag, 1) if collect_corr else None)

    state["spec"] = {}
    base, corr_mag = per_token_nll(collect_corr=shard0)
    if shard0:
        torch.save(base, out / f"baseline_nll{sfx}.pt")
        torch.save(corr_mag, out / f"corr_magnitude{sfx}.pt")
        print(f"[tokfx] baseline mean NLL {base.float().mean():.4f}; corr_mag {tuple(corr_mag.shape)}")

    if args.heads:
        heads = [(int(x.split("/")[0][1:]), int(x.split("/h")[1])) for x in args.heads.split(",")]
        tag = f"custom_{abs(hash(args.heads)) % 10**8:08d}{sfx}"
    else:
        heads = [(l, h) for l in range(1, L) for h in range(H)][args.head_start:args.head_end]
        tag = f"heads{args.head_start:03d}-{args.head_end:03d}{sfx}"
    effects = torch.zeros(len(heads), ids.shape[0], T, dtype=torch.float16)
    for k, (l, h) in enumerate(heads):
        state["spec"] = {l: {"q": ((h,), 0.0), "k": ((h,), 0.0), "v": ((h,), 0.0)}}
        nll, _ = per_token_nll()
        effects[k] = (nll.float() - base.float()).half()
        print(f"[tokfx] L{l}/h{h}: mean dNLL {effects[k].float().mean():+.4f} "
              f"max {effects[k].float().max():.2f} ({k + 1}/{len(heads)})", flush=True)
        if (k + 1) % 10 == 0:
            torch.save({"heads": heads[:k + 1], "effects": effects[:k + 1]}, out / f"effects_{tag}.pt")
    state["spec"] = {}
    torch.save({"heads": heads, "effects": effects}, out / f"effects_{tag}.pt")
    for hk in hooks:
        hk.remove()
    print("[tokfx] DONE")


if __name__ == "__main__":
    main()
