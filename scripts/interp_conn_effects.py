"""Connection-level per-token effects: for each named head, ablate ONE
routing connection at a time — (stream in q/k/v, head h, source s) — by
replacing that entry of the head's effective routing with the layer mean of
the same (stream, source) entry across heads. Records per-token dNLL on the
1000-window corpus. Output: effects_conn_<shard>.pt with a list of
(layer, head, stream, src) and effects [n_conn, N, T] fp16.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from interp_common import load_elh, unflatten_alpha, flatten_alpha
from interp_editing import layer_chunks, stream_slices


def apply_conn(chunk, l, H, conn):
    """conn = (stream, head, src) for layer l; chunk [B,T,chunk_dim]."""
    stream, h, src = conn
    sl, n = stream_slices(l, H)
    s0, s1 = sl[stream]
    out = chunk.clone()
    seg = out[:, :, s0:s1].view(*out.shape[:2], H, n)
    seg[:, :, h, src] = seg[:, :, :, src].mean(dim=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True, help="comma-separated L{l}/h{h} list")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default="experiments/results/interp/token_effects")
    args = ap.parse_args()
    out = Path(args.out)
    elh = load_elh()
    cfg = elh.load_config("configs/fourway_300m_dagformer_mmap.yaml")
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    chunks = layer_chunks(L, H)
    fourway, predictor = elh.load_fourway("checkpoints/fourway_300m_dagformer_mmap/checkpoint_step9000.pt", cfg, device)
    corpus = torch.load(out / "corpus.pt", map_location="cpu", weights_only=False)
    ids, labels = corpus["ids"], corpus["labels"]
    N, T = ids.shape
    base = torch.load(out / "baseline_nll.pt", map_location="cpu").float()

    state = {"conn": None}  # (layer, stream, head, src)
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            def hook(m, inp, o):
                c = state["conn"]
                return apply_conn(o, i + 1, H, c[1:]) if (c and c[0] == i + 1) else o
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))

    @torch.no_grad()
    def run(rows):
        routing = predictor(rows.to(device))
        c = state["conn"]
        if c:
            flat = flatten_alpha(routing).float()
            a, b = chunks[c[0] - 1]
            flat = torch.cat([flat[:, :, :a], apply_conn(flat[:, :, a:b], c[0], H, c[1:]), flat[:, :, b:]], dim=-1)
            routing = {s: [t.to(device) for t in v] for s, v in unflatten_alpha(flat, L, H).items()}
        return fourway(rows.to(device), routing)

    @torch.no_grad()
    def per_token_nll():
        outs = []
        for i in range(0, N, args.batch):
            rows, labs = ids[i:i + args.batch], labels[i:i + args.batch]
            lg = run(rows)
            nll = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), labs.to(device).reshape(-1), reduction="none")
            outs.append(nll.view(rows.shape[0], T).cpu())
        return torch.cat(outs, 0)

    heads = [(int(x.split("/")[0][1:]), int(x.split("/h")[1])) for x in args.heads.split(",")]
    conns = [(l, s, h, src) for (l, h) in heads for s in ("q", "k", "v") for src in range(l + 1)]
    effects = torch.zeros(len(conns), N, T, dtype=torch.float16)
    print(f"[conn] {len(conns)} connections over {len(heads)} heads", flush=True)
    for k, c in enumerate(conns):
        state["conn"] = c
        effects[k] = (per_token_nll() - base).half()
        if (k + 1) % 20 == 0:
            print(f"[conn] {k + 1}/{len(conns)} mean dNLL {effects[k].float().mean():+.4f}", flush=True)
            torch.save({"conns": conns[:k + 1], "effects": effects[:k + 1]}, out / f"effects_conn_{args.shard}.pt")
    state["conn"] = None
    torch.save({"conns": conns, "effects": effects}, out / f"effects_conn_{args.shard}.pt")
    print("[conn] DONE")


if __name__ == "__main__":
    main()
