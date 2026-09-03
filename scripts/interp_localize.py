"""Localize verbatim copying to named per-head Q/K routing, read their wiring,
and steer it bidirectionally.

Effective routing for stream s at layer l, head h = predictor alpha + local
correction, both laid out identically per layer: [q(H,n) k(H,n) v(H,n) r(n)].
A transform spec {layer: {stream: (head_set, gamma)}} rescales each listed
head's deviation from the layer mean: x_h <- m + gamma * (x_h - m), where m is
the mean over the layer's heads at that position. gamma=0 removes the head's
distinct wiring; gamma>1 amplifies it. The same transform is applied to the
predictor output and to the correction output, so it acts on the effective
routing.

Stage A: per-layer scan (all heads of one layer, q+k, gamma=0).
Stage B: per-head scan within the most sensitive layers.
Stage C: combined named-head set: gamma sweep, with copy accuracy on repeats,
         lag-128 log-prob on "repeat-then-novel" text, copy propensity on
         natural text, and held-out NLL as collateral.
Wiring readout: mean predictor Q/K source-layer profiles of the named heads.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from interp_common import load_elh, save_json, unflatten_alpha, flatten_alpha
from interp_editing import layer_chunks, stream_slices


def apply_spec_to_chunk(chunk: torch.Tensor, l: int, H: int, spec: dict) -> torch.Tensor:
    """chunk [1,T,chunk_dim] for layer l; spec[l] = {stream: (heads, gamma)}."""
    if l not in spec:
        return chunk
    sl, n = stream_slices(l, H)
    out = chunk.clone()
    for stream, (heads, gamma) in spec[l].items():
        s0, s1 = sl[stream]
        seg = out[:, :, s0:s1].view(*out.shape[:2], H, n)
        m = seg.mean(dim=2, keepdim=True)
        for h in heads:
            seg[:, :, h, :] = m[:, :, 0, :] + gamma * (seg[:, :, h, :] - m[:, :, 0, :])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump-dir", default="experiments/results/interp/alpha_dumps")
    ap.add_argument("--out", default="experiments/results/interp")
    ap.add_argument("--n-calib", type=int, default=25)
    ap.add_argument("--top-layers", type=int, default=3)
    ap.add_argument("--top-heads", type=int, default=6)
    args = ap.parse_args()

    elh = load_elh()
    cfg = elh.load_config(args.config)
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    chunks = layer_chunks(L, H)
    D = chunks[-1][1]
    dump = Path(args.dump_dir)
    fourway, predictor = elh.load_fourway(args.ckpt, cfg, device)

    corp = torch.load(dump / "corpora.pt", map_location="cpu", weights_only=False)
    ids, labels = corp["eval_ids"], corp["eval_labels"]
    rep_ids, rnd_ids = corp["repeat_ids"], corp["random_ids"]
    nc = args.n_calib
    n_test, T = ids.shape[0] - nc, ids.shape[1]
    PERIOD = 128

    state = {"spec": {}}
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            def hook(m, inp, out):
                return apply_spec_to_chunk(out, i + 1, H, state["spec"]) if state["spec"] else out
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))

    @torch.no_grad()
    def run(row):
        routing = predictor(row.to(device))
        if state["spec"]:
            flat = flatten_alpha(routing).float()
            parts = [apply_spec_to_chunk(flat[:, :, a:b], li + 1, H, state["spec"])
                     for li, (a, b) in enumerate(chunks)]
            flat = torch.cat(parts, dim=-1)
            routing = {s: [t.to(device) for t in v] for s, v in
                       unflatten_alpha(flat, L, H).items()}
        return fourway(row.to(device), routing)

    copy_mask = torch.zeros(T - 1, dtype=torch.bool); copy_mask[PERIOD - 1:] = True

    @torch.no_grad()
    def copy_acc(rows):
        accs = []
        for i in range(rows.shape[0]):
            row = rows[i:i + 1]
            pred = run(row)[:, :-1].argmax(-1).cpu().view(-1)
            accs.append((pred == row[:, 1:].view(-1))[copy_mask].float().mean().item())
        return sum(accs) / len(accs)

    @torch.no_grad()
    def nll_test(n=None):
        n = n or n_test
        vals = []
        for i in range(n):
            lg = run(ids[nc + i:nc + i + 1])
            vals.append(F.cross_entropy(lg.float().view(-1, lg.shape[-1]),
                                        labels[nc + i:nc + i + 1].to(device).view(-1)).item())
        return sum(vals) / len(vals)

    @torch.no_grad()
    def copy_propensity(rows):
        """Mean probability mass on tokens seen in the previous 128 positions."""
        vals = []
        for i in range(rows.shape[0]):
            row = rows[i:i + 1]
            p = torch.softmax(run(row)[:, :-1].float(), dim=-1)[0].cpu()
            mass = []
            for t in range(T - 1):
                prev = torch.unique(row[0, max(0, t - PERIOD + 1):t + 1])
                mass.append(p[t, prev].sum().item())
            vals.append(sum(mass) / len(mass))
        return sum(vals) / len(vals)

    g = torch.Generator().manual_seed(7)
    flat_corpus = ids.reshape(-1)
    def sample(n):
        return flat_corpus[torch.randint(0, flat_corpus.shape[0], (n,), generator=g)]
    half = torch.stack([torch.cat([sample(PERIOD).repeat(3), sample(T - 3 * PERIOD)])
                        for _ in range(16)])

    @torch.no_grad()
    def tail_lag_logprob(rows):
        vals = []
        for i in range(rows.shape[0]):
            row = rows[i:i + 1]
            lg = torch.log_softmax(run(row)[:, :-1].float(), dim=-1).cpu()
            lp = [lg[0, t, row[0, t + 1 - PERIOD]].item() for t in range(3 * PERIOD, T - 1)]
            vals.append(sum(lp) / len(lp))
        return sum(vals) / len(vals)

    results = {}
    state["spec"] = {}
    base_copy = copy_acc(rep_ids[16:32])
    base_nll = nll_test()
    results["baseline"] = {"copy_acc": base_copy, "nll": base_nll}
    print(f"[loc] baseline copy={base_copy:.4f} nll={base_nll:.4f}")

    # ---- Stage A: per-layer Q/K head-averaging ----
    all_heads = tuple(range(H))
    stageA = {}
    for l in range(1, L):
        state["spec"] = {l: {"q": (all_heads, 0.0), "k": (all_heads, 0.0)}}
        c = copy_acc(rep_ids[16:32]); n8 = nll_test(8)
        stageA[l] = {"copy_acc": c, "nll8": n8}
        print(f"[loc/A] layer {l:2d} qk head-avg: copy={c:.4f} nll8={n8:.4f}")
    results["stageA_layer_scan"] = stageA
    top_layers = sorted(stageA, key=lambda l: stageA[l]["copy_acc"])[:args.top_layers]
    print(f"[loc/A] most copy-sensitive layers: {top_layers}")

    # ---- Stage B: per-head scan within top layers ----
    stageB = {}
    for l in top_layers:
        for h in range(H):
            state["spec"] = {l: {"q": ((h,), 0.0), "k": ((h,), 0.0)}}
            c = copy_acc(rep_ids[16:32])
            stageB[f"L{l}/h{h}"] = c
            print(f"[loc/B] L{l} h{h:2d} qk->layer-mean: copy={c:.4f}")
    results["stageB_head_scan"] = stageB
    ranked = sorted(stageB, key=stageB.get)[:args.top_heads]
    named = [(int(k.split("/")[0][1:]), int(k.split("/h")[1])) for k in ranked]
    print(f"[loc/B] named heads (largest individual copy drop): {ranked}")

    def spec_for(named_heads, gamma):
        spec = {}
        for l, h in named_heads:
            spec.setdefault(l, {"q": ([], gamma), "k": ([], gamma)})
            spec[l]["q"][0].append(h); spec[l]["k"][0].append(h)
        return spec

    # ---- Stage C: combined named set, gamma sweep ----
    stageC = {}
    for gamma in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        state["spec"] = spec_for(named, gamma) if gamma != 1.0 else {}
        r = {"copy_acc": copy_acc(rep_ids[16:32]), "nll": nll_test(),
             "tail_lag_logprob": tail_lag_logprob(half),
             "copy_propensity_natural": copy_propensity(ids[nc:nc + 8])}
        stageC[f"gamma{gamma}"] = r
        print(f"[loc/C] gamma={gamma}: {r}")
    results["stageC_gamma_sweep"] = {"named_heads": ranked, **stageC}
    state["spec"] = {}

    # ---- Wiring readout: predictor Q/K source profiles of named heads ----
    @torch.no_grad()
    def alpha_profiles(rows):
        acc = {}
        for i in range(rows.shape[0]):
            routing = predictor(rows[i:i + 1].to(device))
            for l, h in named:
                for s in ("q", "k"):
                    v = routing[s][l - 1][0, PERIOD:, h, :].mean(0).cpu()
                    acc.setdefault(f"L{l}/h{h}/{s}", torch.zeros_like(v)).add_(v)
        return {k: (v / rows.shape[0]).tolist() for k, v in acc.items()}

    results["wiring_repeat"] = alpha_profiles(rep_ids[:16])
    results["wiring_random"] = alpha_profiles(rnd_ids[:16])
    for k, v in results["wiring_repeat"].items():
        print(f"[wiring] {k} source-layer weights (repeat text): "
              + " ".join(f"{x:+.2f}" for x in v))

    for hk in hooks:
        hk.remove()
    save_json(results, Path(args.out) / "localize_step9000.json")
    print("[loc] DONE")


if __name__ == "__main__":
    main()
