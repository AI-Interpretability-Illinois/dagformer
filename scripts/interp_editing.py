"""Architecture-advantage demos: the routing interface is named, editable,
steerable, and head-granular.

E4 head-granularity necessity: average effective routing (alpha + corr)
    over heads per (layer, stream, src) — simulating layer-level routing a
    la MUDDFormer. If performance/copying degrades (esp. V), head-level
    granularity is causally necessary; layer-level routers cannot express
    the opposing-sign per-head mechanism.
E1 surgical knockout: zero ONLY the top-K induction-identified correction
    entries (named coordinates, e.g. v/L11/h4/src0) everywhere. Expect:
    copy accuracy drops, held-out NLL ~unchanged (specificity).
E5 steering: add lambda * (repeat - random) correction fingerprint to
    non-repeating text. Expect: copy-from-context rate rises.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from interp_common import STREAMS, load_elh, save_json, unflatten_alpha, flatten_alpha


def layer_chunks(L, H):
    offs, off = [], 0
    for l in range(1, L):
        sz = (3 * H + 1) * (l + 1)
        offs.append((off, off + sz))
        off += sz
    return offs


def stream_slices(l, H):
    """Within a layer chunk: (stream -> (start, end, n_src)) in flat order."""
    n = l + 1
    q = (0, H * n)
    k = (H * n, 2 * H * n)
    v = (2 * H * n, 3 * H * n)
    r = (3 * H * n, 3 * H * n + n)
    return {"q": q, "k": k, "v": v, "r": r}, n


def head_avg_flat(flat: torch.Tensor, L: int, H: int, streams: tuple[str, ...]) -> torch.Tensor:
    """Average selected per-head streams over heads (layer-level routing sim)."""
    out = flat.clone()
    chunks = layer_chunks(L, H)
    for li, (a, b) in enumerate(chunks):
        l = li + 1
        sl, n = stream_slices(l, H)
        for s in streams:
            if s == "r":
                continue
            s0, s1 = sl[s]
            seg = out[:, :, a + s0:a + s1].view(*out.shape[:2], H, n)
            seg.copy_(seg.mean(dim=2, keepdim=True).expand_as(seg))
    return out


def coord_flat_index(L, H, stream, layer, head, src):
    chunks = layer_chunks(L, H)
    a, _ = chunks[layer - 1]
    sl, n = stream_slices(layer, H)
    s0, _ = sl[stream]
    return a + s0 + (head * n if stream != "r" else 0) + src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump-dir", default="experiments/results/interp/alpha_dumps")
    ap.add_argument("--corr-json", default="experiments/results/interp/corr_anatomy_step9000.json")
    ap.add_argument("--out", default="experiments/results/interp")
    ap.add_argument("--n-calib", type=int, default=25)
    ap.add_argument("--topk", type=int, default=6)
    args = ap.parse_args()

    elh = load_elh()
    cfg = elh.load_config(args.config)
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    dump = Path(args.dump_dir)
    fourway, predictor = elh.load_fourway(args.ckpt, cfg, device)
    chunks = layer_chunks(L, H)
    D = chunks[-1][1]

    corp = torch.load(dump / "corpora.pt", map_location="cpu", weights_only=False)
    ids, labels = corp["eval_ids"], corp["eval_labels"]
    rep_ids, rnd_ids = corp["repeat_ids"], corp["random_ids"]
    nc = args.n_calib
    n_test, T = ids.shape[0] - nc, ids.shape[1]

    # --- correction hook with three edit modes ---
    edit = {"mode": "none", "flat_delta": None, "zero_idx": None, "head_avg": None}
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            a, b = chunks[i]
            def hook(m, inp, out):
                o = out
                if edit["head_avg"]:
                    # per-layer head-avg (o is [B,T,chunk]), inline:
                    l = i + 1
                    sl, n = stream_slices(l, H)
                    o = o.clone()
                    for s in edit["head_avg"]:
                        if s == "r":
                            continue
                        s0, s1 = sl[s]
                        seg = o[:, :, s0:s1].view(*o.shape[:2], H, n)
                        seg.copy_(seg.mean(dim=2, keepdim=True).expand_as(seg))
                if edit["zero_idx"] is not None:
                    o = o.clone()
                    for j in edit["zero_idx"]:
                        if a <= j < b:
                            o[:, :, j - a] = 0.0
                if edit["flat_delta"] is not None:
                    o = o + edit["flat_delta"][:, :, a:b].to(o.device, o.dtype)
                return o
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))

    @torch.no_grad()
    def run(row: torch.Tensor, alpha_transform=None) -> torch.Tensor:
        routing = predictor(row.to(device))
        if alpha_transform is not None:
            flat = flatten_alpha(routing).float()
            routing = {s: [t.to(device) for t in v] for s, v in
                       unflatten_alpha(alpha_transform(flat), L, H).items()}
        return fourway(row.to(device), routing)

    @torch.no_grad()
    def nll_test(alpha_transform=None) -> float:
        vals = []
        for i in range(n_test):
            lg = run(ids[nc + i:nc + i + 1], alpha_transform)
            vals.append(F.cross_entropy(lg.float().view(-1, lg.shape[-1]),
                                        labels[nc + i:nc + i + 1].to(device).view(-1)).item())
        return sum(vals) / len(vals)

    copy_mask = torch.zeros(T - 1, dtype=torch.bool)
    copy_mask[127:] = True

    @torch.no_grad()
    def copy_metrics(rows: torch.Tensor, alpha_transform=None) -> dict:
        accs, rates = [], []
        for i in range(rows.shape[0]):
            row = rows[i:i + 1]
            lg = run(row, alpha_transform)[:, :-1]
            pred = lg.argmax(-1).cpu().view(-1)
            lab = row[:, 1:].view(-1)
            accs.append((pred == lab)[copy_mask].float().mean().item())
            seen = [set(row[0, max(0, t - 128):t + 1].tolist()) for t in range(T - 1)]
            rates.append(sum(int(p.item()) in s for p, s in zip(pred, seen)) / (T - 1))
        return {"copy_acc": sum(accs) / len(accs),
                "copy_from_ctx_rate": sum(rates) / len(rates)}

    def set_edit(**kw):
        edit.update({"mode": "none", "flat_delta": None, "zero_idx": None,
                     "head_avg": None})
        edit.update(kw)

    results = {}

    # --- baseline anchors ---
    set_edit()
    results["baseline"] = {"nll": nll_test(),
                           **{f"repeat_{k}": v for k, v in copy_metrics(rep_ids[:16]).items()},
                           **{f"random_{k}": v for k, v in copy_metrics(rnd_ids[:16]).items()}}
    print(f"[edit] baseline: {results['baseline']}")

    # --- E4: head-granularity necessity (alpha AND corrections averaged) ---
    for streams in (("q", "k", "v"), ("v",), ("q", "k")):
        tag = "head_avg_" + "".join(streams)
        set_edit(head_avg=streams)
        tr = lambda flat, s=streams: head_avg_flat(flat, L, H, s)
        results[tag] = {"nll": nll_test(tr),
                        **{f"repeat_{k}": v for k, v in copy_metrics(rep_ids[:16], tr).items()}}
        print(f"[edit] {tag}: {results[tag]}")

    # --- E1: surgical knockout of named induction entries (corr channel) ---
    top = json.load(open(args.corr_json))["induction"]["top_entries"][:args.topk]
    idxs = [coord_flat_index(L, H, e["stream"], e["layer"], e["head"], e["src"])
            for e in top]
    set_edit(zero_idx=idxs)
    results["knockout"] = {"entries": top, "nll": nll_test(),
                           **{f"repeat_{k}": v for k, v in copy_metrics(rep_ids[:16]).items()}}
    print(f"[edit] knockout({len(idxs)}): nll={results['knockout']['nll']:.4f} "
          f"copy_acc={results['knockout']['repeat_copy_acc']:.4f}")

    # --- E5: steering — add repeat-fingerprint to non-repeating text ---
    @torch.no_grad()
    def dump_corr_mean(rows):
        cap = {}
        hs = [mlp.register_forward_hook(lambda m, i_, o, k=k: cap.__setitem__(k, o.detach()))
              for k, mlp in enumerate(fourway.correction_mlps)]
        set_edit()
        acc = torch.zeros(D)
        n = 0
        for i in range(rows.shape[0]):
            cap.clear()
            run(rows[i:i + 1])
            flat = torch.cat([cap[k].float().cpu() for k in range(len(chunks))], dim=-1)
            acc += flat[0, 128:].mean(0)
            n += 1
        for h in hs:
            h.remove()
        return acc / n

    fingerprint = dump_corr_mean(rep_ids[:16]) - dump_corr_mean(rnd_ids[:16])  # [D]
    for lam in (0.5, 1.0, 2.0):
        set_edit(flat_delta=(lam * fingerprint).view(1, 1, -1).expand(1, T, -1))
        m = copy_metrics(rnd_ids[16:32])
        results[f"steer_x{lam}"] = m
        print(f"[edit] steer x{lam}: {m}")
    set_edit()
    results["steer_x0"] = copy_metrics(rnd_ids[16:32])
    print(f"[edit] steer x0 (ref): {results['steer_x0']}")

    for h in hooks:
        h.remove()
    save_json(results, Path(args.out) / "editing_step9000.json")
    print("[edit] DONE")


if __name__ == "__main__":
    main()
