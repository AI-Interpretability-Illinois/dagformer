"""Refined interface-editing experiments (v2, after v1 readout):

E4b component attribution: head-average ONLY alpha (corr live) vs ONLY corr
     (alpha live) — whose head-diversity carries LM quality vs copying?
E1b family knockout: zero the entire named "embedding->V" correction family
     (all layers x heads, src=0; 176 entries) vs a size-matched random
     control family (V-stream, src!=0). Expect copy collapse vs no collapse,
     with held-out NLL spared in both.
E5b sharp steering metric: inject lambda * repeat-fingerprint into
     non-repeating text; measure mean logprob of the lag-128 token (the
     induction hypothesis target) instead of the ceiling-limited
     copy-from-context rate.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from interp_common import STREAMS, load_elh, save_json, unflatten_alpha, flatten_alpha
from interp_editing import layer_chunks, stream_slices, head_avg_flat, coord_flat_index


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

    edit = {"zero_idx": None, "head_avg": None, "flat_delta": None}
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            a, b = chunks[i]
            def hook(m, inp, out):
                o = out
                if edit["head_avg"]:
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
                    mask = edit["zero_idx"][a:b]
                    if mask.any():
                        o = o.clone()
                        o[:, :, mask] = 0.0
                if edit["flat_delta"] is not None:
                    o = o + edit["flat_delta"][:, :, a:b].to(o.device, o.dtype)
                return o
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))

    def set_edit(**kw):
        edit.update({"zero_idx": None, "head_avg": None, "flat_delta": None})
        edit.update(kw)

    @torch.no_grad()
    def run(row, alpha_transform=None):
        routing = predictor(row.to(device))
        if alpha_transform is not None:
            flat = flatten_alpha(routing).float()
            routing = {s: [t.to(device) for t in v] for s, v in
                       unflatten_alpha(alpha_transform(flat), L, H).items()}
        return fourway(row.to(device), routing)

    @torch.no_grad()
    def nll_test(alpha_transform=None):
        vals = []
        for i in range(n_test):
            lg = run(ids[nc + i:nc + i + 1], alpha_transform)
            vals.append(F.cross_entropy(lg.float().view(-1, lg.shape[-1]),
                                        labels[nc + i:nc + i + 1].to(device).view(-1)).item())
        return sum(vals) / len(vals)

    copy_mask = torch.zeros(T - 1, dtype=torch.bool)
    copy_mask[PERIOD - 1:] = True

    @torch.no_grad()
    def copy_acc(rows, alpha_transform=None):
        accs = []
        for i in range(rows.shape[0]):
            row = rows[i:i + 1]
            lg = run(row, alpha_transform)[:, :-1]
            pred = lg.argmax(-1).cpu().view(-1)
            accs.append((pred == row[:, 1:].view(-1))[copy_mask].float().mean().item())
        return sum(accs) / len(accs)

    @torch.no_grad()
    def lag_logprob(rows):
        """Mean logprob assigned to ids[t+1-PERIOD] when predicting pos t+1."""
        vals = []
        for i in range(rows.shape[0]):
            row = rows[i:i + 1]
            lg = torch.log_softmax(run(row)[:, :-1].float(), dim=-1).cpu()
            lp = []
            for t in range(PERIOD - 1, T - 1):
                lp.append(lg[0, t, row[0, t + 1 - PERIOD]].item())
            vals.append(sum(lp) / len(lp))
        return sum(vals) / len(vals)

    results = {}
    set_edit()
    results["baseline"] = {"nll": nll_test(), "repeat_copy_acc": copy_acc(rep_ids[:16])}
    print(f"[edit2] baseline: {results['baseline']}")

    # --- E4b: whose head-diversity? ---
    tr_avg = lambda flat: head_avg_flat(flat, L, H, ("q", "k", "v"))
    set_edit()
    results["head_avg_alpha_only"] = {"nll": nll_test(tr_avg),
                                      "repeat_copy_acc": copy_acc(rep_ids[:16], tr_avg)}
    print(f"[edit2] head_avg_alpha_only: {results['head_avg_alpha_only']}")
    set_edit(head_avg=("q", "k", "v"))
    results["head_avg_corr_only"] = {"nll": nll_test(),
                                     "repeat_copy_acc": copy_acc(rep_ids[:16])}
    print(f"[edit2] head_avg_corr_only: {results['head_avg_corr_only']}")

    # --- E1b: family knockout vs matched control ---
    fam = torch.zeros(D, dtype=torch.bool)
    for l in range(1, L):
        for h in range(H):
            fam[coord_flat_index(L, H, "v", l, h, 0)] = True
    rng = random.Random(0)
    ctrl = torch.zeros(D, dtype=torch.bool)
    for l in range(1, L):
        for h in range(H):
            src = rng.randrange(1, l + 1)
            ctrl[coord_flat_index(L, H, "v", l, h, src)] = True
    for tag, mask in (("knockout_emb2v_family", fam), ("knockout_ctrl_family", ctrl)):
        set_edit(zero_idx=mask)
        results[tag] = {"n_entries": int(mask.sum()), "nll": nll_test(),
                        "repeat_copy_acc": copy_acc(rep_ids[:16])}
        print(f"[edit2] {tag}: {results[tag]}")

    # --- E5b: steering with the sharp lag-128 metric ---
    @torch.no_grad()
    def corr_mean(rows):
        cap = {}
        hs = [mlp.register_forward_hook(lambda m, i_, o, k=k: cap.__setitem__(k, o.detach()))
              for k, mlp in enumerate(fourway.correction_mlps)]
        set_edit()
        acc = torch.zeros(D); n = 0
        for i in range(rows.shape[0]):
            cap.clear()
            run(rows[i:i + 1])
            flat = torch.cat([cap[k].float().cpu() for k in range(len(chunks))], dim=-1)
            acc += flat[0, PERIOD:].mean(0); n += 1
        for h in hs:
            h.remove()
        return acc / n

    fingerprint = corr_mean(rep_ids[:16]) - corr_mean(rnd_ids[:16])
    set_edit()
    results["steer"] = {"x0": {"lag_logprob": lag_logprob(rnd_ids[16:32])}}
    results["steer"]["repeat_ref"] = {"lag_logprob": lag_logprob(rep_ids[:8])}
    print(f"[edit2] steer x0: {results['steer']['x0']}  repeat_ref: {results['steer']['repeat_ref']}")
    for lam in (0.25, 0.5, 1.0):
        set_edit(flat_delta=(lam * fingerprint).view(1, 1, -1).expand(1, T, -1))
        results["steer"][f"x{lam}"] = {"lag_logprob": lag_logprob(rnd_ids[16:32]),
                                       "nll_drift": nll_test()}
        print(f"[edit2] steer x{lam}: {results['steer'][f'x{lam}']}")
    set_edit()

    for h in hooks:
        h.remove()
    save_json(results, Path(args.out) / "editing2_step9000.json")
    print("[edit2] DONE")


if __name__ == "__main__":
    main()
