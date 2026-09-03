"""Specificity control for the copy-suppression intervention.

Critique: subtracting the repeat-fingerprint from correction outputs lowers
verbatim-copy accuracy, but also raises held-out loss — is the effect a
targeted reduction of copying, or generic damage?

Test: compare, at matched collateral cost (held-out NLL increase), the copy
accuracy drop caused by
  (a) the fingerprint direction (scaled by lambda),
  (b) random Gaussian directions of equal L2 norm (several norms, 3 seeds),
  (c) the fingerprint with its coordinates randomly permuted (same
      magnitude profile, wrong structure).
Also report next-token top-1 accuracy on normal held-out text as a
non-copy behavior. A targeted intervention shows: large copy drop,
small normal-accuracy drop, and a steeper copy-drop-per-NLL slope than
random directions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from interp_common import load_elh, save_json
from interp_editing import layer_chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump-dir", default="experiments/results/interp/alpha_dumps")
    ap.add_argument("--out", default="experiments/results/interp")
    ap.add_argument("--n-calib", type=int, default=25)
    ap.add_argument("--random-mults", type=float, nargs="+",
                    default=[1, 2, 4, 8, 16])
    ap.add_argument("--tag", default="")
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

    edit = {"flat_delta": None}
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            a, b = chunks[i]
            def hook(m, inp, out):
                d = edit["flat_delta"]
                return out if d is None else out + d[:, :, a:b].to(out.device, out.dtype)
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))

    def set_delta(vec):
        edit["flat_delta"] = None if vec is None else vec.view(1, 1, -1).expand(1, T, -1)

    @torch.no_grad()
    def run(row):
        return fourway(row.to(device), predictor(row.to(device)))

    @torch.no_grad()
    def corr_mean(rows, from_pos):
        cap = {}
        hs = [mlp.register_forward_hook(lambda m, i_, o, k=k: cap.__setitem__(k, o.detach()))
              for k, mlp in enumerate(fourway.correction_mlps)]
        set_delta(None)
        acc = torch.zeros(D); n = 0
        for i in range(rows.shape[0]):
            cap.clear(); run(rows[i:i + 1])
            flat = torch.cat([cap[k].float().cpu() for k in range(len(chunks))], dim=-1)
            acc += flat[0, from_pos:].mean(0); n += 1
        for h in hs:
            h.remove()
        return acc / n

    @torch.no_grad()
    def copy_acc(rows):
        accs = []
        m = torch.zeros(T - 1, dtype=torch.bool); m[PERIOD - 1:] = True
        for i in range(rows.shape[0]):
            row = rows[i:i + 1]
            pred = run(row)[:, :-1].argmax(-1).cpu().view(-1)
            accs.append((pred == row[:, 1:].view(-1))[m].float().mean().item())
        return sum(accs) / len(accs)

    @torch.no_grad()
    def normal_metrics():
        nll, top1 = [], []
        for i in range(n_test):
            lg = run(ids[nc + i:nc + i + 1])
            lab = labels[nc + i:nc + i + 1].to(device)
            nll.append(F.cross_entropy(lg.float().view(-1, lg.shape[-1]), lab.view(-1)).item())
            top1.append((lg.argmax(-1) == lab).float().mean().item())
        return sum(nll) / len(nll), sum(top1) / len(top1)

    def evaluate(tag, vec):
        set_delta(vec)
        c = copy_acc(rep_ids[16:32])
        n, t = normal_metrics()
        set_delta(None)
        r = {"copy_acc": c, "normal_nll": n, "normal_top1": t}
        print(f"[spec] {tag:28s} copy={c:.4f} nll={n:.4f} top1={t:.4f}")
        return r

    fp = corr_mean(rep_ids[:16], PERIOD) - corr_mean(rnd_ids[:16], PERIOD)
    results = {"baseline": evaluate("baseline", None)}
    for lam in (0.25, 0.5, 1.0):
        results[f"fingerprint_x{lam}"] = evaluate(f"fingerprint x{lam}", -lam * fp)

    ref_norm = (0.5 * fp).norm().item()
    g = torch.Generator().manual_seed(0)
    for mult in args.random_mults:
        for seed in range(3):
            v = torch.randn(D, generator=g)
            v = v / v.norm() * (ref_norm * mult)
            results[f"random_norm{mult}x_seed{seed}"] = evaluate(
                f"random {mult}x|0.5fp| seed{seed}", v)
    for lam in (0.5, 1.0):
        for seed in range(3):
            perm = torch.randperm(D, generator=g)
            results[f"permuted_x{lam}_seed{seed}"] = evaluate(
                f"permuted fp x{lam} seed{seed}", -lam * fp[perm])

    for h in hooks:
        h.remove()
    results["_ref_norm_0.5fp"] = ref_norm
    save_json(results, Path(args.out) / f"specificity{args.tag}_step9000.json")
    print("[spec] DONE")


if __name__ == "__main__":
    main()
