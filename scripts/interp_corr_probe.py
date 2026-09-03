"""Anatomy of the correction MLPs: dump their per-token output (delta-corr,
the hidden-state-conditioned routing adjustment), then ask the same questions
we asked of the external predictor's alpha:
  - variance decomposition: positional vs content share
  - linear probes vs token-id baseline (+ residualization)
  - repeat-vs-random induction delta

The external predictor was shown to be position+lexical only; if content
adaptation exists anywhere in the routing system, it must be here.

Correction outputs are captured with forward hooks on correction_mlps[i];
their raw layout per layer is [q(H*n_src), k, v, r] — identical to the
canonical flat-alpha layout, so per-layer concatenation gives [B,T,3773].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from interp_common import load_elh, save_json


@torch.no_grad()
def dump_corr(fourway, predictor, ids: torch.Tensor, device) -> torch.Tensor:
    """Run full forwards, capture correction outputs -> [N,T,D] fp16 cpu."""
    n_corr = len(fourway.correction_mlps)
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        hooks.append(mlp.register_forward_hook(
            lambda m, inp, out, i=i: captured.__setitem__(i, out.detach())))
    rows = []
    try:
        for j in range(ids.shape[0]):
            captured.clear()
            row = ids[j:j + 1].to(device)
            routing = predictor(row)
            fourway(row, routing)
            assert len(captured) == n_corr, f"captured {len(captured)}/{n_corr}"
            flat = torch.cat([captured[i].float() for i in range(n_corr)], dim=-1)
            rows.append(flat.half().cpu())
    finally:
        for h in hooks:
            h.remove()
    return torch.cat(rows, dim=0)


def var_decomp(a: np.ndarray, streams: np.ndarray) -> dict:
    pos_var = a.mean(axis=0).var(axis=0)
    cont_var = a.var(axis=0).mean(axis=0)
    out = {}
    for s in ("q", "k", "v", "r", "ALL"):
        m = streams == s if s != "ALL" else np.ones_like(streams, dtype=bool)
        p, c = float(pos_var[m].mean()), float(cont_var[m].mean())
        out[s] = {"pos_var": p, "content_var": c,
                  "pos_share": p / (p + c) if p + c > 0 else float("nan")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump-dir", default="experiments/results/interp/alpha_dumps")
    ap.add_argument("--swap-nll", default="experiments/results/interp/per_token_nll_step9000.pt")
    ap.add_argument("--out", default="experiments/results/interp")
    args = ap.parse_args()

    elh = load_elh()
    cfg = elh.load_config(args.config)
    device = torch.device("cuda")
    dump = Path(args.dump_dir)
    fourway, predictor = elh.load_fourway(args.ckpt, cfg, device)
    assert fourway.use_local_correction, "expected corrected model"

    corp = torch.load(dump / "corpora.pt", map_location="cpu", weights_only=False)
    layout = json.load(open(dump / "layout.json"))
    streams = np.array([lab["stream"] for lab in layout])

    print("[corr] dumping eval corpus...")
    c_eval = dump_corr(fourway, predictor, corp["eval_ids"], device)
    torch.save(c_eval, dump / "corr_eval_step9000.pt")
    c_rep = dump_corr(fourway, predictor, corp["repeat_ids"], device)
    c_rnd = dump_corr(fourway, predictor, corp["random_ids"], device)
    dom_corr = {n: dump_corr(fourway, predictor, w, device)
                for n, w in corp["domains"].items()}

    a = c_eval.float().numpy()
    vd = var_decomp(a, streams)
    print("[corr] variance decomposition:", json.dumps(vd, indent=1))

    delta = (c_rep[:, 128:].float().mean(dim=(0, 1))
             - c_rnd[:, 128:].float().mean(dim=(0, 1))).numpy()
    top = np.argsort(-np.abs(delta))[:15]
    induction = {"mean_abs_delta": float(np.abs(delta).mean()),
                 "top_entries": [{**layout[i], "delta": float(delta[i])} for i in top]}
    print(f"[corr] induction mean|delta|={induction['mean_abs_delta']:.4f} "
          f"top={induction['top_entries'][0]}")

    # ---- probes (reuse machinery from interp_probe) ----
    import interp_probe as ip
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])
    ids = corp["eval_ids"]
    N, T = ids.shape
    targets = ip.build_targets(ids, tok)
    if Path(args.swap_nll).exists():
        ptn = torch.load(args.swap_nll, map_location="cpu", weights_only=False)
        dyn = ptn["dynamic"].numpy()
        qs = np.quantile(dyn.reshape(-1), [0.25, 0.5, 0.75])
        full = np.full((N, T), -1, dtype=np.int64)
        full[N - dyn.shape[0]:] = np.digitize(dyn, qs)
        targets["nllquartile"] = full

    tr_seq = np.arange(N) < int(0.8 * N)
    results = {}
    for tname, y in targets.items():
        valid = y >= 0
        Xtr = a[tr_seq][valid[tr_seq]]
        Xte = a[~tr_seq][valid[~tr_seq]]
        ytr = y[tr_seq][valid[tr_seq]]
        yte = y[~tr_seq][valid[~tr_seq]]
        itr = ids.numpy()[tr_seq][valid[tr_seq]]
        ite = ids.numpy()[~tr_seq][valid[~tr_seq]]
        if len(np.unique(ytr)) < 2 or len(yte) == 0:
            continue
        acc, _, _ = ip.fit_probe(Xtr, ytr, Xte, yte)
        residualize = ip.per_id_mean_alpha(itr, Xtr)
        acc_r, _, _ = ip.fit_probe(residualize(itr, Xtr), ytr,
                                   residualize(ite, Xte), yte)
        results[tname] = {"corr": acc, "corr_resid": acc_r,
                          "token_id": ip.token_id_baseline(itr, ytr, ite, yte),
                          "majority": ip.majority_baseline(ytr, yte)}
        print(f"[corrprobe] {tname:12s} corr={acc:.3f} resid={acc_r:.3f} "
              f"token_id={results[tname]['token_id']:.3f} "
              f"majority={results[tname]['majority']:.3f}")

    # domain probe on corr features
    dom_names = sorted(dom_corr)
    if len(dom_names) >= 2:
        Xd, yd, idd, wind = [], [], [], []
        for di, name in enumerate(dom_names):
            arr = dom_corr[name].float().numpy()
            w, t_len, _ = arr.shape
            Xd.append(arr.reshape(-1, arr.shape[-1]))
            yd.append(np.full(w * t_len, di))
            idd.append(corp["domains"][name].numpy().reshape(-1))
            wind.append(np.repeat(np.arange(w), t_len) + 100 * di)
        Xd = np.concatenate(Xd); yd = np.concatenate(yd)
        idd = np.concatenate(idd); wind = np.concatenate(wind)
        te = np.isin(wind, [100 * di + k for di in range(len(dom_names)) for k in (0, 1)])
        tr = ~te
        acc, _, _ = ip.fit_probe(Xd[tr], yd[tr], Xd[te], yd[te])
        residualize = ip.per_id_mean_alpha(idd[tr], Xd[tr])
        acc_r, _, _ = ip.fit_probe(residualize(idd[tr], Xd[tr]), yd[tr],
                                   residualize(idd[te], Xd[te]), yd[te])
        results["domain"] = {"corr": acc, "corr_resid": acc_r,
                             "token_id": ip.token_id_baseline(idd[tr], yd[tr], idd[te], yd[te]),
                             "classes": dom_names}
        print(f"[corrprobe] domain       corr={acc:.3f} resid={acc_r:.3f} "
              f"token_id={results['domain']['token_id']:.3f}")

    save_json({"variance_decomposition": vd, "induction": induction,
               "probes": results}, Path(args.out) / "corr_anatomy_step9000.json")
    print("[corr] DONE")


if __name__ == "__main__":
    main()
