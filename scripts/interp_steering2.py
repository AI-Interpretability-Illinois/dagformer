"""Steering round 2 — three redesigned attempts after the v1 failure.

v1 failed because (a) it injected the V-stream "content-carrying" signal,
but knockout showed copying is driven by the Q/K "where-to-look" pathway;
(b) it asked the model to copy on pure random text where no repeat exists.

S1 DIAL-DOWN: on genuinely repeating text, SUBTRACT lambda x the
   repeat-fingerprint from the correction outputs. Suppressing an active
   mechanism should be easier than conjuring one. Also try subtracting only
   the Q/K part vs only the V part to locate the trigger pathway.
S2 FAIR-TARGET INDUCE: sequences that repeat a 128-token chunk 3x then
   switch to fresh random text. In the fresh tail, does +fingerprint make
   the model keep betting on "still repeating" (logprob of the token from
   128 positions back)?
S3 DOMAIN STEER: inject (code-context minus prose-context) correction
   fingerprint into prose windows; measure probability mass the model puts
   on code-typical tokens. Control: same injection on the external
   predictor's alpha is predicted to do NOTHING (its domain means were
   shown interchangeable).
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from interp_common import (alpha_layout, load_elh, save_json,
                           unflatten_alpha, flatten_alpha)
from interp_editing import layer_chunks


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

    layout = alpha_layout(L, H)
    mask_qk = torch.tensor([lab["stream"] in ("q", "k") for lab in layout])
    mask_v = torch.tensor([lab["stream"] == "v" for lab in layout])

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

    def set_delta(vec: torch.Tensor | None):
        edit["flat_delta"] = None if vec is None else vec.view(1, 1, -1).expand(1, T, -1)

    @torch.no_grad()
    def run(row, alpha_delta=None):
        routing = predictor(row.to(device))
        if alpha_delta is not None:
            flat = flatten_alpha(routing).float()
            flat = flat + alpha_delta.view(1, 1, -1).to(flat.device)
            routing = {s: [t.to(device) for t in v] for s, v in
                       unflatten_alpha(flat, L, H).items()}
        return fourway(row.to(device), routing)

    @torch.no_grad()
    def corr_mean(rows, from_pos=0):
        cap = {}
        hs = [mlp.register_forward_hook(lambda m, i_, o, k=k: cap.__setitem__(k, o.detach()))
              for k, mlp in enumerate(fourway.correction_mlps)]
        set_delta(None)
        acc = torch.zeros(D); n = 0
        for i in range(rows.shape[0]):
            cap.clear()
            run(rows[i:i + 1])
            flat = torch.cat([cap[k].float().cpu() for k in range(len(chunks))], dim=-1)
            acc += flat[0, from_pos:].mean(0); n += 1
        for h in hs:
            h.remove()
        return acc / n

    @torch.no_grad()
    def copy_acc(rows):
        accs = []
        for i in range(rows.shape[0]):
            row = rows[i:i + 1]
            lg = run(row)[:, :-1]
            pred = lg.argmax(-1).cpu().view(-1)
            m = torch.zeros(T - 1, dtype=torch.bool); m[PERIOD - 1:] = True
            accs.append((pred == row[:, 1:].view(-1))[m].float().mean().item())
        return sum(accs) / len(accs)

    @torch.no_grad()
    def nll_test():
        vals = []
        for i in range(n_test):
            lg = run(ids[nc + i:nc + i + 1])
            vals.append(F.cross_entropy(lg.float().view(-1, lg.shape[-1]),
                                        labels[nc + i:nc + i + 1].to(device).view(-1)).item())
        return sum(vals) / len(vals)

    results = {}
    fp = corr_mean(rep_ids[:16], from_pos=PERIOD) - corr_mean(rnd_ids[:16], from_pos=PERIOD)

    # ---- S1: dial verbatim copying DOWN on real repeats ----
    set_delta(None)
    s1 = {"lam0": {"copy_acc": copy_acc(rep_ids[16:32]), "nll": nll_test()}}
    print(f"[steer2/S1] lam=0: {s1['lam0']}")
    for lam in (0.25, 0.5, 1.0, 2.0):
        set_delta(-lam * fp)
        s1[f"lam{lam}"] = {"copy_acc": copy_acc(rep_ids[16:32]), "nll": nll_test()}
        print(f"[steer2/S1] -{lam}x full: {s1[f'lam{lam}']}")
    for tag, m in (("qk_only", mask_qk), ("v_only", mask_v)):
        set_delta(-0.5 * fp * m)
        s1[f"lam0.5_{tag}"] = {"copy_acc": copy_acc(rep_ids[16:32])}
        print(f"[steer2/S1] -0.5x {tag}: {s1[f'lam0.5_{tag}']}")
    results["S1_suppress"] = s1

    # ---- S2: fair-target induction (repeat 3x then fresh tail) ----
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
            lp = [lg[0, t, row[0, t + 1 - PERIOD]].item()
                  for t in range(3 * PERIOD, T - 1)]
            vals.append(sum(lp) / len(lp))
        return sum(vals) / len(vals)

    set_delta(None)
    s2 = {"lam0": tail_lag_logprob(half)}
    print(f"[steer2/S2] lam=0 tail lag-logprob: {s2['lam0']:.3f}")
    for lam in (0.25, 0.5, 1.0):
        set_delta(lam * fp)
        s2[f"lam{lam}_full"] = tail_lag_logprob(half)
        set_delta(lam * fp * mask_qk)
        s2[f"lam{lam}_qk"] = tail_lag_logprob(half)
        print(f"[steer2/S2] +{lam}: full={s2[f'lam{lam}_full']:.3f} qk={s2[f'lam{lam}_qk']:.3f}")
    results["S2_fair_induce"] = s2

    # ---- S3: domain steering (code fingerprint into prose) ----
    code_w, prose_w = corp["domains"]["code"], corp["domains"]["prose"]
    cc = Counter(code_w.reshape(-1).tolist())
    pc = Counter(prose_w.reshape(-1).tolist())
    tot_c, tot_p = sum(cc.values()), sum(pc.values())
    code_set = torch.tensor(sorted(
        t for t, n in cc.items()
        if n >= 20 and (n / tot_c) / ((pc.get(t, 0) + 1) / tot_p) >= 5.0))
    print(f"[steer2/S3] code-typical token set: {len(code_set)} tokens")

    fp_dom = corr_mean(code_w) - corr_mean(prose_w)
    a_code = torch.load(dump / "alpha_domain_code_step9000.pt", map_location="cpu").float()
    a_prose = torch.load(dump / "alpha_domain_prose_step9000.pt", map_location="cpu").float()
    fp_dom_alpha = a_code.reshape(-1, D).mean(0) - a_prose.reshape(-1, D).mean(0)

    @torch.no_grad()
    def code_mass_and_nll(rows, alpha_delta=None):
        mass, nll = [], []
        cs = code_set.to(device)
        for i in range(rows.shape[0]):
            row = rows[i:i + 1]
            lg = run(row, alpha_delta)[:, :-1].float()
            p = torch.softmax(lg, dim=-1)
            mass.append(p[0, :, cs].sum(-1).mean().item())
            nll.append(F.cross_entropy(lg.view(-1, lg.shape[-1]),
                                       row[:, 1:].to(device).view(-1)).item())
        return sum(mass) / len(mass), sum(nll) / len(nll)

    set_delta(None)
    m0, n0 = code_mass_and_nll(prose_w)
    s3 = {"lam0": {"code_mass": m0, "prose_nll": n0}}
    mc, nc_ = code_mass_and_nll(code_w)
    s3["code_ref"] = {"code_mass": mc, "nll": nc_}
    print(f"[steer2/S3] lam=0 prose: mass={m0:.4f} nll={n0:.3f} | code ref mass={mc:.4f}")
    for lam in (0.5, 1.0, 2.0):
        set_delta(lam * fp_dom)
        m, n = code_mass_and_nll(prose_w)
        s3[f"lam{lam}_corr"] = {"code_mass": m, "prose_nll": n}
        print(f"[steer2/S3] +{lam} corr-side: mass={m:.4f} nll={n:.3f}")
    set_delta(None)
    for lam in (1.0, 2.0):
        m, n = code_mass_and_nll(prose_w, alpha_delta=lam * fp_dom_alpha)
        s3[f"lam{lam}_alpha"] = {"code_mass": m, "prose_nll": n}
        print(f"[steer2/S3] +{lam} alpha-side (predicted no-op): mass={m:.4f} nll={n:.3f}")
    results["S3_domain"] = s3

    set_delta(None)
    for h in hooks:
        h.remove()
    save_json(results, Path(args.out) / "steering2_step9000.json")
    print("[steer2] DONE")


if __name__ == "__main__":
    main()
