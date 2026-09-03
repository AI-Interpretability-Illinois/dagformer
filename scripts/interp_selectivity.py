"""Is the copy-suppression intervention SELECTIVE, or generic damage?

Addresses the critique: "subtracting the repetition signature may simply
degrade the model rather than specifically reduce copying."

A. Random-direction control: perturb correction outputs by random vectors
   with the SAME L2 norm as lambda*fingerprint (5 seeds), plus the
   sign-flipped fingerprint. Compare copy-accuracy loss per unit of
   held-out NLL cost against the fingerprint direction.
B. Stratified natural-text loss: under the intervention, split held-out
   tokens into "target occurred in the preceding 128 tokens" (in-context
   copyable) vs "novel". Selective suppression should raise loss mainly on
   the copyable group.
C. Audit signal: per-token projection of live correction outputs onto the
   fingerprint direction as an online copy-mode detector; ROC-AUC against
   ground-truth copy events on natural text, vs a magnitude-only baseline.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            a, b = chunks[i]
            def hook(m, inp, out):
                captured[i] = out.detach()
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
        set_delta(None)
        acc = torch.zeros(D); n = 0
        for i in range(rows.shape[0]):
            captured.clear()
            run(rows[i:i + 1])
            flat = torch.cat([captured[k].float().cpu() for k in range(len(chunks))], dim=-1)
            acc += flat[0, from_pos:].mean(0); n += 1
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

    # in-context mask for natural test text: label token occurred in previous 128 inputs
    incontext = torch.zeros(n_test, T, dtype=torch.bool)
    for i in range(n_test):
        row = ids[nc + i]
        lab = labels[nc + i]
        for t in range(T):
            lo = max(0, t - PERIOD + 1)
            incontext[i, t] = bool((row[lo:t + 1] == lab[t]).any())
    print(f"[sel] in-context copyable fraction of natural tokens: {incontext.float().mean():.3f}")

    @torch.no_grad()
    def natural_nll_split():
        per = []
        for i in range(n_test):
            lg = run(ids[nc + i:nc + i + 1])
            nll = F.cross_entropy(lg.float().view(-1, lg.shape[-1]),
                                  labels[nc + i:nc + i + 1].to(device).view(-1),
                                  reduction="none").cpu()
            per.append(nll)
        per = torch.stack(per)
        return {"all": per.mean().item(),
                "in_context": per[incontext].mean().item(),
                "novel": per[~incontext].mean().item()}

    fp = corr_mean(rep_ids[:16], PERIOD) - corr_mean(rnd_ids[:16], PERIOD)
    results = {"fingerprint_norm": fp.norm().item()}

    # ---- A: random-direction control ----
    A = {}
    set_delta(None)
    A["none"] = {"copy_acc": copy_acc(rep_ids[16:32]), **natural_nll_split()}
    print(f"[sel/A] none: {A['none']}")
    g = torch.Generator().manual_seed(0)
    for lam in (0.5, 1.0):
        set_delta(-lam * fp)
        A[f"fp_minus_{lam}"] = {"copy_acc": copy_acc(rep_ids[16:32]), **natural_nll_split()}
        print(f"[sel/A] -{lam} fingerprint: {A[f'fp_minus_{lam}']}")
        set_delta(+lam * fp)
        A[f"fp_plus_{lam}"] = {"copy_acc": copy_acc(rep_ids[16:32]), **natural_nll_split()}
        print(f"[sel/A] +{lam} fingerprint: {A[f'fp_plus_{lam}']}")
        norm = (lam * fp).norm()
        for s in range(5):
            v = torch.randn(D, generator=g)
            v = v / v.norm() * norm
            set_delta(v)
            r = {"copy_acc": copy_acc(rep_ids[16:32]), **natural_nll_split()}
            A[f"random_{lam}_seed{s}"] = r
            print(f"[sel/A] random norm-matched lam={lam} seed{s}: {r}")
    set_delta(None)
    results["A_direction_control"] = A

    # ---- C: audit detector on natural text ----
    from sklearn.metrics import roc_auc_score
    scores_proj, scores_mag, y_copy, y_inctx = [], [], [], []
    fp_unit = fp / fp.norm()
    with torch.no_grad():
        for i in range(n_test):
            captured.clear()
            lg = run(ids[nc + i:nc + i + 1])
            flat = torch.cat([captured[k].float().cpu() for k in range(len(chunks))], dim=-1)[0]
            pred = lg[0].argmax(-1).cpu()
            correct = pred == labels[nc + i]
            scores_proj.append(flat @ fp_unit)
            scores_mag.append(flat.norm(dim=-1))
            y_inctx.append(incontext[i])
            y_copy.append(incontext[i] & correct)
    sp = torch.cat(scores_proj).numpy(); sm = torch.cat(scores_mag).numpy()
    yc = torch.cat(y_copy).numpy(); yi = torch.cat(y_inctx).numpy()
    C = {
        "auc_projection_vs_copy_event": float(roc_auc_score(yc, sp)),
        "auc_magnitude_vs_copy_event": float(roc_auc_score(yc, sm)),
        "auc_projection_vs_incontext": float(roc_auc_score(yi, sp)),
        "auc_magnitude_vs_incontext": float(roc_auc_score(yi, sm)),
        "copy_event_rate": float(yc.mean()),
    }
    print(f"[sel/C] {C}")
    results["C_audit"] = C

    for h in hooks:
        h.remove()
    save_json(results, Path(args.out) / "selectivity_step9000.json")
    print("[sel] DONE")


if __name__ == "__main__":
    main()
