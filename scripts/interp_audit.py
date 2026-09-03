"""Audit monitor + refined selectivity stratification.

AUDIT: train a linear monitor on the correction-module outputs to flag
copy events on natural held-out text (event = target token occurred in the
preceding 128 tokens AND the model predicts it correctly). Report test-set
ROC-AUC for: correction outputs (3773-d), correction outputs residualized
by current-token identity, hidden states at layers 3/6/9 (3072-d, the
correction modules' own inputs), and a token-identity baseline
(P(event | token id) from the training split).

SELECTIVITY: per-token loss under the copy-suppression intervention,
stratified by (in-context vs novel) x (rare vs frequent target token).
The tokens that truly require copying are the rare in-context ones.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
    ap.add_argument("--hidden-layers", default="3,6,9")
    ap.add_argument("--rare-max-count", type=int, default=5)
    args = ap.parse_args()

    elh = load_elh()
    cfg = elh.load_config(args.config)
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    chunks = layer_chunks(L, H)
    D = chunks[-1][1]
    dump = Path(args.dump_dir)
    fourway, predictor = elh.load_fourway(args.ckpt, cfg, device)
    hid_layers = [int(x) for x in args.hidden_layers.split(",")]

    corp = torch.load(dump / "corpora.pt", map_location="cpu", weights_only=False)
    ids, labels = corp["eval_ids"], corp["eval_labels"]
    N, T = ids.shape
    PERIOD = 128

    edit = {"vec": None}
    cap_out: dict[int, torch.Tensor] = {}
    cap_in: dict[int, torch.Tensor] = {}
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            a, b = chunks[i]
            def hook(m, inp, out):
                cap_out[i] = out.detach()
                cap_in[i] = inp[0].detach()          # hidden state fed to this layer's corrections
                v = edit["vec"]
                return out if v is None else out + v[a:b].to(out.device, out.dtype).view(1, 1, -1)
            return hook
        mlp.register_forward_hook(make(i))

    @torch.no_grad()
    def fwd(row):
        row = row.to(device)
        return fourway(row, predictor(row))

    # ---- ground truth per token ----
    counts = Counter(ids.reshape(-1).tolist())
    rare = torch.tensor([[counts[int(t)] <= args.rare_max_count for t in lab] for lab in labels.tolist()])
    inctx = torch.zeros(N, T, dtype=torch.bool)
    for i in range(N):
        row, lab = ids[i], labels[i]
        for t in range(T):
            lo = max(0, t - PERIOD + 1)
            inctx[i, t] = bool((row[lo:t + 1] == lab[t]).any())

    # ---- pass 1: features + correctness (no intervention) ----
    edit["vec"] = None
    feats_corr, feats_hid, correct, nll0 = [], [], [], []
    with torch.no_grad():
        for i in range(N):
            cap_out.clear(); cap_in.clear()
            lg = fwd(ids[i:i + 1])
            pred = lg[0].argmax(-1).cpu()
            correct.append(pred == labels[i])
            nll0.append(F.cross_entropy(lg[0].float(), labels[i].to(device), reduction="none").cpu())
            feats_corr.append(torch.cat([cap_out[k].float().cpu()[0] for k in range(len(chunks))], -1).half())
            feats_hid.append(torch.cat([cap_in[l - 1].float().cpu()[0] for l in hid_layers], -1).half())
    correct = torch.stack(correct); nll0 = torch.stack(nll0)
    Xc = torch.stack(feats_corr).float().numpy()      # [N,T,D]
    Xh = torch.stack(feats_hid).float().numpy()       # [N,T,3072]
    event = (inctx & correct).numpy()
    print(f"[audit] copy-event rate {event.mean():.3f}; rare-token rate {rare.float().mean():.3f}")

    # ---- audit monitors ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    import interp_probe as ip
    tr = np.arange(N) < int(0.8 * N)
    def flat(X, m): return X[m].reshape(-1, X.shape[-1])
    ytr, yte = event[tr].reshape(-1).astype(int), event[~tr].reshape(-1).astype(int)
    itr, ite = ids.numpy()[tr].reshape(-1), ids.numpy()[~tr].reshape(-1)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(ytr), min(30000, len(ytr)), replace=False)

    def auc_probe(Xtr, Xte):
        sc = StandardScaler().fit(Xtr[sub])
        clf = LogisticRegression(max_iter=300).fit(sc.transform(Xtr[sub]), ytr[sub])
        return float(roc_auc_score(yte, clf.decision_function(sc.transform(Xte))))

    audit = {}
    audit["auc_corrections"] = auc_probe(flat(Xc, tr), flat(Xc, ~tr))
    res = ip.per_id_mean_alpha(itr, flat(Xc, tr))
    audit["auc_corrections_resid_tokenid"] = auc_probe(res(itr, flat(Xc, tr)), res(ite, flat(Xc, ~tr)))
    audit["auc_hidden_states_L%s" % args.hidden_layers] = auc_probe(flat(Xh, tr), flat(Xh, ~tr))
    # token-identity baseline: P(event | id) estimated on train split
    tab, cnt = {}, {}
    for i_, y_ in zip(itr, ytr):
        tab[int(i_)] = tab.get(int(i_), 0) + y_; cnt[int(i_)] = cnt.get(int(i_), 0) + 1
    prior = ytr.mean()
    score_id = np.array([tab[int(i_)] / cnt[int(i_)] if int(i_) in cnt else prior for i_ in ite])
    audit["auc_token_identity"] = float(roc_auc_score(yte, score_id))
    print(f"[audit] {audit}")

    # ---- selectivity stratification under intervention ----
    def corr_mean(rows, from_pos):
        edit["vec"] = None
        acc = torch.zeros(D); n = 0
        for i in range(rows.shape[0]):
            cap_out.clear(); fwd(rows[i:i + 1])
            acc += torch.cat([cap_out[k].float().cpu() for k in range(len(chunks))], -1)[0, from_pos:].mean(0); n += 1
        return acc / n
    fp = corr_mean(corp["repeat_ids"][:16], PERIOD) - corr_mean(corp["random_ids"][:16], PERIOD)

    groups = {
        "in_context_rare": (inctx & rare), "in_context_frequent": (inctx & ~rare),
        "novel_rare": (~inctx & rare), "novel_frequent": (~inctx & ~rare),
    }
    test = ~torch.tensor(tr)
    strat = {}
    for lam in (0.0, 0.5, 1.0):
        edit["vec"] = None if lam == 0 else -lam * fp
        per = []
        with torch.no_grad():
            for i in np.where(~tr)[0]:
                lg = fwd(ids[i:i + 1])
                per.append(F.cross_entropy(lg[0].float(), labels[i].to(device), reduction="none").cpu())
        per = torch.stack(per)
        row = {}
        for g, m in groups.items():
            mm = m[test]
            row[g] = {"nll": per[mm].mean().item(), "frac": mm.float().mean().item()}
        strat[f"lam{lam}"] = row
        print(f"[audit] lam={lam}: " + ", ".join(f"{g}={v['nll']:.3f}" for g, v in row.items()))
    edit["vec"] = None

    save_json({"audit": audit, "stratified": strat, "rare_max_count": args.rare_max_count,
               "hidden_layers": hid_layers}, Path(args.out) / "audit_step9000.json")
    print("[audit] DONE")


if __name__ == "__main__":
    main()
