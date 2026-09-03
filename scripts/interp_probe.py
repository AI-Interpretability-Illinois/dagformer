"""Linear probes on per-token routing weights (alpha): does the topology
predictor encode interpretable context features, beyond current-token identity?

Feature sets per token t:
  alpha        flat routing vector [D]
  alpha_resid  alpha minus the per-token-id mean alpha (train-estimated) —
               any decodability left is context information by construction
Baselines:
  majority     global majority class
  token_id     conditional-majority lookup on the current token id (the
               "current token explains it" null hypothesis)

Targets:
  charclass    character class of current token (sanity: alpha sees the token)
  posbucket    position quartile in the 1024 window (context)
  freqbucket   corpus-frequency quartile of current token
  isrepeat64   current token occurred in the previous 64 tokens (context)
  nllquartile  difficulty of the NEXT-token prediction (from swap_eval dump)
  domain       3-class source style (code / prose / latex)

Also: induction delta-alpha analysis (repeat vs matched random sequences) and
emergence curves (probe accuracy across training checkpoints).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from interp_common import load_elh, save_json

RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def token_charclass(tok_str: str) -> int:
    s = tok_str.replace("Ġ", " ").replace("Ċ", "\n")
    t = s.strip()
    if t == "":
        return 0                       # whitespace
    if all(not c.isalnum() for c in t):
        return 1                       # punctuation/symbol
    if any(c.isdigit() for c in t) and not any(c.isalpha() for c in t):
        return 2                       # numeric
    alpha = [c for c in t if c.isalpha()]
    if alpha and alpha[0].isupper():
        return 3                       # capitalized
    if alpha:
        return 4                       # lowercase word(piece)
    return 5                           # mixed


def build_targets(ids: torch.Tensor, tok) -> dict[str, np.ndarray]:
    """ids [N,T] -> per-token label arrays [N,T]."""
    N, T = ids.shape
    uniq = torch.unique(ids).tolist()
    tok_strs = tok.convert_ids_to_tokens(uniq)
    cls_map = {i: token_charclass(s) for i, s in zip(uniq, tok_strs)}

    counts = Counter(ids.reshape(-1).tolist())
    occ_counts = np.array([counts[i.item()] for i in ids.reshape(-1)])
    qs = np.quantile(occ_counts, [0.25, 0.5, 0.75])
    freq = np.digitize(occ_counts, qs).reshape(N, T)

    charcls = np.array([[cls_map[i] for i in row] for row in ids.tolist()])
    pos = np.tile(np.digitize(np.arange(T), [T // 4, T // 2, 3 * T // 4]), (N, 1))

    isrep = np.zeros((N, T), dtype=np.int64)
    arr = ids.numpy()
    for n in range(N):
        for t in range(T):
            lo = max(0, t - 64)
            isrep[n, t] = int(arr[n, t] in arr[n, lo:t]) if t > lo else 0

    return {"charclass": charcls, "posbucket": pos,
            "freqbucket": freq, "isrepeat64": isrep}


# ---------------------------------------------------------------------------
# Probes & baselines
# ---------------------------------------------------------------------------

def fit_probe(Xtr, ytr, Xte, yte, max_train: int = 30000) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if Xtr.shape[0] > max_train:
        idx = RNG.choice(Xtr.shape[0], max_train, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=300, C=1.0, n_jobs=-1)
    clf.fit(sc.transform(Xtr), ytr)
    return float(clf.score(sc.transform(Xte), yte)), clf, sc


def token_id_baseline(ids_tr, y_tr, ids_te, y_te) -> float:
    table = defaultdict(Counter)
    for i, y in zip(ids_tr, y_tr):
        table[int(i)][int(y)] += 1
    glob = Counter(y_tr.tolist()).most_common(1)[0][0]
    pred = np.array([table[int(i)].most_common(1)[0][0] if table[int(i)] else glob
                     for i in ids_te])
    return float((pred == y_te).mean())


def majority_baseline(y_tr, y_te) -> float:
    m = Counter(y_tr.tolist()).most_common(1)[0][0]
    return float((y_te == m).mean())


def per_id_mean_alpha(ids_flat: np.ndarray, X: np.ndarray):
    """Fit id -> mean alpha on train tokens; return residualizer fn."""
    uniq, inv = np.unique(ids_flat, return_inverse=True)
    sums = np.zeros((len(uniq), X.shape[1]), dtype=np.float64)
    np.add.at(sums, inv, X)
    cnt = np.bincount(inv).astype(np.float64)
    means = (sums / cnt[:, None]).astype(np.float32)
    lookup = {int(u): k for k, u in enumerate(uniq)}
    gmean = X.mean(0)

    def residualize(ids_q: np.ndarray, Xq: np.ndarray) -> np.ndarray:
        M = np.array([means[lookup[int(i)]] if int(i) in lookup else gmean
                      for i in ids_q], dtype=np.float32)
        return Xq - M
    return residualize


# ---------------------------------------------------------------------------

def agg_by_stream_layer(vec: np.ndarray, layout: list[dict]) -> dict:
    box = defaultdict(list)
    for v, lab in zip(vec, layout):
        box[f"{lab['stream']}/L{lab['layer']}"].append(abs(float(v)))
    return {k: float(np.mean(v)) for k, v in sorted(box.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dump-dir", default="experiments/results/interp/alpha_dumps")
    ap.add_argument("--swap-nll", default=None,
                    help="per_token_nll_*.pt from interp_swap_eval.py")
    ap.add_argument("--main-step", type=int, default=9000)
    ap.add_argument("--n-calib", type=int, default=25,
                    help="must match interp_swap_eval (its test half = seqs n_calib:)")
    ap.add_argument("--out", default="experiments/results/interp")
    args = ap.parse_args()

    elh = load_elh()
    cfg = elh.load_config(args.config)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])
    dump = Path(args.dump_dir)
    layout = json.load(open(dump / "layout.json"))
    corp = torch.load(dump / "corpora.pt", map_location="cpu", weights_only=False)
    ids = corp["eval_ids"]
    N, T = ids.shape
    alpha = torch.load(dump / f"alpha_eval_step{args.main_step}.pt",
                       map_location="cpu").float().numpy()
    print(f"[probe] alpha {alpha.shape}, eval ids {tuple(ids.shape)}")

    targets = build_targets(ids, tok)

    # Difficulty target from swap eval (covers seqs n_calib: only)
    if args.swap_nll and Path(args.swap_nll).exists():
        ptn = torch.load(args.swap_nll, map_location="cpu", weights_only=False)
        dyn = ptn["dynamic"].numpy()                      # [n_test, T]
        qs = np.quantile(dyn.reshape(-1), [0.25, 0.5, 0.75])
        nllq = np.digitize(dyn, qs)
        full = np.full((N, T), -1, dtype=np.int64)
        full[args.n_calib:args.n_calib + dyn.shape[0]] = nllq
        targets["nllquartile"] = full

    # Split by sequence
    tr_seq = np.arange(N) < int(0.8 * N)
    results = {}
    coef_maps = {}
    for tname, y in targets.items():
        valid = y >= 0
        Xtr = alpha[tr_seq][valid[tr_seq]]
        Xte = alpha[~tr_seq][valid[~tr_seq]]
        ytr = y[tr_seq][valid[tr_seq]]
        yte = y[~tr_seq][valid[~tr_seq]]
        itr = ids.numpy()[tr_seq][valid[tr_seq]]
        ite = ids.numpy()[~tr_seq][valid[~tr_seq]]
        if len(np.unique(ytr)) < 2 or len(yte) == 0:
            print(f"[probe] skip {tname}: degenerate")
            continue
        acc_alpha, clf, _ = fit_probe(Xtr, ytr, Xte, yte)
        residualize = per_id_mean_alpha(itr, Xtr)
        acc_resid, _, _ = fit_probe(residualize(itr, Xtr), ytr,
                                    residualize(ite, Xte), yte)
        res = {
            "alpha": acc_alpha,
            "alpha_resid": acc_resid,
            "token_id": token_id_baseline(itr, ytr, ite, yte),
            "majority": majority_baseline(ytr, yte),
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
        }
        results[tname] = res
        coef = np.abs(clf.coef_).mean(0)
        coef_maps[tname] = agg_by_stream_layer(coef, layout)
        print(f"[probe] {tname:12s} alpha={acc_alpha:.3f} resid={acc_resid:.3f} "
              f"token_id={res['token_id']:.3f} majority={res['majority']:.3f}")

    # ---- Domain probe ----
    dom_names = sorted(k for k in corp["domains"])
    if len(dom_names) >= 2:
        Xd, yd, idd, wind = [], [], [], []
        for di, name in enumerate(dom_names):
            a = torch.load(dump / f"alpha_domain_{name}_step{args.main_step}.pt",
                           map_location="cpu").float().numpy()
            w, t_len, _ = a.shape
            Xd.append(a.reshape(-1, a.shape[-1]))
            yd.append(np.full(w * t_len, di))
            idd.append(corp["domains"][name].numpy().reshape(-1))
            wind.append(np.repeat(np.arange(w), t_len) + 100 * di)
        Xd = np.concatenate(Xd); yd = np.concatenate(yd)
        idd = np.concatenate(idd); wind = np.concatenate(wind)
        te_wins = {w for di in range(len(dom_names))
                   for w in (100 * di, 100 * di + 1)}   # 2 held-out windows/domain
        te = np.isin(wind, list(te_wins)); tr = ~te
        acc_alpha, _, _ = fit_probe(Xd[tr], yd[tr], Xd[te], yd[te])
        residualize = per_id_mean_alpha(idd[tr], Xd[tr])
        acc_resid, _, _ = fit_probe(residualize(idd[tr], Xd[tr]), yd[tr],
                                    residualize(idd[te], Xd[te]), yd[te])
        results["domain"] = {
            "alpha": acc_alpha, "alpha_resid": acc_resid,
            "token_id": token_id_baseline(idd[tr], yd[tr], idd[te], yd[te]),
            "majority": majority_baseline(yd[tr], yd[te]),
            "classes": dom_names,
        }
        print(f"[probe] domain       alpha={acc_alpha:.3f} resid={acc_resid:.3f} "
              f"token_id={results['domain']['token_id']:.3f}")

    # ---- Induction delta-alpha (repeat vs matched random, positions >=128) ----
    a_rep = torch.load(dump / f"alpha_repeat_step{args.main_step}.pt",
                       map_location="cpu").float()
    a_rnd = torch.load(dump / f"alpha_random_step{args.main_step}.pt",
                       map_location="cpu").float()
    delta = (a_rep[:, 128:].mean(dim=(0, 1)) - a_rnd[:, 128:].mean(dim=(0, 1))).numpy()
    top = np.argsort(-np.abs(delta))[:20]
    induction = {
        "by_stream_layer": agg_by_stream_layer(delta, layout),
        "top_entries": [{**layout[i], "delta": float(delta[i])} for i in top],
        "mean_abs_delta": float(np.abs(delta).mean()),
    }
    print(f"[induction] mean|delta|={induction['mean_abs_delta']:.4f}; "
          f"top entry: {induction['top_entries'][0]}")

    # ---- Emergence: probe accuracy across checkpoints (subsampled dumps) ----
    emergence = {}
    sub_targets = {k: v[:, ::2] for k, v in targets.items()
                   if k in ("isrepeat64", "charclass")}
    for p in sorted(dump.glob("alpha_eval_sub_step*.pt")):
        step = int(re.search(r"step(\d+)", p.name).group(1))
        a = torch.load(p, map_location="cpu").float().numpy()
        row = {}
        for tname, y in sub_targets.items():
            Xtr, Xte = a[tr_seq].reshape(-1, a.shape[-1]), a[~tr_seq].reshape(-1, a.shape[-1])
            ytr, yte = y[tr_seq].reshape(-1), y[~tr_seq].reshape(-1)
            row[tname], _, _ = fit_probe(Xtr, ytr, Xte, yte, max_train=20000)
        emergence[step] = row
        print(f"[emergence] step {step}: {row}")

    save_json({"probes": results, "coef_by_stream_layer": coef_maps,
               "induction": induction, "emergence": emergence,
               "main_step": args.main_step},
              Path(args.out) / f"probe_results_step{args.main_step}.json")
    print("[probe] DONE")


if __name__ == "__main__":
    main()
