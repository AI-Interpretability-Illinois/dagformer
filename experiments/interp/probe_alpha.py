"""E1 — if the structure predictor does not encode behaviour, what does it encode?

`routing_leverage.py` found that alpha_pred varies by only 4.2% of its own
magnitude across tokens and 6.6% across prompts.  Small, but not zero.  This
asks what that residual variation is *about*, by decoding candidate variables
out of alpha_pred with linear probes:

    context_free    how much of alpha_pred is just a lookup on the current
                    token id -- fit the per-token-id mean and measure what
                    fraction of the variance survives.  This is the headline:
                    if nothing survives, the "predicted topology" is a static
                    embedding table and there is no per-context structure to
                    interpret at all.
    token           current token id (top-C most frequent classes)
    prev_token      previous token id -- the cheapest test for context
    position        absolute position in the window (ridge, R^2)
    log_freq        log corpus frequency rank of the current token (ridge)
    polarity        pos vs neg instruction, from the behaviour prompt set

Every probe is scored on held-out data grouped so that no window (or, for
polarity, no item) spans the split, and every probe is run again on shuffled
labels so the reported number has its own null beside it.

The intended substitutions: no POS tagger or parser is installed in this
environment, so `prev_token` and `log_freq` stand in for the syntactic probes
-- they test whether alpha_pred sees context and lexical class at all, which
is the prerequisite those probes would have been measuring.

Only the predictor is loaded (`need_base=False`), so this does not need the
2 GB base model.

Usage:
    python experiments/interp/probe_alpha.py --config ... --ckpt ... \
        --behavior domain_code --out .../probe_alpha
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from behaviors import get_behavior
from circuit_common import (RoutingLayout, RoutingRunner, build_prompt_set,
                            capability_windows, load_models, load_tokenizer,
                            save_json)

DEFAULT_TOKENIZER = "/work/hdd/bfqt/shared/dagformer-models/tokenizer"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--behavior", default="domain_code")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-windows", type=int, default=32)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--cap-cache", default=None,
                    help="cached eval batch for the probe corpus (else repo prose)")
    ap.add_argument("--n-classes", type=int, default=40,
                    help="most frequent token ids kept for the token probes")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--n-shuffles", type=int, default=5)
    ap.add_argument("--eligible", default=None,
                    help="*_stats.npz; restricts the probe features to eligible edges")
    ap.add_argument("--hyper-only", action="store_true",
                    help="probe only the true hyperconnection coordinates")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Probes.  sklearn is available in this environment (1.7.2), but the models are
# kept deliberately plain: a linear probe is a claim about what is *linearly*
# present, and any tuning knob would blur that claim.
# ---------------------------------------------------------------------------

def _split(groups: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a third of the groups, so no group straddles the split."""
    uniq = np.unique(groups)
    held = rng.choice(uniq, size=max(1, len(uniq) // 3), replace=False)
    te = np.isin(groups, held)
    return ~te, te


def probe_classify(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   rng: np.random.Generator, n_shuffles: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr, te = _split(groups, rng)

    def fit(labels):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], labels[tr])
        return float((clf.predict(X[te]) == y[te]).mean())

    acc = fit(y)
    null = [fit(rng.permutation(y)) for _ in range(n_shuffles)]
    vals, cnt = np.unique(y[tr], return_counts=True)
    major = float((y[te] == vals[cnt.argmax()]).mean())
    return {"kind": "accuracy", "score": acc, "null_mean": float(np.mean(null)),
            "null_max": float(np.max(null)), "majority": major,
            "n_classes": int(len(np.unique(y))), "n_train": int(tr.sum()),
            "n_test": int(te.sum())}


def probe_regress(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                  rng: np.random.Generator, n_shuffles: int) -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    tr, te = _split(groups, rng)

    def fit(t):
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(X[tr], t[tr])
        resid = ((y[te] - m.predict(X[te])) ** 2).mean()
        return float(1.0 - resid / max(y[te].var(), 1e-12))

    r2 = fit(y)
    null = [fit(rng.permutation(y)) for _ in range(n_shuffles)]
    return {"kind": "r2", "score": r2, "null_mean": float(np.mean(null)),
            "null_max": float(np.max(null)), "majority": 0.0,
            "n_train": int(tr.sum()), "n_test": int(te.sum())}


def _within_fraction(A: np.ndarray, key: np.ndarray) -> float:
    """Share of alpha_pred variance left after conditioning on `key`."""
    total = A.var(0).sum()
    within = sum(A[key == v].var(0).sum() * int((key == v).sum())
                 for v in np.unique(key))
    return float(within / len(key) / max(total, 1e-12))


def context_free_fraction(A: np.ndarray, tok: np.ndarray,
                          pos: np.ndarray) -> dict:
    """How much alpha_pred variance survives conditioning on the current token.

    If alpha_pred were a pure per-token lookup this is 0: every occurrence of
    the same token id would carry exactly the same routing weights regardless
    of what came before it.

    Absolute position turns out to dominate alpha_pred, and position is not
    context in any interesting sense, so the same number is reported again
    after subtracting the per-position mean.  That residual figure is the one
    that answers "does alpha_pred respond to what the text says".
    """
    ids, counts = np.unique(tok, return_counts=True)
    rep = set(ids[counts >= 2].tolist())
    keep = np.array([int(t) in rep for t in tok])
    A, tok, pos = A[keep], tok[keep], pos[keep]
    R = A.copy()
    for p in np.unique(pos):                       # nonparametric position removal
        m = pos == p
        R[m] -= A[m].mean(0)
    return {"frac_variance_within_token": _within_fraction(A, tok),
            "frac_variance_within_token_pos_removed": _within_fraction(R, tok),
            "frac_variance_within_position": _within_fraction(A, pos),
            "n_positions": int(len(tok)), "n_token_types": int(len(np.unique(tok)))}


# ---------------------------------------------------------------------------

@torch.no_grad()
def alpha_for(runner: RoutingRunner, ids: torch.Tensor, bs: int) -> torch.Tensor:
    return torch.cat([runner.alpha_pred(ids[i:i + bs]) for i in range(0, len(ids), bs)])


def report(res: dict) -> str:
    cf = res["context_free"]
    L = ["# What does alpha_pred encode?", "",
         f"Corpus: {res['n_positions']} positions, {res['n_features']} routing "
         f"coordinates.  Behaviour: `{res['behavior']}`.", "",
         f"Variance of alpha_pred left unexplained by "
         f"({cf['n_token_types']} token types over {cf['n_positions']} positions):", "",
         f"- absolute position: **{cf['frac_variance_within_position']:.4f}**",
         f"- current token id: **{cf['frac_variance_within_token']:.4f}**",
         f"- current token id, after the per-position mean is removed: "
         f"**{cf['frac_variance_within_token_pos_removed']:.4f}**", "",
         "The last figure is the one that answers whether alpha_pred responds "
         "to context at all: near 0 means the predicted topology is a lookup "
         "table on (position, token) with nothing else in it.", "",
         "| probe | metric | score | shuffled-label null | trivial baseline |",
         "|---|---|---|---|---|"]
    for name, p in res["probes"].items():
        L.append(f"| {name} | {p['kind']} | {p['score']:.3f} | "
                 f"{p['null_mean']:.3f} (max {p['null_max']:.3f}) | "
                 f"{p['majority']:.3f} |")
    return "\n".join(L) + "\n"


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    cfg, _, predictor = load_models(args.config, args.ckpt, device, need_base=False)
    layout = RoutingLayout(cfg["num_hidden_layers"], cfg["num_attention_heads"])
    tokenizer = load_tokenizer(cfg, args.tokenizer)
    runner = RoutingRunner(layout, predictor, None, device=device)

    feat = np.ones(layout.D, dtype=bool)
    if args.eligible:
        feat &= np.load(args.eligible)["eligible"].astype(bool)
    if args.hyper_only:
        feat &= layout.hyper_arr
    print(f"[probe] L={layout.L} H={layout.H} D={layout.D}, "
          f"{int(feat.sum())} feature coordinates", flush=True)

    # -- corpus ------------------------------------------------------------
    win = capability_windows(tokenizer, args.seq_len, args.n_windows, args.cap_cache)
    A = alpha_for(runner, win, args.batch_size).numpy()          # [N, T, D]
    N, T, _ = A.shape
    print(f"[probe] alpha_pred over {N} windows x {T} tokens", flush=True)

    toks = win.numpy()
    # drop position 0: it has no previous token
    X = A[:, 1:, :][:, :, feat].reshape(-1, int(feat.sum())).astype(np.float64)
    cur = toks[:, 1:].reshape(-1)
    prev = toks[:, :-1].reshape(-1)
    pos = np.tile(np.arange(1, T), N).astype(np.float64)
    grp = np.repeat(np.arange(N), T - 1)

    # frequency rank of the current token within this corpus
    ids, counts = np.unique(toks.reshape(-1), return_counts=True)
    rank = {int(t): float(np.log1p(r)) for r, t in
            enumerate(ids[np.argsort(-counts)])}
    logfreq = np.array([rank[int(t)] for t in cur])

    probes: dict[str, dict] = {}
    probes["position"] = probe_regress(X, pos, grp, rng, args.n_shuffles)
    print(f"  position     R2={probes['position']['score']:.3f}", flush=True)
    probes["log_freq_rank"] = probe_regress(X, logfreq, grp, rng, args.n_shuffles)
    print(f"  log_freq     R2={probes['log_freq_rank']['score']:.3f}", flush=True)

    top = set(ids[np.argsort(-counts)][:args.n_classes].tolist())
    for name, lab in (("token", cur), ("prev_token", prev)):
        m = np.array([int(t) in top for t in lab])
        probes[name] = probe_classify(X[m], lab[m], grp[m], rng, args.n_shuffles)
        print(f"  {name:<12} acc={probes[name]['score']:.3f} "
              f"(majority {probes[name]['majority']:.3f})", flush=True)

    cf = context_free_fraction(X, cur, pos)
    print(f"  variance left after token id {cf['frac_variance_within_token']:.4f} "
          f"(position removed first: "
          f"{cf['frac_variance_within_token_pos_removed']:.4f}); "
          f"after position {cf['frac_variance_within_position']:.4f}", flush=True)

    # -- instruction polarity ---------------------------------------------
    spec = get_behavior(args.behavior)
    ps = build_prompt_set(spec, tokenizer, max_items=args.max_items)
    lo, hi = ps.span
    sel = [p for p in ps.prompts if p.polarity in ("pos", "neg")]
    pids = torch.tensor([p.ids for p in sel], dtype=torch.long)
    Ap = alpha_for(runner, pids, args.batch_size).numpy()
    Xp = Ap[:, lo:hi, :].mean(1)[:, feat].astype(np.float64)
    yp = np.array([p.polarity == "pos" for p in sel]).astype(int)
    gp = np.array([p.item for p in sel])          # split by item, not paraphrase
    probes["polarity"] = probe_classify(Xp, yp, gp, rng, args.n_shuffles)
    print(f"  polarity     acc={probes['polarity']['score']:.3f} "
          f"(majority {probes['polarity']['majority']:.3f})", flush=True)

    res = {"args": vars(args), "behavior": args.behavior, "probes": probes,
           "context_free": cf, "n_positions": int(X.shape[0]),
           "n_features": int(feat.sum()),
           "note": "scores are held-out; groups (windows, or items for polarity) "
                   "do not straddle the split; nulls retrain on shuffled labels"}
    text = report(res)
    print("\n" + text)
    if args.out:
        out = Path(args.out)
        save_json(res, out.with_suffix(".json"))
        out.with_suffix(".md").write_text(text)
        print(f"[saved] {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
