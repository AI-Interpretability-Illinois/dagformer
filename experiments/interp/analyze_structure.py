"""How compressible is the routing difference?

Stage 3.5 of the pipeline, between discovery and verification, and the only
stage that needs no model: it reads the `*_stats.npz` written by
`discover_circuit.py` and asks what kind of object `Delta` is.

The motivation is that edge-level sparsity is a proxy for what actually makes
a circuit interpretable -- a short description in a basis whose elements have
names -- and it is the *weakest* form that property can take.  A `Delta` that
is dense in edges may still be sparse in heads, or rank-2, or constant within
(stream, source-distance) cells, and each of those is a sentence a person can
read.  So instead of asking "is it sparse?" this script walks a ladder of
compressions and reports the cheapest one that still reproduces `Delta`:

    top-k edges            edge-sparse           k
    head groups            group-sparse          #coords in the kept heads
    rank-r                 low-rank              r * (rows + cols)
    cell averages          structured-dense      #cells
    one scalar             unstructured          1

Fidelity here is *structural* -- cosine with the real `Delta` over eligible
coordinates.  It is a necessary condition, not a sufficient one: a compression
that cannot reproduce the vector cannot reproduce its causal effect, but one
that can still has to be run through `verify_circuit.py` to earn the claim.
That is what `--emit-masks` is for.

Every concentration number is reported against a permutation null (the same
`Delta` values shuffled across eligible coordinates), because a dense vector
with a heavy-tailed value distribution concentrates in *any* grouping by
chance, and without the null the tables below all look like structure.

That permutation null is necessary but far too weak on its own, and the first
run of this script is the reason the `--contrast` option exists.  It found
that `honesty_fewshot` -- a behaviour where *zero* edges survive selection,
i.e. pure noise -- compresses *better* than `domain_code`: rank-1 cosine 0.944
against 0.859.  Routing differences are low-rank whether or not a behaviour
caused them, because alpha itself is low-rank across tokens.  Shuffling
destroys that shared structure and so scores it as signal.

The honest control re-runs the whole ladder on the **balanced-split null**
(`discover_circuit.balanced_null`): re-partition the same paraphrases so both
sides carry the same polarity mix, which cancels the behavioural term exactly
while leaving every other source of routing variation in place.  A structural
claim survives only if the real `Delta` compresses *better than those draws*.

Usage
-----
    python analyze_structure.py --stats results/circuits/circuit_*_stats.npz
    python analyze_structure.py --stats <f> --contrast contrast_<behavior>.pt
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from circuit_common import RoutingLayout, save_json
from discover_circuit import balanced_null, group_alpha


# ---------------------------------------------------------------------------
# Matrix view of the flat routing vector
# ---------------------------------------------------------------------------

def to_matrix(layout: RoutingLayout, values: np.ndarray,
              eligible: np.ndarray) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """Flat [D] -> dense [n_slots, L] matrix plus a validity mask.

    A *slot* is a target (stream, layer, head); the columns are sources
    s = 0..L-1.  The matrix is ragged by construction (layer l only has
    sources s <= l), so `W` marks which cells exist and are eligible; every
    consumer below is mask-aware.
    """
    slots: dict[tuple, int] = {}
    rows: list[tuple] = []
    for e in layout.edges:
        key = (e.stream, e.layer, e.head)
        if key not in slots:
            slots[key] = len(rows)
            rows.append(key)
    M = np.zeros((len(rows), layout.L))
    W = np.zeros((len(rows), layout.L), dtype=bool)
    idx = np.full((len(rows), layout.L), -1, dtype=np.int64)
    for i, e in enumerate(layout.edges):
        r, c = slots[(e.stream, e.layer, e.head)], e.src
        M[r, c] = values[i]
        W[r, c] = eligible[i]
        idx[r, c] = i
    return M, W, rows, idx


def masked_lowrank(M: np.ndarray, W: np.ndarray, r: int, iters: int = 200,
                   ridge: float = 1e-6, seed: int = 0) -> np.ndarray:
    """Rank-r fit of M over the cells where W, by alternating ridge least squares.

    Plain SVD on a zero-filled ragged matrix spends its first component
    explaining the padding, so the invalid cells have to be excluded from the
    objective rather than set to zero.
    """
    rng = np.random.default_rng(seed)
    n, m = M.shape
    U = rng.normal(scale=0.01, size=(n, r))
    V = rng.normal(scale=0.01, size=(m, r))
    Wf = W.astype(float)
    for _ in range(iters):
        for i in range(n):
            w = Wf[i]
            if w.sum() == 0:
                continue
            A = (V * w[:, None]).T @ V + ridge * np.eye(r)
            U[i] = np.linalg.solve(A, (V * w[:, None]).T @ M[i])
        for j in range(m):
            w = Wf[:, j]
            if w.sum() == 0:
                continue
            A = (U * w[:, None]).T @ U + ridge * np.eye(r)
            V[j] = np.linalg.solve(A, (U * w[:, None]).T @ M[:, j])
    return U @ V.T


# ---------------------------------------------------------------------------
# Fidelity of a reconstruction
# ---------------------------------------------------------------------------

def fidelity(delta: np.ndarray, recon: np.ndarray, eligible: np.ndarray) -> dict:
    """Cosine and explained variance of `recon` against `delta`, over eligible.

    Cosine is the headline because `verify_circuit.py` rescales every edit to a
    matched norm, so a reconstruction that gets the direction right and the
    magnitude wrong loses nothing downstream.
    """
    a, b = delta[eligible], recon[eligible]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos = float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
    ev = float(1.0 - np.sum((a - b) ** 2) / np.sum(a ** 2)) if na > 0 else 0.0
    return {"cosine": cos, "explained_var": ev}


def group_keys(layout: RoutingLayout, by: str) -> np.ndarray:
    """Integer group id per edge for one of the named groupings."""
    L, s, l, h, src = layout, layout.stream_arr, layout.layer_arr, layout.head_arr, layout.src_arr
    if by == "slot":       cols = (s, l, h)
    elif by == "head":     cols = (l, h)
    elif by == "layer":    cols = (l,)
    elif by == "stream":   cols = (s,)
    elif by == "src":      cols = (src,)
    elif by == "srcdist":  cols = (l - src,)
    elif by == "stream_srcdist":  cols = (s, l - src)
    elif by == "stream_layer":    cols = (s, l)
    elif by == "stream_layer_srcdist": cols = (s, l, l - src)
    else: raise ValueError(f"unknown grouping {by!r}")
    _ = L
    tup = np.array(["|".join(str(c[i]) for c in cols) for i in range(layout.D)])
    return np.unique(tup, return_inverse=True)[1]


def concentration(delta: np.ndarray, eligible: np.ndarray, gids: np.ndarray,
                  tops=(1, 2, 4, 8, 16)) -> dict:
    """Share of |Delta| mass held by the heaviest groups, and groups for 50%."""
    mass = np.abs(delta) * eligible
    n_g = gids.max() + 1
    per = np.bincount(gids, weights=mass, minlength=n_g)
    live = per[np.bincount(gids, weights=eligible.astype(float), minlength=n_g) > 0]
    tot = live.sum()
    if tot <= 0:
        return {"n_groups": int(live.size), "top": {}, "n_for_50pct": 0}
    order = np.sort(live)[::-1]
    cum = np.cumsum(order) / tot
    return {"n_groups": int(live.size),
            "top": {str(k): float(cum[min(k, cum.size) - 1]) for k in tops},
            "n_for_50pct": int(np.searchsorted(cum, 0.5) + 1)}


def perm_null(delta: np.ndarray, eligible: np.ndarray, gids: np.ndarray,
              n_draws: int, rng: np.random.Generator, tops=(1, 2, 4, 8, 16)) -> dict:
    """Same concentration statistic with Delta shuffled across eligible edges."""
    idx = np.flatnonzero(eligible)
    draws = {str(k): [] for k in tops}
    fifty = []
    d = delta.copy()
    for _ in range(n_draws):
        d[idx] = delta[rng.permutation(idx)]
        c = concentration(d, eligible, gids, tops)
        for k in tops:
            draws[str(k)].append(c["top"][str(k)])
        fifty.append(c["n_for_50pct"])
    return {"top_mean": {k: float(np.mean(v)) for k, v in draws.items()},
            "top_p95": {k: float(np.percentile(v, 95)) for k, v in draws.items()},
            "n_for_50pct_mean": float(np.mean(fifty))}


# ---------------------------------------------------------------------------
# Compressions
# ---------------------------------------------------------------------------

def recon_topk(delta: np.ndarray, eligible: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros_like(delta)
    idx = np.flatnonzero(eligible)
    keep = idx[np.argsort(np.abs(delta[idx]))[::-1][:k]]
    out[keep] = delta[keep]
    return out


def recon_groups(delta: np.ndarray, eligible: np.ndarray, gids: np.ndarray,
                 m: int) -> tuple[np.ndarray, int]:
    """Keep every eligible edge inside the m heaviest groups, verbatim."""
    mass = np.abs(delta) * eligible
    per = np.bincount(gids, weights=mass, minlength=gids.max() + 1)
    keep_g = np.argsort(per)[::-1][:m]
    sel = np.isin(gids, keep_g) & eligible
    out = np.zeros_like(delta)
    out[sel] = delta[sel]
    return out, int(sel.sum())


def recon_cells(delta: np.ndarray, eligible: np.ndarray, gids: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace every edge by the mean Delta of its cell: the structured-dense rung."""
    n_g = gids.max() + 1
    cnt = np.bincount(gids, weights=eligible.astype(float), minlength=n_g)
    tot = np.bincount(gids, weights=delta * eligible, minlength=n_g)
    mean = np.divide(tot, cnt, out=np.zeros_like(tot), where=cnt > 0)
    out = mean[gids] * eligible
    return out, int((cnt > 0).sum())


# ---------------------------------------------------------------------------
# Connectivity of the selected edge set
# ---------------------------------------------------------------------------

def connectivity(layout: RoutingLayout, mask: np.ndarray) -> dict:
    """Do selected edges chain into paths?

    Edge (stream, l, h, s) carries the output of layer s-1 (or the embedding,
    s = 0) into layer l.  Edge B follows edge A when B reads what A wrote,
    i.e. `B.src - 1 == A.layer`.  A set that is merely a collection of
    independent nudges will have far fewer chained edges than one that routes
    signal along a path.
    """
    sel = np.flatnonzero(mask)
    if sel.size == 0:
        return {"n": 0, "frac_with_pred": 0.0, "frac_with_succ": 0.0,
                "longest_path": 0, "n_nodes": 0}
    lay, src = layout.layer_arr[sel], layout.src_arr[sel]
    produced = set(lay.tolist())              # layers the set writes into
    consumed = set((src - 1).tolist())        # layers the set reads from
    has_pred = np.isin(src - 1, list(produced))
    has_succ = np.isin(lay, list(consumed))
    # longest chain over target layers, by layer order (the graph is a DAG in l)
    depth: dict[int, int] = {}
    for l in sorted(produced):
        srcs = (src - 1)[lay == l]
        depth[l] = 1 + max((depth.get(int(p), 0) for p in srcs), default=0)
    return {"n": int(sel.size),
            "frac_with_pred": float(has_pred.mean()),
            "frac_with_succ": float(has_succ.mean()),
            "longest_path": int(max(depth.values())) if depth else 0,
            "n_nodes": len({(int(a), int(b)) for a, b in
                            zip(lay, layout.head_arr[sel])})}


def ladder_summary(layout: RoutingLayout, delta: np.ndarray, eligible: np.ndarray,
                   args, seed: int = 0) -> dict:
    """Cosine of each compression against `delta`, as one flat dict.

    Compact enough to run on every balanced-null draw, which is what turns the
    ladder from a description into a test.
    """
    out = {}
    for k in args.topk:
        out[f"top{k}"] = fidelity(delta, recon_topk(delta, eligible, k), eligible)["cosine"]
    gids = group_keys(layout, "head")
    for m in args.group_tops:
        rec, _ = recon_groups(delta, eligible, gids, m)
        out[f"head{m}"] = fidelity(delta, rec, eligible)["cosine"]
    M, W, _, idx = to_matrix(layout, delta, eligible)
    ok = W & (idx >= 0)
    for r in args.null_ranks:
        fit = masked_lowrank(M, W, r, iters=args.als_iters, seed=seed)
        rec = np.zeros_like(delta)
        rec[idx[ok]] = fit[ok]
        out[f"rank{r}"] = fidelity(delta, rec, eligible)["cosine"]
    uni = np.sign(delta) * np.abs(delta[eligible]).mean() * eligible
    out["sign_only"] = fidelity(delta, uni, eligible)["cosine"]
    return out


def behaviour_free_null(path: Path, channel: str, layout: RoutingLayout,
                        eligible: np.ndarray, args, rng: np.random.Generator) -> dict:
    """Run the ladder on balanced-split draws: same noise, no behaviour.

    These draws are the only control that preserves whatever structure alpha
    differences have for reasons unrelated to the instruction.  A compression
    that scores no better on the real Delta than on these has not found
    anything about the behaviour.
    """
    import torch
    d = torch.load(path, map_location="cpu", weights_only=False)
    alpha = d["alpha"][channel]
    n_items, var = d["n_items"], d["variants"]
    pos = group_alpha(alpha, d["meta"], "pos", n_items, var["pos"])
    neg = group_alpha(alpha, d["meta"], "neg", n_items, var["neg"])
    draws = balanced_null(pos, neg, args.n_contrast_draws, rng)
    acc: dict[str, list[float]] = {}
    for i, dr in enumerate(draws):
        for k, v in ladder_summary(layout, dr.mean(0), eligible, args, seed=i).items():
            acc.setdefault(k, []).append(v)
    return {"n_draws": len(draws),
            "mean": {k: float(np.mean(v)) for k, v in acc.items()},
            "p95": {k: float(np.percentile(v, 95)) for k, v in acc.items()}}


def connectivity_null(layout: RoutingLayout, eligible: np.ndarray, n: int,
                      n_draws: int, rng: np.random.Generator) -> dict:
    idx = np.flatnonzero(eligible)
    keys = ("frac_with_pred", "frac_with_succ", "longest_path", "n_nodes")
    acc = {k: [] for k in keys}
    for _ in range(n_draws):
        m = np.zeros(layout.D, dtype=bool)
        m[rng.choice(idx, size=min(n, idx.size), replace=False)] = True
        c = connectivity(layout, m)
        for k in keys:
            acc[k].append(c[k])
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ---------------------------------------------------------------------------

def analyse(path: Path, args, rng: np.random.Generator) -> dict:
    z = np.load(path)
    delta = z[args.which].astype(np.float64)
    eligible = z["eligible"].astype(bool)
    chosen = z["chosen"].astype(bool)
    meta = {}
    js = path.with_name(path.name.replace("_stats.npz", ".json"))
    if js.exists():
        d = json.loads(js.read_text())
        meta = {"behavior": d.get("behavior"), "channel": d.get("channel"),
                "layout": d.get("layout", {})}
    L = meta.get("layout", {}).get("L", 12)
    H = meta.get("layout", {}).get("H", 16)
    layout = RoutingLayout(L, H)
    assert layout.D == delta.size, f"{path}: layout D={layout.D} != delta {delta.size}"

    out: dict = {"file": str(path), "which": args.which, **meta,
                 "n_eligible": int(eligible.sum()), "n_chosen": int(chosen.sum())}

    # --- rung 3: low rank -------------------------------------------------
    M, W, rows, idx = to_matrix(layout, delta, eligible)
    ranks = {}
    for r in args.ranks:
        fit = masked_lowrank(M, W, r, iters=args.als_iters, seed=args.seed)
        rec = np.zeros_like(delta)
        ok = W & (idx >= 0)
        rec[idx[ok]] = fit[ok]
        ranks[str(r)] = {**fidelity(delta, rec, eligible),
                         "params": int(r * (M.shape[0] + M.shape[1]))}
    # null: same values, shuffled, same fit -- a random matrix is not rank-1
    dn = delta.copy()
    ix = np.flatnonzero(eligible)
    dn[ix] = delta[rng.permutation(ix)]
    Mn, Wn, _, idxn = to_matrix(layout, dn, eligible)
    fitn = masked_lowrank(Mn, Wn, 1, iters=args.als_iters, seed=args.seed)
    recn = np.zeros_like(delta)
    okn = Wn & (idxn >= 0)
    recn[idxn[okn]] = fitn[okn]
    ranks["1_null"] = fidelity(dn, recn, eligible)
    out["lowrank"] = ranks

    # --- rung 2 + 4: groups ----------------------------------------------
    groups = {}
    for by in args.groupings:
        gids = group_keys(layout, by)
        con = concentration(delta, eligible, gids)
        null = perm_null(delta, eligible, gids, args.n_null, rng)
        cells, n_cells = recon_cells(delta, eligible, gids)
        entry = {"concentration": con, "null": null,
                 "cell_average": {**fidelity(delta, cells, eligible), "params": n_cells}}
        if by in ("head", "slot"):
            entry["group_sparse"] = {}
            for m in args.group_tops:
                rec, n_kept = recon_groups(delta, eligible, gids, m)
                entry["group_sparse"][str(m)] = {
                    **fidelity(delta, rec, eligible), "params": n_kept}
        groups[by] = entry
    out["groups"] = groups

    # --- rung 1: top-k ----------------------------------------------------
    out["topk"] = {str(k): {**fidelity(delta, recon_topk(delta, eligible, k), eligible),
                            "params": k} for k in args.topk}

    # --- rung 5: one scalar ----------------------------------------------
    uni = np.sign(delta) * np.abs(delta[eligible]).mean() * eligible
    out["uniform_signed"] = {**fidelity(delta, uni, eligible), "params": 1}

    # --- connectivity of the discovered circuit --------------------------
    out["connectivity"] = {
        "chosen": connectivity(layout, chosen),
        "null": connectivity_null(layout, eligible, int(chosen.sum()),
                                  args.n_null, rng) if chosen.sum() else {}}

    # --- the control that matters: same noise, no behaviour --------------
    if args.contrast:
        ct = _match_contrast(path, args.contrast, meta.get("behavior"))
        if ct is not None:
            out["real_ladder"] = ladder_summary(layout, delta, eligible, args, args.seed)
            out["behaviour_free"] = behaviour_free_null(
                ct, meta.get("channel", "pred"), layout, eligible, args, rng)
    return out


def _match_contrast(stats: Path, patterns: list[str], behavior: str | None) -> Path | None:
    """Pick the contrast_<behavior>.pt that goes with this stats file."""
    cands = [Path(f) for pat in patterns for f in glob.glob(pat)]
    for c in cands:
        if behavior and c.stem == f"contrast_{behavior}":
            return c
    return cands[0] if len(cands) == 1 and not behavior else None


def report(res: dict) -> list[str]:
    L = [f"## {res.get('behavior')} / {res.get('channel')}", "",
         f"eligible {res['n_eligible']}  chosen {res['n_chosen']}  "
         f"(`{res['which']}`)", "",
         "### Compression ladder", "",
         "Cosine with the real Delta over eligible coordinates. `params` is the "
         "size of the description; the cheapest row that stays high is the "
         "description length of the behavioural routing difference.", "",
         "| rung | compression | params | cosine | expl.var |",
         "|---|---|---|---|---|"]
    for k, v in res["topk"].items():
        L.append(f"| 1 edge-sparse | top-{k} | {v['params']} | {v['cosine']:.3f} | {v['explained_var']:.3f} |")
    for by in ("head", "slot"):
        gs = res["groups"].get(by, {}).get("group_sparse", {})
        for m, v in gs.items():
            L.append(f"| 2 group-sparse | top-{m} {by}s | {v['params']} | {v['cosine']:.3f} | {v['explained_var']:.3f} |")
    for r, v in res["lowrank"].items():
        if r.endswith("_null"):
            continue
        L.append(f"| 3 low-rank | rank-{r} | {v['params']} | {v['cosine']:.3f} | {v['explained_var']:.3f} |")
    for by, g in res["groups"].items():
        v = g["cell_average"]
        L.append(f"| 4 structured | mean per {by} | {v['params']} | {v['cosine']:.3f} | {v['explained_var']:.3f} |")
    v = res["uniform_signed"]
    L.append(f"| 5 unstructured | sign x mean\\|Delta\\| | 1 | {v['cosine']:.3f} | {v['explained_var']:.3f} |")

    n1 = res["lowrank"].get("1_null", {}).get("cosine")
    if n1 is not None:
        L += ["", f"Rank-1 on shuffled Delta reaches cosine {n1:.3f}; anything at or "
                  f"below that is what a structureless vector gives for free.", ""]

    if "behaviour_free" in res:
        real, bf = res["real_ladder"], res["behaviour_free"]
        L += ["### Against a behaviour-free Delta", "",
              f"The same ladder on {bf['n_draws']} balanced-split draws: identical "
              "items, paraphrases and averaging depth, with the pos-vs-neg term "
              "cancelled. Only an excess over these is about the behaviour.", "",
              "| compression | real | null mean | null p95 | excess |",
              "|---|---|---|---|---|"]
        for k in real:
            r, m, p = real[k], bf["mean"][k], bf["p95"][k]
            flag = "**+%.3f**" % (r - p) if r > p else "%.3f" % (r - p)
            L.append(f"| {k} | {r:.3f} | {m:.3f} | {p:.3f} | {flag} |")
        wins = [k for k in real if real[k] > bf["p95"][k]]
        L += ["", ("Beats the behaviour-free null at: " + ", ".join(wins)) if wins else
              "**No compression beats the behaviour-free null.** Whatever structure "
              "the ladder above reports is a property of routing differences in "
              "general, not of this behaviour.", ""]

    L += ["### Concentration vs a shuffled Delta", "",
          "| grouping | groups | top-8 mass | null top-8 (p95) | groups for 50% | null |",
          "|---|---|---|---|---|---|"]
    for by, g in res["groups"].items():
        c, n = g["concentration"], g["null"]
        if not c["top"]:
            continue
        L.append(f"| {by} | {c['n_groups']} | {c['top'].get('8', 0):.3f} | "
                 f"{n['top_mean'].get('8', 0):.3f} ({n['top_p95'].get('8', 0):.3f}) | "
                 f"{c['n_for_50pct']} | {n['n_for_50pct_mean']:.0f} |")

    cc, cn = res["connectivity"]["chosen"], res["connectivity"]["null"]
    if cc["n"]:
        L += ["", "### Connectivity of the selected set", "",
              f"- {cc['n']} edges over {cc['n_nodes']} (layer, head) nodes",
              f"- with a selected predecessor: {cc['frac_with_pred']:.2f} "
              f"(random subset of the same size: {cn.get('frac_with_pred', 0):.2f})",
              f"- with a selected successor: {cc['frac_with_succ']:.2f} "
              f"(random: {cn.get('frac_with_succ', 0):.2f})",
              f"- longest chain: {cc['longest_path']} (random: {cn.get('longest_path', 0):.1f})",
              "", "Connectivity at or near the random level means the set is a "
              "collection of independent nudges, not a path."]
    return L


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stats", nargs="+", required=True,
                   help="*_stats.npz files from discover_circuit.py (globs ok)")
    p.add_argument("--which", default="delta_all",
                   choices=["delta_all", "delta_train", "delta_test"],
                   help="which Delta to decompose (default: all items)")
    p.add_argument("--out", default=None, help="write JSON + markdown here")
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--topk", type=int, nargs="+", default=[16, 64, 256, 1024])
    p.add_argument("--group-tops", type=int, nargs="+", default=[2, 8, 32])
    p.add_argument("--groupings", nargs="+",
                   default=["head", "slot", "layer", "stream", "srcdist",
                            "stream_srcdist", "stream_layer", "stream_layer_srcdist"])
    p.add_argument("--contrast", nargs="+", default=None,
                   help="contrast_<behavior>.pt from extract_contrast.py; enables "
                        "the balanced-split (behaviour-free) null, which is the "
                        "only control that preserves non-behavioural structure")
    p.add_argument("--n-contrast-draws", type=int, default=20)
    p.add_argument("--null-ranks", type=int, nargs="+", default=[1, 2],
                   help="ranks to fit on each behaviour-free draw (kept short)")
    p.add_argument("--n-null", type=int, default=200)
    p.add_argument("--als-iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted({Path(f) for pat in args.stats for f in glob.glob(pat)})
    assert paths, f"no files matched {args.stats}"
    rng = np.random.default_rng(args.seed)
    results, lines = [], ["# Structure of the routing difference", ""]
    for path in paths:
        res = analyse(path, args, rng)
        results.append(res)
        lines += report(res) + [""]
    text = "\n".join(lines)
    print(text)
    if args.out:
        out = Path(args.out)
        save_json({"args": vars(args), "results": results},
                  out.with_suffix(".json"))
        out.with_suffix(".md").write_text(text)
        print(f"[saved] {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
