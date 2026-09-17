"""Step 2 — turn the routing-weight difference into a circuit.

Given the per-prompt routing vectors from step 1:

    signal   Delta[i] = mean_v alpha_pos[i, v] - mean_v alpha_neg[i, v]

for item i, averaged over instruction paraphrases v.  The proposal is that
Delta is sparse whenever the behaviour is carried by a local circuit, so the
job here is to decide *which* coordinates are really non-zero rather than
picking a threshold by eye.  Three things do that:

1. A sign-flip permutation test on the paired differences.  Under the null
   that the two instructions are interchangeable, each item's difference is
   sign-symmetric; flipping signs leaves the sum of squares fixed, so the
   permuted t statistic is exact and cheap.  Per-coordinate p-values go
   through Benjamini-Hochberg; the max-|t| null gives a family-wise
   alternative that assumes nothing about independence between edges.

2. A *balanced-split null*.  The pos and neg paraphrases are re-partitioned
   into two halves that each contain the same number of pos and neg prompts,
   and differenced.  The behavioural contrast cancels exactly while the
   averaging depth, the paraphrase diversity and the item count all match the
   real contrast, so running the identical selection rule on it estimates how
   many edges the rule would have picked from nothing at all.

3. A held-out item split.  The circuit is selected on half the items; the
   other half only checks it — sign agreement and correlation of Delta on the
   selected edges.  A circuit that does not replicate across items is a
   circuit for these particular questions.

The surviving coordinates are reassembled into a graph over
(source layer output) -> (stream, target head) and written out as JSON, a
Graphviz .dot, an .npz of the raw statistics and a markdown report.

Usage:
    python experiments/interp/discover_circuit.py --behavior honesty --channel pred
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from circuit_common import (RoutingLayout, bh_fdr, gini, paired_t, save_json,
                            signflip_null)

STREAM_COLOR = {"q": "#1f77b4", "k": "#ff7f0e", "v": "#2ca02c", "r": "#9467bd"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--behavior", default="honesty")
    ap.add_argument("--in-dir", default="experiments/results/interp/circuits")
    ap.add_argument("--out-dir", default="experiments/results/interp/circuits")
    ap.add_argument("--channel", default="pred", choices=("pred", "corr", "eff"),
                    help="which routing channel to build the circuit from")
    ap.add_argument("--select", default="null",
                    choices=("null", "fdr", "fwer", "topk", "abs"),
                    help="selection rule for circuit membership. 'null' "
                         "calibrates against the balanced-split null and is the "
                         "one to use when Delta has little item-level variance; "
                         "'fdr'/'fwer' test significance, which is a different "
                         "and often useless question here (see the README)")
    ap.add_argument("--q", type=float, default=0.005,
                    help="tail fraction of the empirical null to allow through "
                         "(--select null), or the FDR level / FWER alpha. For "
                         "the null rule this is the main knob on the "
                         "signal/null selection ratio: 0.05 gave 27x on the "
                         "300M checkpoint's usable behaviour, 0.005 gave 131x")
    ap.add_argument("--topk", type=int, default=64, help="k for --select topk")
    ap.add_argument("--abs-thresh", type=float, default=0.01,
                    help="|Delta| cutoff for --select abs")
    ap.add_argument("--min-abs", type=float, default=0.0,
                    help="additional |Delta| floor applied to every rule; use "
                         "it to drop statistically clear but tiny edges")
    ap.add_argument("--hyper-only", action="store_true",
                    help="consider only skip connections (src < target layer), "
                         "excluding the sequential read-the-layer-below path")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-null-draws", type=int, default=8)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ---------------------------------------------------------------------------

def group_alpha(alpha: torch.Tensor, meta: list[dict], polarity: str,
                n_items: int, n_var: int) -> np.ndarray:
    """[n_prompts, D] -> [n_items, n_variants, D] for one polarity."""
    D = alpha.shape[1]
    out = np.full((n_items, n_var, D), np.nan, dtype=np.float64)
    for row, m in enumerate(meta):
        if m["polarity"] == polarity:
            out[m["item"], m["variant"]] = alpha[row].numpy()
    assert not np.isnan(out).any(), f"missing prompts for polarity {polarity}"
    return out


def balanced_null(pos: np.ndarray, neg: np.ndarray, n_draws: int,
                  rng: np.random.Generator) -> list[np.ndarray]:
    """Contrasts with the behavioural signal cancelled but the noise intact.

    Each draw splits the pos paraphrases in half and the neg paraphrases in
    half, then differences (pos_A + neg_A)/n vs (pos_B + neg_B)/n. Both sides
    hold the same polarity mix, so any real pos-vs-neg effect cancels, while
    the number of prompts averaged on each side equals that of the real
    contrast.
    """
    vp, vn = pos.shape[1], neg.shape[1]
    assert vp >= 2 and vn >= 2, "need >=2 paraphrases per polarity for the null"
    hp, hn = vp // 2, vn // 2
    draws = []
    for _ in range(n_draws):
        ip = rng.permutation(vp)
        in_ = rng.permutation(vn)
        a = np.concatenate([pos[:, ip[:hp]], neg[:, in_[:hn]]], axis=1).mean(1)
        b = np.concatenate([pos[:, ip[hp:2 * hp]], neg[:, in_[hn:2 * hn]]], axis=1).mean(1)
        draws.append(a - b)
    return draws


def effect_sizes(pos: np.ndarray, neg: np.ndarray, eligible: np.ndarray) -> dict:
    """How much does the instruction move alpha, next to everything else?

    The whole circuit story needs the instruction to change the routing
    weights. This reports by how much, in units of alpha itself, alongside the
    two other things the same weights vary with: the content item, and the
    choice of paraphrase within one instruction pool. An instruction effect far
    below the content effect means the predictor reads the instruction without
    re-routing on it -- no selection rule can rescue that, and it caps what any
    intervention on these coordinates can do.

    Args:
        pos, neg: [n_items, n_variants, D] routing weights per polarity.
        eligible: [D] coordinates to average over.
    """
    both = np.concatenate([pos, neg], axis=1)
    scale = np.abs(both).mean((0, 1))                            # mean |alpha|
    instruction = np.abs(pos.mean(1) - neg.mean(1)).mean(0)      # |Delta|
    content = both.mean(1).std(0)                                # across items
    paraphrase = np.concatenate([pos.std(1), neg.std(1)], 0).mean(0)

    def m(v: np.ndarray) -> float:
        return float(v[eligible].mean())

    s = max(m(scale), 1e-12)
    return {
        "mean_abs_alpha": m(scale),
        "instruction_effect": m(instruction),
        "content_effect": m(content),
        "paraphrase_effect": m(paraphrase),
        "instruction_over_alpha": m(instruction) / s,
        "content_over_alpha": m(content) / s,
        "paraphrase_over_alpha": m(paraphrase) / s,
        "instruction_over_content": m(instruction) / max(m(content), 1e-12),
        "instruction_over_paraphrase": m(instruction) / max(m(paraphrase), 1e-12),
    }


def null_context(est: list[np.ndarray], cal: list[np.ndarray], tr: np.ndarray,
                 eligible: np.ndarray, q: float) -> dict:
    """Calibrate a threshold against the balanced-split null.

    Significance testing asks "is this edge's difference non-zero?".  When the
    predictor's Delta barely varies across items — which is what actually
    happens, because the routing weights are close to a deterministic function
    of the instruction — that question has the answer "yes" for nearly every
    edge, and an FDR rule selects nearly everything.  The useful question is
    the other one: *is this edge's difference bigger than the difference the
    same averaging produces when there is no behavioural contrast at all?*

    So: estimate a per-coordinate null scale from one set of balanced-split
    draws, form S = |Delta| / sigma, and take the threshold from the (1-q)
    quantile of S on a disjoint set of draws.  Selecting S(Delta_real) >= that
    threshold admits a q-fraction of pure-noise coordinates by construction.
    """
    est_m = np.stack([d[tr].mean(0) for d in est])          # [n_est, D]
    sigma = est_m.std(0, ddof=1)
    # Shrink toward the bulk: a coordinate whose few null draws happened to
    # agree would otherwise get a near-zero sigma and dominate the ranking.
    floor = 0.25 * np.median(sigma[eligible])
    sigma = np.maximum(sigma, max(floor, 1e-12))
    cal_s = np.concatenate([np.abs(d[tr].mean(0))[eligible] / sigma[eligible]
                            for d in cal])
    return {"sigma": sigma, "thr": float(np.quantile(cal_s, 1.0 - q)),
            "n_est": len(est), "n_cal": len(cal)}


def select(delta: np.ndarray, t: np.ndarray, p: np.ndarray, maxnull: np.ndarray,
           args, eligible: np.ndarray, ctx: dict | None = None
           ) -> tuple[np.ndarray, dict]:
    """Apply the chosen selection rule inside the eligible coordinate set."""
    D = delta.shape[0]
    chosen = np.zeros(D, dtype=bool)
    info: dict = {"rule": args.select}
    if args.select == "null":
        assert ctx, "--select null needs a calibrated null context"
        s = np.abs(delta) / ctx["sigma"]
        chosen = eligible & (s >= ctx["thr"])
        info.update(q=args.q, s_cutoff=ctx["thr"],
                    expected_false_positives=args.q * float(eligible.sum()),
                    n_null_draws_est=ctx["n_est"], n_null_draws_cal=ctx["n_cal"])
    elif args.select == "fdr":
        sub, cut = bh_fdr(p[eligible], args.q)
        chosen[np.nonzero(eligible)[0][sub]] = True
        info.update(q=args.q, p_cutoff=cut)
    elif args.select == "fwer":
        thr = float(np.quantile(maxnull, 1 - args.q))
        chosen = eligible & (np.abs(t) >= thr)
        info.update(alpha=args.q, t_cutoff=thr)
    elif args.select == "topk":
        idx = np.nonzero(eligible)[0]
        order = idx[np.argsort(-np.abs(t[idx]))][:args.topk]
        chosen[order] = True
        info.update(k=args.topk)
    elif args.select == "abs":
        chosen = eligible & (np.abs(delta) >= args.abs_thresh)
        info.update(abs_thresh=args.abs_thresh)
    if args.min_abs > 0:
        chosen &= np.abs(delta) >= args.min_abs
        info["min_abs"] = args.min_abs
    return chosen, info


def build_graph(layout: RoutingLayout, chosen: np.ndarray, delta: np.ndarray,
                t: np.ndarray) -> dict:
    """Reassemble selected coordinates into the routing graph.

    Sources are *layer outputs* (the embedding for src 0), targets are
    individual heads, so the graph is bipartite-by-construction between
    "what a layer produced" and "which head reads it on which stream".
    """
    idx = np.nonzero(chosen)[0]
    edges = []
    for i in idx:
        e = layout.edges[i]
        edges.append({**e.to_json(), "flat_index": int(i),
                      "delta": float(delta[i]), "t": float(t[i]),
                      "source_node": e.source_name, "target_node": e.target_name})
    edges.sort(key=lambda r: -abs(r["delta"]))

    src_nodes = sorted({e["source_node"] for e in edges})
    tgt_nodes = sorted({e["target_node"] for e in edges})
    in_deg = Counter(e["target_node"] for e in edges)
    out_deg = Counter(e["source_node"] for e in edges)
    by_stream = Counter(e["stream"] for e in edges)
    by_layer = Counter(e["layer"] for e in edges)
    by_src = Counter(e["src"] for e in edges)

    # Connected components over the undirected version.
    adj: dict[str, set[str]] = defaultdict(set)
    for e in edges:
        adj[e["source_node"]].add(e["target_node"])
        adj[e["target_node"]].add(e["source_node"])
    seen: set[str] = set()
    components = []
    for n in list(adj):
        if n in seen:
            continue
        stack, comp = [n], []
        seen.add(n)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        components.append(sorted(comp))
    components.sort(key=len, reverse=True)

    return {
        "n_edges": len(edges),
        "n_hyper_edges": sum(1 for e in edges if e["is_hyper"]),
        "edges": edges,
        "source_nodes": src_nodes, "target_nodes": tgt_nodes,
        "n_distinct_heads": len(tgt_nodes),
        "n_distinct_layers": len({e["layer"] for e in edges}),
        "in_degree": dict(in_deg.most_common()),
        "out_degree": dict(out_deg.most_common()),
        "by_stream": dict(by_stream), "by_layer": dict(sorted(by_layer.items())),
        "by_src": dict(sorted(by_src.items())),
        "components": components,
        "n_components": len(components),
        "largest_component": len(components[0]) if components else 0,
    }


def write_dot(graph: dict, path: Path, behavior: str) -> None:
    lines = [f'digraph "{behavior}" {{', "  rankdir=LR;",
             '  node [fontname="Helvetica", fontsize=10];',
             '  edge [fontname="Helvetica", fontsize=8];']
    for n in graph["source_nodes"]:
        lines.append(f'  "{n}" [shape=box, style=filled, fillcolor="#eeeeee"];')
    for n in graph["target_nodes"]:
        lines.append(f'  "{n}" [shape=ellipse];')
    mx = max((abs(e["delta"]) for e in graph["edges"]), default=1.0) or 1.0
    for e in graph["edges"]:
        w = 0.5 + 3.5 * abs(e["delta"]) / mx
        style = "solid" if e["delta"] > 0 else "dashed"
        lines.append(
            f'  "{e["source_node"]}" -> "{e["target_node"]}" '
            f'[color="{STREAM_COLOR.get(e["stream"], "#666666")}", '
            f'penwidth={w:.2f}, style={style}, label="{e["stream"]}"];')
    lines.append("}")
    path.write_text("\n".join(lines))
    print(f"[saved] {path}")


def fmt_table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    in_path = Path(args.in_dir) / f"contrast_{args.behavior}.pt"
    payload = torch.load(in_path, map_location="cpu", weights_only=False)
    print(f"[discover] loaded {in_path}")

    L, H, D = (payload["layout"][k] for k in ("L", "H", "D"))
    layout = RoutingLayout(L, H)
    assert layout.D == D, f"layout mismatch {layout.D} != {D}"
    if args.channel not in payload["alpha"]:
        raise SystemExit(f"channel {args.channel!r} not in this contrast file "
                         f"(have {sorted(payload['alpha'])}); re-run "
                         f"extract_contrast.py with a wider --channel")
    alpha = payload["alpha"][args.channel]
    meta, n_items = payload["meta"], payload["n_items"]
    pos = group_alpha(alpha, meta, "pos", n_items, payload["variants"]["pos"])
    neg = group_alpha(alpha, meta, "neg", n_items, payload["variants"]["neg"])

    X = pos.mean(1) - neg.mean(1)                      # [n_items, D]
    rng = np.random.default_rng(args.seed)
    # Three disjoint sets of balanced-split draws: one to estimate the null
    # scale, one to calibrate the threshold, one held back to report how many
    # edges the finished rule picks from a contrast with no signal in it. The
    # third set has to be disjoint or that number is circular.
    nd = args.n_null_draws
    draws = balanced_null(pos, neg, 3 * nd, rng)
    est, cal, chk = draws[:nd], draws[nd:2 * nd], draws[2 * nd:]

    # --- held-out item split --------------------------------------------
    perm = np.random.default_rng(args.split_seed).permutation(n_items)
    tr, te = perm[: n_items // 2], perm[n_items // 2:]
    print(f"[discover] channel={args.channel} items={n_items} "
          f"(discover on {len(tr)}, hold out {len(te)})")

    delta_tr, t_tr = paired_t(X[tr])
    delta_te, t_te = paired_t(X[te])
    delta_all, t_all = paired_t(X)

    eligible = layout.hyper_arr.copy() if args.hyper_only else np.ones(D, dtype=bool)

    eff = effect_sizes(pos, neg, eligible)
    print(f"[discover] alpha responds to: instruction "
          f"{eff['instruction_over_alpha']:.4f}, content "
          f"{eff['content_over_alpha']:.4f}, paraphrase "
          f"{eff['paraphrase_over_alpha']:.4f} (as fractions of mean |alpha|) "
          f"-> instruction/content = {eff['instruction_over_content']:.3f}")
    if eff["instruction_over_content"] < 0.1:
        print("[discover] NOTE: the instruction moves the routing weights by "
              "less than a tenth of what the content moves them. Whatever is "
              "selected below is a small perturbation on a mostly "
              "content-driven routing map.")

    needs_perm = args.select in ("fdr", "fwer")
    if needs_perm:
        p_tr, maxnull_tr = signflip_null(X[tr], args.n_perm, seed=args.seed)
    else:
        p_tr = np.ones(D)
        maxnull_tr = np.zeros(1)

    ctx = null_context(est, cal, tr, eligible, args.q) if args.select == "null" else None
    chosen, sel_info = select(delta_tr, t_tr, p_tr, maxnull_tr, args, eligible, ctx)
    n_sel = int(chosen.sum())
    print(f"[discover] rule={sel_info} -> {n_sel} / {int(eligible.sum())} edges "
          f"({100 * n_sel / max(1, eligible.sum()):.2f}%)")

    # --- how many would the same rule pick from a null contrast? ---------
    null_counts = []
    for j, Xn in enumerate(chk):
        d_n, t_n = paired_t(Xn[tr])
        if needs_perm:
            p_n, mx_n = signflip_null(Xn[tr], max(1000, args.n_perm // 5),
                                      seed=args.seed + 100 + j)
        else:
            p_n, mx_n = np.ones(D), np.zeros(1)
        c_n, _ = select(d_n, t_n, p_n, mx_n, args, eligible, ctx)
        null_counts.append(int(c_n.sum()))
    null_mean = float(np.mean(null_counts)) if null_counts else 0.0
    ratio = n_sel / max(null_mean, 1e-9)
    print(f"[discover] balanced-split null selects {null_counts} "
          f"(mean {null_mean:.1f}) under the same rule -> "
          f"signal/noise selection ratio {ratio:.2f}x")
    if n_sel and null_mean >= 0.7 * n_sel:
        print("[discover] WARNING: the rule picks almost as many edges from a "
              "contrast with no behavioural signal as from the real one. This "
              "circuit is not distinguishable from paraphrase noise — tighten "
              "--q, or read the sparsity numbers as a negative result.")

    # --- does the circuit replicate on held-out items? -------------------
    if n_sel > 0:
        sign_agree = float((np.sign(delta_tr[chosen]) == np.sign(delta_te[chosen])).mean())
        cc = np.corrcoef(delta_tr[chosen], delta_te[chosen])[0, 1] if n_sel > 1 else float("nan")
        rand_idx = rng.choice(np.nonzero(eligible)[0], size=n_sel, replace=False)
        sign_agree_rand = float(
            (np.sign(delta_tr[rand_idx]) == np.sign(delta_te[rand_idx])).mean())
    else:
        sign_agree = cc = sign_agree_rand = float("nan")
    print(f"[discover] held-out replication: sign agreement {sign_agree:.3f} "
          f"(random edges {sign_agree_rand:.3f}), corr(Delta_tr, Delta_te) = {cc:.3f}")

    # --- sparsity of the difference matrix -------------------------------
    absd = np.abs(delta_all)
    order = np.argsort(-absd)
    mass = absd[order].cumsum() / max(absd.sum(), 1e-12)
    sparsity = {
        "gini_abs_delta": gini(delta_all),
        "top_1pct_mass_share": float(mass[max(0, D // 100 - 1)]),
        "top_64_mass_share": float(mass[min(63, D - 1)]),
        "n_edges_for_50pct_mass": int(np.searchsorted(mass, 0.5) + 1),
        "n_edges_for_90pct_mass": int(np.searchsorted(mass, 0.9) + 1),
        "mean_abs_delta": float(absd.mean()),
        "max_abs_delta": float(absd.max()),
        "selected": n_sel,
        "selected_frac": float(n_sel / D),
        "null_selected_mean": null_mean,
        "null_selected_draws": null_counts,
        "signal_over_null_selection_ratio": ratio,
    }
    print(f"[discover] sparsity: {sparsity['n_edges_for_50pct_mass']} edges carry "
          f"50% of |Delta| mass, gini={sparsity['gini_abs_delta']:.3f}")

    graph = build_graph(layout, chosen, delta_tr, t_tr)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"circuit_{args.behavior}_{args.channel}"

    np.savez(out_dir / f"{stem}_stats.npz", delta_train=delta_tr, delta_test=delta_te,
             delta_all=delta_all, t_train=t_tr, t_test=t_te, p_train=p_tr,
             chosen=chosen, eligible=eligible, maxnull=maxnull_tr,
             train_items=tr, test_items=te)
    print(f"[saved] {out_dir / f'{stem}_stats.npz'}")

    result = {
        "behavior": args.behavior, "channel": args.channel,
        "layout": {"L": L, "H": H, "D": D},
        "selection": sel_info,
        "hyper_only": args.hyper_only,
        "n_items": n_items, "n_train_items": len(tr), "n_test_items": len(te),
        "effect_sizes": eff,
        "sparsity": sparsity,
        "replication": {"sign_agreement": sign_agree,
                        "sign_agreement_random_edges": sign_agree_rand,
                        "corr_train_test_delta": float(cc)},
        "graph": graph,
        "behavior_scores": payload.get("behavior_scores", {}),
        "source": str(in_path),
    }
    save_json(result, out_dir / f"{stem}.json")
    write_dot(graph, out_dir / f"{stem}.dot", args.behavior)

    # --- markdown report --------------------------------------------------
    gap = payload.get("behavior_scores", {}).get("gap", {})
    top = graph["edges"][:25]
    md = [
        f"# Circuit for `{args.behavior}` — channel `{args.channel}`",
        "",
        f"Source: `{in_path}`  ",
        f"Selection: `{json.dumps(sel_info)}`, "
        f"{'hyperconnections only' if args.hyper_only else 'all edges'}",
        "",
        "## Is there a behaviour to explain?",
        "",
        (f"Instruction gap (pos - neg) = **{gap.get('pos_minus_neg', float('nan')):+.4f}** "
         f"(paired t = {gap.get('paired_t', float('nan')):+.2f} over "
         f"{gap.get('n_items', 0)} items) — {gap.get('verdict', 'not measured')}"
         if gap else "Not measured (extract was run with --no-behavior-score)."),
        "",
        "## Does the instruction move the routing weights at all?",
        "",
        fmt_table([
            ["mean |alpha| over eligible edges", f"{eff['mean_abs_alpha']:.4f}"],
            ["instruction effect (mean |Delta|)",
             f"{eff['instruction_effect']:.5f} "
             f"({100 * eff['instruction_over_alpha']:.2f}% of |alpha|)"],
            ["content effect (sd across items)",
             f"{eff['content_effect']:.5f} "
             f"({100 * eff['content_over_alpha']:.2f}% of |alpha|)"],
            ["paraphrase effect (sd within a pool)",
             f"{eff['paraphrase_effect']:.5f} "
             f"({100 * eff['paraphrase_over_alpha']:.2f}% of |alpha|)"],
            ["instruction / content", f"{eff['instruction_over_content']:.3f}"],
            ["instruction / paraphrase", f"{eff['instruction_over_paraphrase']:.3f}"],
        ], ["quantity", "value"]),
        "",
        "These ratios describe the size of the observed routing response. "
        "They do not bound the output effect: a small routing change could "
        "still matter on sensitive coordinates. Causal verification tests "
        "whether the selected differences change the scored behavior.",
        "",
        "## Sparsity of the difference matrix",
        "",
        fmt_table([
            ["edges total", D],
            ["edges selected", f"{n_sel} ({100 * n_sel / D:.2f}%)"],
            ["selected by the matched null (mean)", f"{null_mean:.1f}"],
            ["signal / null selection ratio", f"{ratio:.2f}x"],
            ["edges holding 50% of |Delta| mass", sparsity["n_edges_for_50pct_mass"]],
            ["edges holding 90% of |Delta| mass", sparsity["n_edges_for_90pct_mass"]],
            ["Gini of |Delta|", f"{sparsity['gini_abs_delta']:.3f}"],
            ["max |Delta|", f"{sparsity['max_abs_delta']:.4f}"],
            ["mean |Delta|", f"{sparsity['mean_abs_delta']:.5f}"],
        ], ["quantity", "value"]),
        "",
        "## Does it replicate on held-out items?",
        "",
        fmt_table([
            ["sign agreement on selected edges", f"{sign_agree:.3f}"],
            ["sign agreement on random edges", f"{sign_agree_rand:.3f}"],
            ["corr(Delta_train, Delta_test) on selected", f"{cc:.3f}"],
        ], ["quantity", "value"]),
        "",
        "## Circuit shape",
        "",
        fmt_table([
            ["edges", graph["n_edges"]],
            ["of which hyperconnections (src < layer)", graph["n_hyper_edges"]],
            ["distinct target heads", graph["n_distinct_heads"]],
            ["distinct target layers", graph["n_distinct_layers"]],
            ["connected components", graph["n_components"]],
            ["largest component (nodes)", graph["largest_component"]],
            ["by stream", json.dumps(graph["by_stream"])],
            ["by target layer", json.dumps(graph["by_layer"])],
            ["by source", json.dumps(graph["by_src"])],
        ], ["quantity", "value"]),
        "",
        "## Top edges",
        "",
        fmt_table([[e["name"], f"{e['delta']:+.4f}", f"{e['t']:+.2f}",
                    "hyper" if e["is_hyper"] else "sequential"] for e in top],
                  ["edge", "Delta", "t", "kind"]),
        "",
        "Positive Delta means the connection is weighted *more* under the "
        f"`pos` instruction than under `neg`.",
        "",
        "## Reading the selection",
        "",
        (f"The rule picked {n_sel} edges from the real contrast and "
         f"{null_mean:.1f} on average from balanced-split nulls that contain no "
         f"behavioural signal — a ratio of **{ratio:.2f}x**. "
         + ("A ratio near 1 means the selection is not distinguishable from "
            "paraphrase noise, whatever the p-values say."
            if null_mean >= 0.7 * n_sel else
            "The circuit is selected well above what paraphrase noise produces.")),
        "",
        (f"Held-out sign agreement is {sign_agree:.3f} on selected edges against "
         f"{sign_agree_rand:.3f} on random ones. When those two are equal, Delta "
         f"is essentially the same for every item — the routing difference is a "
         f"function of the instruction alone, so per-item significance testing "
         f"has no power to separate edges and `--select null` is the rule to use."),
        "",
    ]
    (out_dir / f"{stem}.md").write_text("\n".join(md))
    print(f"[saved] {out_dir / f'{stem}.md'}")
    print("[discover] DONE")


if __name__ == "__main__":
    main()
