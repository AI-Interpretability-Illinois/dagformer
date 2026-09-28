"""Scaling-law analysis of DAGFormer (routed) vs dense OLMo-2 backbones.

Inputs (all optional, whatever exists is used):
  experiments/scaling/collected_<machine>.json   from scripts/scaling_collect.py
  experiments/scaling/common_eval_<machine>.json from scripts/scaling_common_eval.py

Outputs in experiments/scaling/:
  runs_table.md            one row per run (N, D, C, own-curve final, common-eval NLLs)
  scaling_fits.json        fitted parameters and derived quantities
  results.md               auto-generated tables (fits, gaps, effective-parameter multipliers)
  fig_loss_vs_params.png   common-eval NLL vs backbone params, per family, near-Chinchilla runs, with L(N)=E+A/N^a fits
  fig_loss_vs_tokens.png   training-time eval NLL vs tokens seen, dense vs routed at each (corpus, size)
  fig_loss_vs_compute.png  NLL vs training FLOPs (finals + curves): the iso-compute view
  fig_gap_vs_scale.png     dense - routed NLL vs N (finals) and vs tokens (curves)

Run: python scripts/scaling_fit.py [--eval-key dolma21b] [--chinchilla-band 12 30]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np

FAMILY_STYLE = {  # fixed colours (never cycled); routed variants share a hue family
    "dense": ("#444444", "o", "dense OLMo-2"),
    "corrected": ("#1f77b4", "s", "fourway_corrected (DAGFormer)"),
    "modular": ("#ff7f0e", "^", "fourway_modular"),
    "global": ("#2ca02c", "v", "fourway (predictor only)"),
    "local": ("#9467bd", "D", "corrections only"),
    "per_layer": ("#8c564b", "P", "per-layer predictors"),
    "modular_sparse": ("#e377c2", "X", "fourway_modular + sparsity"),
}
ROUTED_MAIN = ("corrected", "modular")


def load_inputs(root: str):
    runs = {}
    for f in sorted(glob.glob(os.path.join(root, "collected_*.json"))):
        for r in json.load(open(f))["runs"]:
            runs[r["name"]] = r
    common = {}
    for f in sorted(glob.glob(os.path.join(root, "common_eval_*.json"))):
        common.update(json.load(open(f)))
    for n, r in runs.items():
        r["common"] = common.get(n, {})
    return runs


# ---------------- fits ----------------
def fit_power(x, y, floor_init=1.5):
    """L = E + A * x^-a, least squares in linear space, bounded. Returns dict or None."""
    from scipy.optimize import curve_fit
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    f = lambda x, E, A, a: E + A * x ** (-a)  # noqa: E731
    best = None
    for a0 in (0.1, 0.3, 0.6):
        for E0 in (0.0, min(y) * 0.5, min(y) * 0.9):
            try:
                A0 = (y[0] - E0) * x[0] ** a0
                popt, _ = curve_fit(f, x, y, p0=[E0, max(A0, 1e-3), a0],
                                    bounds=([0, 1e-9, 0.01], [min(y), np.inf, 2.0]), maxfev=20000)
                res = float(np.sum((f(x, *popt) - y) ** 2))
                if best is None or res < best[1]:
                    best = (popt, res)
            except Exception:
                continue
    if best is None:
        return None
    E, A, a = best[0]
    return {"E": float(E), "A": float(A), "alpha": float(a), "rss": best[1], "n_points": int(len(x))}


def fit_joint(N, D, L):
    """Chinchilla form L = E + A/N^a + B/D^b on (N, D, L) triples (needs >= 2 distinct N and D)."""
    from scipy.optimize import least_squares
    N, D, L = map(lambda v: np.asarray(v, float), (N, D, L))
    if len(set(N)) < 2 or len(set(D)) < 3 or len(L) < 6:
        return None
    lnN, lnD = np.log(N), np.log(D)

    def pred(p):
        e, a_, alpha, b_, beta = p
        return np.exp(e) + np.exp(a_ - alpha * lnN) + np.exp(b_ - beta * lnD)

    best = None
    for alpha0 in (0.2, 0.4):
        for beta0 in (0.2, 0.4):
            p0 = [math.log(max(min(L) * 0.7, 1e-3)), math.log(0.5) + alpha0 * math.log(np.median(N)),
                  alpha0, math.log(0.5) + beta0 * math.log(np.median(D)), beta0]
            try:
                r = least_squares(lambda p: pred(p) - L, p0, loss="huber", f_scale=0.02,
                                  bounds=([-10, -50, 0.01, -50, 0.01], [math.log(min(L)), 50, 2, 50, 2]))
                if best is None or r.cost < best.cost:
                    best = r
            except Exception:
                continue
    if best is None:
        return None
    e, a_, alpha, b_, beta = best.x
    return {"E": float(np.exp(e)), "A": float(np.exp(a_)), "alpha": float(alpha), "B": float(np.exp(b_)),
            "beta": float(beta), "rss": float(2 * best.cost), "n_points": int(len(L))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="experiments/scaling")
    p.add_argument("--eval-key", default="dolma21b", help="common-eval cache used for the N-scaling fits")
    p.add_argument("--chinchilla-band", nargs=2, type=float, default=[12, 30], help="tokens/param range counted as 'Chinchilla-scale'")
    p.add_argument("--params", default="total", choices=["total", "non_embed"], help="which backbone count is N")
    args = p.parse_args()
    root = args.root
    runs = load_inputs(root)
    if not runs:
        raise SystemExit("no collected_*.json found")
    ek = args.eval_key
    Nof = lambda r: r["params"][args.params]  # noqa: E731
    lo, hi = args.chinchilla_band

    # ---------- table of runs ----------
    rows = ["| run | machine | corpus | family | size | N backbone (M) | non-embed (M) | routing (M) | D tokens (B) | tok/param | FLOPs/token (G) | train C (PF) | done | own-curve final | "
            + " | ".join(f"common {k}" for k in ("dolma21b", "wikitext2", "mathinstruct", "gsm8k")) + " |",
            "|" + "---|" * 19]
    for n, r in sorted(runs.items(), key=lambda kv: (kv[1]["corpus"], Nof(kv[1]), kv[1]["family"])):
        c = r["common"]
        rows.append(f"| {n} | {r['machine']} | {r['corpus']} | {r['family']} | {r['size']} | {r['params']['total']/1e6:.1f} | {r['params']['non_embed']/1e6:.1f} | "
                    f"{(r['routing_params'] or 0)/1e6:.1f} | {r['total_tokens']/1e9:.2f} | {r['total_tokens']/r['params']['total']:.1f} | "
                    f"{r['flops']['train_flops_per_token']/1e9:.2f} | {r['train_flops_total']/1e15:.0f} | {'y' if r['done'] else 'n'} | "
                    f"{r['final_eval_nll'] if r['final_eval_nll'] is not None else '-'} | "
                    + " | ".join(f"{c[k]:.4f}" if k in c else "-" for k in ("dolma21b", "wikitext2", "mathinstruct", "gsm8k")) + " |")
    open(os.path.join(root, "runs_table.md"), "w").write("\n".join(rows) + "\n")

    fits: dict = {"eval_key": ek, "params": args.params, "chinchilla_band": [lo, hi], "L_of_N": {}, "joint": {},
                  "L_of_C": {}, "effective_params": {}, "gaps_final": [], "gaps_curve": {}}
    md = [f"# Scaling analysis (auto-generated by scripts/scaling_fit.py; N = backbone {args.params} params, "
          f"L = held-out NLL on the common `{ek}` cache unless stated)\n"]

    # ---------- L(N) at Chinchilla-scale budgets ----------
    chin = {}
    for n, r in runs.items():
        if r["done"] and ek in r["common"] and lo <= r["total_tokens"] / Nof(r) <= hi:
            chin.setdefault(r["family"], []).append((Nof(r), r["common"][ek], n, r["total_tokens"]))
    md.append(f"## L(N) at Chinchilla-scale budgets ({lo:g}-{hi:g} tokens/param), common `{ek}` NLL\n")
    md.append("| family | points (N, tok/param) | E | A | alpha | rss |\n|---|---|---|---|---|---|")
    for fam, pts in sorted(chin.items()):
        pts.sort()
        fit = fit_power([p[0] for p in pts], [p[1] for p in pts])
        fits["L_of_N"][fam] = {"points": [{"name": p[2], "N": p[0], "L": p[1], "D": p[3]} for p in pts], "fit": fit}
        desc = ", ".join(f"{p[0]/1e6:.0f}M@{p[3]/p[0]:.0f}" for p in pts)
        md.append(f"| {fam} | {desc} | " + (f"{fit['E']:.3f} | {fit['A']:.3g} | {fit['alpha']:.3f} | {fit['rss']:.2e}" if fit else "- | - | - | (need >= 3 points)") + " |")

    # effective-parameter multiplier of routed families vs the dense L(N) fit
    dfit = fits["L_of_N"].get("dense", {}).get("fit")
    if dfit:
        md.append("\n### Effective parameters: dense size that matches each routed run's loss (from the dense L(N) fit)\n")
        md.append("| run | family | N (M) | L | N_eff dense (M) | N_eff / N | tok/param |\n|---|---|---|---|---|---|---|")
        for fam, d in fits["L_of_N"].items():
            if fam == "dense":
                continue
            for pt in d["points"]:
                if pt["L"] > dfit["E"]:
                    neff = ((pt["L"] - dfit["E"]) / dfit["A"]) ** (-1.0 / dfit["alpha"])
                    fits["effective_params"][pt["name"]] = {"N": pt["N"], "N_eff": neff, "ratio": neff / pt["N"]}
                    md.append(f"| {pt['name']} | {fam} | {pt['N']/1e6:.0f} | {pt['L']:.4f} | {neff/1e6:.0f} | {neff/pt['N']:.2f} | {pt['D']/pt['N']:.0f} |")

    # ---------- final gaps dense - routed at matched (corpus, size) ----------
    md.append("\n## Dense - routed at matched corpus, size and tokens (common-eval NLL of the final checkpoints)\n")
    md.append("| corpus | size | N (M) | D (B) | dense | routed family | routed | gap (nats) | routed FLOPs / dense FLOPs |\n|---|---|---|---|---|---|---|---|---|")
    by_cs = {}
    for n, r in runs.items():
        if r["done"] and ek in r["common"]:
            by_cs.setdefault((r["corpus"], r["size"]), {})[r["family"]] = (n, r)
    for (corpus, size), fams in sorted(by_cs.items(), key=lambda kv: Nof(list(kv[1].values())[0][1])):
        if "dense" not in fams:
            continue
        dn, dr = fams["dense"]
        for fam, (rn, rr) in sorted(fams.items()):
            if fam == "dense" or abs(rr["total_tokens"] - dr["total_tokens"]) / dr["total_tokens"] > 0.05:
                continue
            gap = dr["common"][ek] - rr["common"][ek]
            ratio = rr["flops"]["train_flops_per_token"] / dr["flops"]["train_flops_per_token"]
            fits["gaps_final"].append({"corpus": corpus, "size": size, "N": Nof(dr), "D": dr["total_tokens"], "family": fam,
                                       "dense": dr["common"][ek], "routed": rr["common"][ek], "gap": gap, "flops_ratio": ratio})
            md.append(f"| {corpus} | {size} | {Nof(dr)/1e6:.0f} | {dr['total_tokens']/1e9:.2f} | {dr['common'][ek]:.4f} | {fam} | {rr['common'][ek]:.4f} | {gap:+.4f} | {ratio:.2f} |")

    # ---------- gap along training (own-corpus curves, same eval cache within a corpus) ----------
    md.append("\n## Dense - routed along training (training-time eval on the run's own corpus cache; same cache within a corpus)\n")
    by_cs_all = {}
    for n, r in runs.items():
        if r["curve"]:
            by_cs_all.setdefault((r["corpus"], r["size"]), {})[r["family"]] = r
    for (corpus, size), fams in sorted(by_cs_all.items(), key=lambda kv: Nof(list(kv[1].values())[0])):
        if "dense" not in fams:
            continue
        dcurve = {c["step"]: c for c in fams["dense"]["curve"]}
        for fam, rr in sorted(fams.items()):
            if fam == "dense":
                continue
            pts = [(c["tokens"], dcurve[c["step"]]["eval_nll"] - c["eval_nll"]) for c in rr["curve"] if c["step"] in dcurve]
            if pts:
                fits["gaps_curve"][f"{corpus}/{size}/{fam}"] = pts
                quart = [pts[max(0, int(len(pts) * q) - 1)] for q in (0.25, 0.5, 0.75, 1.0)]
                md.append(f"- {corpus} {size} dense - {fam}: " + ", ".join(f"{g:+.3f} @ {t/1e9:.2f}B" for t, g in quart))

    # ---------- joint L(N, D) per (family, corpus) from training curves ----------
    md.append("\n## Joint L(N, D) = E + A/N^alpha + B/D^beta per family and corpus (training-curve points, second half of each run)\n")
    md.append("Caveat: every size has ONE run with a cosine schedule, so intermediate points sit above what a run ending "
              "there would reach; beta is an upper bound on the true data exponent and alpha is what these few sizes support.\n")
    md.append("| family | corpus | sizes | points | E | A | alpha | B | beta | rss |\n|---|---|---|---|---|---|---|---|---|---|")
    groups = {}
    for n, r in runs.items():
        if r["curve"]:
            groups.setdefault((r["family"], r["corpus"]), []).append(r)
    for (fam, corpus), rs in sorted(groups.items()):
        N, D, L, sizes = [], [], [], set()
        for r in rs:
            sizes.add(r["size"])
            for c in r["curve"]:
                if c["tokens"] >= 0.5 * r["total_tokens"] and c["tokens"] > 0:
                    N.append(Nof(r)); D.append(c["tokens"]); L.append(c["eval_nll"])
        fit = fit_joint(N, D, L)
        fits["joint"][f"{fam}/{corpus}"] = {"sizes": sorted(sizes), "fit": fit}
        md.append(f"| {fam} | {corpus} | {', '.join(sorted(sizes))} | {len(L)} | "
                  + (f"{fit['E']:.3f} | {fit['A']:.3g} | {fit['alpha']:.3f} | {fit['B']:.3g} | {fit['beta']:.3f} | {fit['rss']:.2e}" if fit else "- | - | - | - | - | (need >= 2 sizes, >= 6 points)") + " |")

    # ---------- L(C): iso-compute ----------
    md.append("\n## L(C) with C = training FLOPs (6 x MACs/token x tokens, routing cost included), common-eval finals\n")
    md.append("| family | points | E | K | gamma |\n|---|---|---|---|---|")
    for fam in sorted({r["family"] for r in runs.values()}):
        pts = sorted((r["train_flops_total"], r["common"][ek]) for r in runs.values() if r["family"] == fam and r["done"] and ek in r["common"])
        fit = fit_power([p[0] for p in pts], [p[1] for p in pts])
        fits["L_of_C"][fam] = {"points": pts, "fit": fit}
        md.append(f"| {fam} | {len(pts)} | " + (f"{fit['E']:.3f} | {fit['A']:.3g} | {fit['alpha']:.3f}" if fit else "- | - | -") + " |")
    if fits["L_of_C"].get("dense", {}).get("fit") and fits["L_of_C"].get("corrected", {}).get("fit"):
        d, c = fits["L_of_C"]["dense"]["fit"], fits["L_of_C"]["corrected"]["fit"]
        md.append("\nCompute multiplier: for each routed final, the dense compute C_eq that reaches the same loss on the dense L(C) fit, "
                  "and C_eq / C_routed (>1 means the routed model is compute-efficient, <1 means the extra routing FLOPs are not repaid).\n")
        md.append("| routed run | C (PF) | L | C_eq dense (PF) | C_eq / C |\n|---|---|---|---|---|")
        for fam in ROUTED_MAIN:
            for n, r in sorted(runs.items(), key=lambda kv: kv[1]["train_flops_total"]):
                if r["family"] == fam and r["done"] and ek in r["common"] and r["common"][ek] > d["E"]:
                    ceq = ((r["common"][ek] - d["E"]) / d["A"]) ** (-1 / d["alpha"])
                    md.append(f"| {n} | {r['train_flops_total']/1e15:.0f} | {r['common'][ek]:.4f} | {ceq/1e15:.0f} | {ceq/r['train_flops_total']:.2f} |")

    json.dump(fits, open(os.path.join(root, "scaling_fits.json"), "w"), indent=1)
    open(os.path.join(root, "results.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    make_figures(runs, fits, root, ek, Nof, lo, hi)


# ---------------- figures ----------------
def make_figures(runs, fits, root, ek, Nof, lo, hi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    corpus_marker_alpha = {"delta21b": 1.0, "shared12b": 0.75, "timan12b": 0.55, "delta1p7b": 0.4}

    # 1. L vs N (Chinchilla-scale finals, common eval) + fits
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for fam, d in fits["L_of_N"].items():
        col, mk, label = FAMILY_STYLE.get(fam, ("#7f7f7f", "o", fam))
        xs = [p["N"] for p in d["points"]]; ys = [p["L"] for p in d["points"]]
        ax.scatter(xs, ys, color=col, marker=mk, s=42, label=label, zorder=3)
        for p in d["points"]:
            ax.annotate(runs[p["name"]]["corpus"], (p["N"], p["L"]), fontsize=6, xytext=(3, 3), textcoords="offset points", color=col)
        if d["fit"]:
            g = np.geomspace(min(xs) / 1.5, max(xs) * 2, 100)
            ax.plot(g, d["fit"]["E"] + d["fit"]["A"] * g ** (-d["fit"]["alpha"]), color=col, lw=1.5, alpha=0.8,
                    label=f"  fit: {d['fit']['E']:.2f} + {d['fit']['A']:.2g}·N^-{d['fit']['alpha']:.2f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("backbone parameters N"); ax.set_ylabel(f"held-out NLL (common {ek} cache)")
    ax.set_title(f"Loss vs model size at {lo:g}-{hi:g} tokens/param", fontsize=10)
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig_loss_vs_params.png"), dpi=160); plt.close(fig)

    # 2. L vs tokens: one panel per (corpus, size) with curves of every family
    groups = {}
    for n, r in runs.items():
        if r["curve"]:
            groups.setdefault((r["corpus"], r["size"]), []).append(r)
    keys = sorted(groups, key=lambda k: (k[0], Nof(groups[k][0])))
    if keys:
        ncol = min(4, len(keys)); nrow = math.ceil(len(keys) / ncol)
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False)
        for ax, key in zip(axes.flat, keys):
            for r in sorted(groups[key], key=lambda r: r["family"]):
                col, mk, label = FAMILY_STYLE.get(r["family"], ("#7f7f7f", "o", r["family"]))
                ax.plot([c["tokens"] for c in r["curve"]], [c["eval_nll"] for c in r["curve"]], color=col, marker=mk, ms=3, lw=1.2, label=label)
            ax.set_xscale("log"); ax.set_title(f"{key[0]} {key[1]} (N={Nof(groups[key][0])/1e6:.0f}M)", fontsize=8)
            ax.grid(True, which="both", alpha=0.25); ax.tick_params(labelsize=7); ax.legend(fontsize=5.5)
            ax.set_xlabel("tokens seen", fontsize=7); ax.set_ylabel("eval NLL (own corpus cache)", fontsize=7)
        for ax in list(axes.flat)[len(keys):]:
            ax.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(root, "fig_loss_vs_tokens.png"), dpi=160); plt.close(fig)

    # 3. L vs compute: finals (common eval) as points + own-curve trajectories faint
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for fam in sorted({r["family"] for r in runs.values()}):
        col, mk, label = FAMILY_STYLE.get(fam, ("#7f7f7f", "o", fam))
        fin = [(r["train_flops_total"], r["common"][ek]) for r in runs.values() if r["family"] == fam and r["done"] and ek in r["common"]]
        if fin:
            fin.sort(); ax.scatter([f[0] for f in fin], [f[1] for f in fin], color=col, marker=mk, s=40, label=label, zorder=3)
        fit = fits["L_of_C"].get(fam, {}).get("fit")
        if fit and fin:
            g = np.geomspace(min(f[0] for f in fin) / 2, max(f[0] for f in fin) * 3, 100)
            ax.plot(g, fit["E"] + fit["A"] * g ** (-fit["alpha"]), color=col, lw=1.2, alpha=0.7)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("training FLOPs (routing cost included)")
    ax.set_ylabel(f"held-out NLL (common {ek} cache)"); ax.set_title("Iso-compute view: final checkpoints", fontsize=10)
    ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig_loss_vs_compute.png"), dpi=160); plt.close(fig)

    # 4. gaps: dense - routed vs N (finals) and vs tokens (curves)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for fam in ROUTED_MAIN + ("global", "local", "per_layer", "modular_sparse"):
        col, mk, label = FAMILY_STYLE.get(fam, ("#7f7f7f", "o", fam))
        g = [(x["N"], x["gap"], x["corpus"]) for x in fits["gaps_final"] if x["family"] == fam]
        if g:
            g.sort(); a1.plot([v[0] for v in g], [v[1] for v in g], color=col, marker=mk, lw=1, label=label)
            for v in g:
                a1.annotate(v[2], (v[0], v[1]), fontsize=6, xytext=(3, 3), textcoords="offset points", color=col)
    a1.set_xscale("log"); a1.axhline(0, color="k", lw=0.6); a1.set_xlabel("backbone parameters N"); a1.set_ylabel("dense - routed NLL (final, common cache)")
    a1.set_title("Gap vs model size", fontsize=9); a1.grid(True, which="both", alpha=0.25); a1.legend(fontsize=6.5)
    for key, pts in sorted(fits["gaps_curve"].items()):
        corpus, size, fam = key.split("/")
        col, mk, _ = FAMILY_STYLE.get(fam, ("#7f7f7f", "o", fam))
        alpha = corpus_marker_alpha.get(corpus, 0.6)
        a2.plot([t for t, _ in pts], [g for _, g in pts], color=col, marker=mk, ms=3, lw=1, alpha=alpha, label=f"{size} {fam} ({corpus})")
    a2.set_xscale("log"); a2.axhline(0, color="k", lw=0.6); a2.set_xlabel("tokens seen"); a2.set_ylabel("dense - routed NLL (own-corpus cache)")
    a2.set_title("Gap along training", fontsize=9); a2.grid(True, which="both", alpha=0.25); a2.legend(fontsize=5.5, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(root, "fig_gap_vs_scale.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
