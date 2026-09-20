"""Which routing pathways carry the computation? (description, not steering)

The steering programme in this directory returned a clean negative: alpha is
causally inert at any magnitude that leaves the model intact. That result says
nothing about the *other* half of the interpretability claim, which does not
need alpha to be a good lever -- only a good basis.

Two things are exact here that are approximate in a vanilla transformer, and
both follow from alpha entering the mixture linearly (`y = sum_s alpha_s x_s`):

  exact ablation     Removing an edge means setting alpha_eff = 0 on that
                     coordinate, which is precisely "this head no longer reads
                     that layer". No donor distribution to choose, no mean to
                     patch in, no corrupted run to interpolate toward. The
                     counterfactual is the definition, not a stand-in for it.
  exact attribution  dL/dalpha_s = <dL/dy, x_s>, so one backward pass scores
                     all 3773 edges with no path-patching approximation and no
                     SAE in the loop. EAP in an ordinary transformer is
                     estimating the object that is handed to us here.

Two things are still approximate, and the code is built around measuring the
second rather than asserting it away. First, the correction MLPs re-read hidden
states, so the gradient collected below is a total derivative through them, not
the single inner product -- which is the right linearisation of the ablation we
actually perform, but not a closed form. Second, and more importantly,
attribution linearises a *finite* removal, and how good that is, is an
empirical question. `--part attribute` measures it by brute-force ablating a
stratified sample of edges and correlating predicted against actual damage. A
circuit story built on attribution alone is worth exactly that correlation.

Parts:

  --part channel     coarse knockout ladder. Zero `corr` alone, `pred` alone,
                     all hyperconnections, all sequential paths, each stream.
                     Answers which channel actually carries the computation,
                     rather than which one has the larger variance.
  --part attribute   exact per-edge attribution, validated against brute-force
                     ablation on a stratified sample.
  --part circuit     rank by attribution, ablate top-k against matched-random
                     and bottom-k controls, and test specificity by ablating a
                     circuit found on one corpus against another.

The task is next-token NLL on a corpus, not a behaviour score, and that is
deliberate: the model demonstrably has capability to lose, whereas only one
behaviour on this checkpoint has usable headroom. "Which pathways carry the
computation" is answerable where "which pathways carry lying" is not.

Usage:
    python experiments/interp/edge_ablation.py --part channel \
        --config $M/config.yaml --ckpt $M/checkpoint.pt --out $R/ablate_channel
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from circuit_common import (REPO, Edit, RoutingLayout, RoutingRunner,
                            load_models, load_tokenizer, save_json)
from interp_common import flatten_alpha, unflatten_alpha

DEFAULT_TOKENIZER = "/work/hdd/bfqt/shared/dagformer-models/tokenizer"

# Two distributions that live in the repo, so the specificity test needs no
# external data. They are genuinely different: one is Python, one is English.
# Every prose path is named rather than globbed over `*.md`, for two reasons:
# `experiments/results.md` is 83 KB of bilingual numeric tables and would have
# been two thirds of the corpus (neither prose nor a contrast with code), and a
# bare `experiments/interp/*.md` would sweep in untracked scratch files, so the
# corpus would differ between this run and a reviewer's checkout.
CORPORA = {
    "code": ["src/**/*.py", "scripts/*.py"],
    "prose": ["readme.md", "experiments/METHODOLOGY_*.md",
              "experiments/reviews/*.md", "experiments/prompts/*.md",
              "experiments/interp/README.md"],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="channel",
                    choices=("channel", "attribute", "circuit"))
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpora", nargs="+", default=["code", "prose"])
    ap.add_argument("--n-windows", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--hyper-only", action="store_true",
                    help="restrict attribution and circuits to true hyperconnections")
    # -- attribute --
    ap.add_argument("--n-validate", type=int, default=40,
                    help="edges brute-force ablated to check the linearisation")
    # -- circuit --
    ap.add_argument("--topk", type=int, nargs="+",
                    default=[8, 32, 128, 512, 2048])
    ap.add_argument("--attribution", default=None,
                    help="reuse a saved *_attribution.npz instead of recomputing")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------

MAX_CHARS = 4_000_000   # enough for any window budget; avoids tokenising the repo


def build_windows(tokenizer, patterns: list[str], seq_len: int,
                  n_windows: int) -> torch.Tensor:
    """Token windows [N, seq_len] spread over repo files matching `patterns`."""
    texts, n_chars = [], 0
    for pat in patterns:
        for p in sorted(REPO.glob(pat)):
            if p.is_file() and n_chars < MAX_CHARS:
                texts.append(p.read_text(errors="ignore"))
                n_chars += len(texts[-1])
    assert texts, f"no files matched {patterns}"
    toks = tokenizer("\n\n".join(texts), add_special_tokens=False)["input_ids"]
    n = min(n_windows, len(toks) // seq_len)
    assert n > 0, f"not enough text for one {seq_len}-token window from {patterns}"
    # Spread the windows across the whole stream. The first n*seq_len tokens are
    # one or two alphabetically-early files, which is a sample of a file rather
    # than of a corpus, and the specificity arm needs the corpora to differ by
    # language and not by which file happened to sort first.
    starts = np.linspace(0, len(toks) - seq_len, n).astype(int)
    return torch.tensor([toks[s:s + seq_len] for s in starts], dtype=torch.long)


# ---------------------------------------------------------------------------
# Losses and exact ablation
# ---------------------------------------------------------------------------

def nll_from_logits(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                           ids[:, 1:].reshape(-1).to(logits.device))


@torch.no_grad()
def corpus_nll(runner: RoutingRunner, windows: torch.Tensor, bs: int) -> float:
    tot, n = 0.0, 0
    for i in range(0, windows.shape[0], bs):
        ids = windows[i:i + bs]
        tot += float(nll_from_logits(runner.forward(ids), ids)) * ids.shape[0]
        n += ids.shape[0]
    return tot / n


def knockout(mask: np.ndarray, channels=("pred", "corr")) -> dict:
    """Edits that set alpha_eff to exactly zero on `mask`.

    An edge lives in both channels (alpha_eff = alpha_pred + delta_corr), so
    removing it means zeroing both. Zeroing only one is a different and much
    weaker intervention -- it leaves the other channel free to supply the
    connection -- which is why the channel ladder reports them separately.
    """
    m = torch.from_numpy(mask.astype(bool))
    e = Edit(op="zero", mask=m)
    return {c: (e if c in channels else None) for c in ("pred", "corr")}


# ---------------------------------------------------------------------------
# Exact attribution
# ---------------------------------------------------------------------------

def attribute(runner: RoutingRunner, fourway, predictor, layout: RoutingLayout,
              windows: torch.Tensor, bs: int) -> dict:
    """Per-edge first-order estimate of the damage from ablating that edge.

    alpha enters linearly, so one backward pass gives dL/dalpha for every edge
    at every position. Ablation moves the coordinate from its current value to
    zero, so the linear estimate of the resulting loss change is
    `-sum_{b,t} alpha_eff * dL/dalpha`. Positive = ablating this edge hurts.

    Units matter here, because the point of the part is to compare this number
    against a measured dNLL. The loss is a mean over each batch's tokens, so
    each batch enters the corpus mean with weight `len(batch) / N` -- without
    that weight the attribution comes out scaled by the batch size, which
    leaves the correlations intact but makes the magnitudes meaningless.
    """
    L, H, D = layout.L, layout.H, layout.D
    grad_sum = torch.zeros(D, dtype=torch.float64)
    alpha_sum = torch.zeros(D, dtype=torch.float64)
    prod_sum = torch.zeros(D, dtype=torch.float64)
    N, T = windows.shape
    for i in range(0, N, bs):
        ids = windows[i:i + bs].to(runner.device)
        w = ids.shape[0] / N
        with torch.no_grad():
            a0 = flatten_alpha(predictor(ids)).float()
            # alpha_eff needs the correction channel, which the hooks capture
            if runner.has_corr:
                _, corr = runner.forward_capture(ids)
                a_eff = a0 + corr.to(a0.device, a0.dtype)
            else:
                a_eff = a0
        a = a0.detach().clone().requires_grad_(True)
        logits = fourway(ids, unflatten_alpha(a, L, H))
        nll_from_logits(logits, ids).backward()
        g = a.grad.detach().double().cpu()
        ae = a_eff.double().cpu()
        grad_sum += g.sum((0, 1)) * w
        prod_sum += (ae * g).sum((0, 1)) * w
        alpha_sum += ae.sum((0, 1))
        a.grad = None
    return {"attribution": (-prod_sum).numpy(),
            "grad": grad_sum.numpy(),
            "alpha_eff": (alpha_sum / (N * T)).numpy()}


def validate_attribution(runner, layout, windows, bs, attribution, eligible,
                         n_validate, base_nll, rng) -> dict:
    """Brute-force ablate a stratified sample; correlate predicted vs actual.

    Stratified by predicted magnitude, because a uniform sample of 3234 edges
    is a sample of edges that do nothing, and a correlation computed over those
    measures noise against noise.
    """
    if n_validate <= 0:
        return {"n": 0, "pearson": float("nan"), "spearman": float("nan"),
                "sign_agreement": float("nan"), "predicted": [], "actual": [],
                "edges": [], "note": "skipped (--n-validate 0)"}
    idx = np.flatnonzero(eligible)
    order = idx[np.argsort(-np.abs(attribution[idx]))]
    n_top = min(n_validate // 2, len(order))
    rest = order[n_top:]
    n_rand = min(n_validate - n_top, len(rest))
    picks = np.concatenate([order[:n_top],
                            rng.choice(rest, n_rand, replace=False)])
    pred, actual = [], []
    for e in picks:
        m = np.zeros(layout.D, dtype=bool)
        m[e] = True
        with runner.edited(**knockout(m)):
            actual.append(corpus_nll(runner, windows, bs) - base_nll)
        pred.append(float(attribution[e]))
    pred, actual = np.array(pred), np.array(actual)
    def _r(a, b):
        if a.std() < 1e-12 or b.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    return {"n": int(len(picks)), "pearson": _r(pred, actual),
            "spearman": _r(rank(pred), rank(actual)),
            "sign_agreement": float((np.sign(pred) == np.sign(actual)).mean()),
            "predicted": pred.tolist(), "actual": actual.tolist(),
            "edges": [layout.edges[int(e)].name for e in picks]}


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------

def part_channel(args, runner, layout, corpora) -> dict:
    """Coarse knockouts: which channel and which edge class carries the model."""
    D = layout.D
    seq = (layout.src_arr == layout.layer_arr)
    arms = {
        "corr_off  (alpha_eff = alpha_pred)": (np.ones(D, bool), ("corr",)),
        "pred_off  (alpha_eff = delta_corr)": (np.ones(D, bool), ("pred",)),
        "both_off  (alpha_eff = 0)": (np.ones(D, bool), ("pred", "corr")),
        "hyper_off (sequential only)": (layout.hyper_arr, ("pred", "corr")),
        "seq_off   (hyperconnections only)": (seq, ("pred", "corr")),
    }
    for s in ("q", "k", "v", "r"):
        arms[f"{s}_off"] = (layout.stream_arr == s, ("pred", "corr"))

    base = {c: corpus_nll(runner, w, args.batch_size) for c, w in corpora.items()}
    print(f"[channel] baseline NLL {base}", flush=True)
    rows = []
    for name, (mask, channels) in arms.items():
        r = {"arm": name, "n_edges": int(mask.sum()), "channels": list(channels)}
        with runner.edited(**knockout(mask, channels)):
            for c, w in corpora.items():
                r[f"nll_{c}"] = corpus_nll(runner, w, args.batch_size)
                r[f"rise_{c}"] = r[f"nll_{c}"] - base[c]
        rows.append(r)
        print(f"  {name:<36} " + "  ".join(
            f"{c}:{r[f'rise_{c}']:+.3f}" for c in corpora), flush=True)
    return {"baseline_nll": base, "rows": rows}


def part_attribute(args, runner, fourway, predictor, layout, corpora, rng) -> dict:
    eligible = layout.hyper_arr.copy() if args.hyper_only else np.ones(layout.D, bool)
    out = {"eligible": int(eligible.sum()), "corpora": {}}
    for c, w in corpora.items():
        print(f"[attribute] {c}: {w.shape[0]} windows", flush=True)
        att = attribute(runner, fourway, predictor, layout, w, args.batch_size)
        base = corpus_nll(runner, w, args.batch_size)
        val = validate_attribution(runner, layout, w, args.batch_size,
                                   att["attribution"], eligible,
                                   args.n_validate, base, rng)
        print(f"  pearson {val['pearson']:.3f}  spearman {val['spearman']:.3f}  "
              f"sign {val['sign_agreement']:.2f}", flush=True)
        out["corpora"][c] = {"baseline_nll": base, "validation": val,
                             "attribution": att["attribution"].tolist()}
    return out


def part_circuit(args, runner, layout, corpora, att_by_corpus, rng) -> dict:
    eligible = layout.hyper_arr.copy() if args.hyper_only else np.ones(layout.D, bool)
    idx = np.flatnonzero(eligible)
    out = {"corpora": {}, "specificity": {}}
    base = {c: corpus_nll(runner, w, args.batch_size) for c, w in corpora.items()}
    tops: dict[str, dict[int, np.ndarray]] = {}

    for c, w in corpora.items():
        att = np.asarray(att_by_corpus[c])
        order = idx[np.argsort(-att[idx])]          # most damaging first
        rows, tops[c] = [], {}
        for k in args.topk:
            if k > len(order):
                continue
            sets = {"top": order[:k], "bottom": order[-k:],
                    "random": rng.choice(order, k, replace=False)}
            tops[c][k] = sets["top"]
            r = {"k": k}
            for label, sel in sets.items():
                m = np.zeros(layout.D, bool)
                m[sel] = True
                with runner.edited(**knockout(m)):
                    r[f"rise_{label}"] = corpus_nll(runner, w, args.batch_size) - base[c]
            rows.append(r)
            print(f"  {c} k={k:<5d} top {r['rise_top']:+.3f}  "
                  f"random {r['rise_random']:+.3f}  bottom {r['rise_bottom']:+.3f}",
                  flush=True)
        out["corpora"][c] = {"baseline_nll": base[c], "rows": rows}

    # Specificity: does a circuit found on one corpus damage the other?
    names = list(corpora)
    if len(names) >= 2:
        for k in args.topk:
            if any(k not in tops[c] for c in names):
                continue
            a, b = names[0], names[1]
            inter = len(set(tops[a][k].tolist()) & set(tops[b][k].tolist()))
            cross = {}
            for src in (a, b):
                m = np.zeros(layout.D, bool)
                m[tops[src][k]] = True
                with runner.edited(**knockout(m)):
                    for tgt in (a, b):
                        cross[f"{src}_on_{tgt}"] = (
                            corpus_nll(runner, corpora[tgt], args.batch_size)
                            - base[tgt])
            out["specificity"][str(k)] = {
                "jaccard": inter / (2 * k - inter), "overlap": inter, **cross}
            print(f"  specificity k={k}: jaccard "
                  f"{out['specificity'][str(k)]['jaccard']:.3f}", flush=True)
    return out


# ---------------------------------------------------------------------------

def report(res: dict, part: str, corpora: list[str]) -> str:
    L = [f"# Routing-edge ablation ({part})", ""]
    if part == "channel":
        L += ["Baseline NLL: " + ", ".join(
            f"`{c}` {v:.4f}" for c, v in res["baseline_nll"].items()), "",
            "Each arm sets `alpha_eff` to exactly zero on the named coordinates "
            "-- in both channels unless the arm says otherwise.", "",
            "| arm | edges | " + " | ".join(f"dNLL {c}" for c in corpora) + " |",
            "|---|---|" + "---|" * len(corpora)]
        for r in res["rows"]:
            L.append(f"| {r['arm']} | {r['n_edges']} | " + " | ".join(
                f"{r[f'rise_{c}']:+.3f}" for c in corpora) + " |")
    elif part == "attribute":
        L += ["How well does the gradient predict the ablation? Brute-force "
              "ablation of a stratified sample of edges, against the "
              "first-order estimate.", "",
              "| corpus | baseline NLL | n | Pearson | Spearman | sign agreement |",
              "|---|---|---|---|---|---|"]
        for c, d in res["corpora"].items():
            v = d["validation"]
            L.append(f"| {c} | {d['baseline_nll']:.4f} | {v['n']} | "
                     f"{v['pearson']:.3f} | {v['spearman']:.3f} | "
                     f"{v['sign_agreement']:.2f} |")
    else:
        L += ["Ablating the k edges attribution ranks as most damaging, against "
              "matched-size random and least-damaging controls.", ""]
        for c, d in res["corpora"].items():
            L += [f"**`{c}`** (baseline NLL {d['baseline_nll']:.4f})", "",
                  "| k | dNLL top-k | dNLL random-k | dNLL bottom-k |",
                  "|---|---|---|---|"]
            for r in d["rows"]:
                L.append(f"| {r['k']} | {r['rise_top']:+.3f} | "
                         f"{r['rise_random']:+.3f} | {r['rise_bottom']:+.3f} |")
            L.append("")
        if res["specificity"]:
            L += ["Specificity — is the circuit corpus-specific, or just the "
                  "model's load-bearing wiring?", "",
                  "| k | Jaccard | " + " | ".join(
                      k for k in next(iter(res["specificity"].values()))
                      if "_on_" in k) + " |",
                  "|---|---|" + "---|" * sum(
                      1 for k in next(iter(res["specificity"].values()))
                      if "_on_" in k)]
            for k, d in res["specificity"].items():
                L.append(f"| {k} | {d['jaccard']:.3f} | " + " | ".join(
                    f"{v:+.3f}" for kk, v in d.items() if "_on_" in kk) + " |")
    return "\n".join(L) + "\n"


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    # float32 throughout: the attribution part differentiates through the whole
    # stack, and bf16 gradients would blur the very correlation being measured.
    cfg, fourway, predictor = load_models(args.config, args.ckpt, device,
                                          dtype=torch.float32)
    for p in list(fourway.parameters()) + list(predictor.parameters()):
        p.requires_grad_(False)
    layout = RoutingLayout(cfg["num_hidden_layers"], cfg["num_attention_heads"])
    tokenizer = load_tokenizer(cfg, args.tokenizer)
    runner = RoutingRunner(layout, predictor, fourway, device=device)
    corpora = {c: build_windows(tokenizer, CORPORA[c], args.seq_len, args.n_windows)
               for c in args.corpora}
    print(f"[edge_ablation] L={layout.L} H={layout.H} D={layout.D}; corpora " +
          ", ".join(f"{c}={w.shape[0]}x{w.shape[1]}" for c, w in corpora.items()),
          flush=True)

    if args.part == "channel":
        res = part_channel(args, runner, layout, corpora)
    elif args.part == "attribute":
        res = part_attribute(args, runner, fourway, predictor, layout, corpora, rng)
    else:
        if args.attribution:
            z = np.load(args.attribution, allow_pickle=True)
            att = {c: z[c] for c in args.corpora}
        else:
            a = part_attribute(args, runner, fourway, predictor, layout, corpora, rng)
            att = {c: np.asarray(d["attribution"]) for c, d in a["corpora"].items()}
        res = part_circuit(args, runner, layout, corpora, att, rng)

    res.update(part=args.part, hyper_only=args.hyper_only, args=vars(args))
    text = report(res, args.part, list(corpora))
    print("\n" + text)
    if args.out:
        out = Path(args.out)
        save_json(res, out.with_suffix(".json"))
        out.with_suffix(".md").write_text(text)
        if args.part == "attribute":
            np.savez(out.with_suffix(".npz"),
                     **{c: np.asarray(d["attribution"])
                        for c, d in res["corpora"].items()})
        print(f"[saved] {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
