"""Discovery-style head naming from per-token causal effects.

Inputs (from interp_token_effects.py): corpus ids/labels, baseline per-token
NLL, per-head per-token effects (dNLL when the head's routing is ablated),
per-head per-token routing-correction magnitude.

Protocol (split-half): sequences [0, n/2) are the DISCOVERY half, [n/2, n) the
VERIFICATION half.
  1. Per head: concentration of its effect mass; top-K discovery tokens by
     dNLL; enrichment statistics of those tokens (target char class, target
     token identity, preceding token, position, seen-earlier-in-context).
  2. Auto-hypothesis: a transparent rule derived from the strongest enrichment
     (e.g. "target token in {')', ']'}", "target char-class = digit",
     "target seen in previous 128 tokens", "position < 64").
  3. Verification: on the held-out half, the fraction of the head's positive
     effect mass that falls on rule-matching tokens, divided by the rule's
     base rate (enrichment ratio). A head is NAMED only if enrichment >= 3
     and the rule covers >= 30% of its effect mass.
  4. Per-head markdown card with the top snippets for human inspection.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from interp_common import load_elh
from interp_probe import token_charclass

CLASS_NAMES = ["whitespace", "punctuation", "digit", "capitalized", "lowercase", "mixed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fourway_300m_dagformer_mmap.yaml")
    ap.add_argument("--dir", default="experiments/results/interp/token_effects")
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--min-enrich", type=float, default=3.0)
    ap.add_argument("--min-cover", type=float, default=0.30)
    args = ap.parse_args()
    d = Path(args.dir)
    elh = load_elh()
    cfg = elh.load_config(args.config)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])

    corpus = torch.load(d / "corpus.pt", map_location="cpu", weights_only=False)
    ids, labels = corpus["ids"].numpy(), corpus["labels"].numpy()
    N, T = ids.shape
    half = N // 2
    base = torch.load(d / "baseline_nll.pt", map_location="cpu").float().numpy()
    parts = [torch.load(p, map_location="cpu", weights_only=False) for p in sorted(glob.glob(str(d / "effects_heads*.pt")))]
    heads = [h for p in parts for h in p["heads"]]
    eff = torch.cat([p["effects"] for p in parts], 0).float().numpy()   # [n_heads, N, T]
    print(f"[disc] {len(heads)} heads, corpus {N}x{T}")

    # token-level features (target side)
    uniq = np.unique(labels)
    tok_str = dict(zip(uniq.tolist(), tok.convert_ids_to_tokens(uniq.tolist())))
    cls_of = {i: token_charclass(s) for i, s in tok_str.items()}
    tcls = np.vectorize(cls_of.get)(labels)
    pos = np.tile(np.arange(T), (N, 1))
    seen = np.zeros((N, T), dtype=bool)
    for s in range(N):
        row = ids[s]
        for t in range(T):
            seen[s, t] = labels[s, t] in row[max(0, t - 127):t + 1]
    prev_tok = ids  # token at position t is the one immediately before target labels[t]

    base_rates = {
        **{f"class={CLASS_NAMES[c]}": float((tcls[half:] == c).mean()) for c in range(6)},
        "seen_in_prev128": float(seen[half:].mean()),
        "pos<64": float((pos[half:] < 64).mean()),
    }

    def rule_mask(rule, sl):
        kind, val = rule
        if kind == "class":
            return tcls[sl] == val
        if kind == "target_in":
            return np.isin(labels[sl], list(val))
        if kind == "prev_in":
            return np.isin(prev_tok[sl], list(val))
        if kind == "seen":
            return seen[sl]
        if kind == "pos<64":
            return pos[sl] < 64
        raise ValueError(kind)

    def rule_name(rule):
        kind, val = rule
        if kind == "class":
            return f"target char-class = {CLASS_NAMES[val]}"
        if kind in ("target_in", "prev_in"):
            toks = [repr(tok_str.get(int(v), str(v))) for v in val]
            return ("target" if kind == "target_in" else "preceding") + " token in {" + ", ".join(toks) + "}"
        if kind == "seen":
            return "target token occurred in the previous 128 tokens"
        return "position < 64"

    cards_dir = d / "cards"; cards_dir.mkdir(exist_ok=True)
    summary = []
    disc, ver = slice(0, half), slice(half, N)
    for k, (l, h) in enumerate(heads):
        e = eff[k]
        pos_mass = np.clip(e, 0, None)
        total = pos_mass[disc].sum() + 1e-9
        flat = pos_mass[disc].reshape(-1)
        top = np.argsort(-flat)[:args.topk]
        conc = float(flat[top].sum() / total)
        ts, tt = np.unravel_index(top, pos_mass[disc].shape)
        # enrichment candidates
        cand = []
        cc = Counter(tcls[disc][ts, tt].tolist())
        for c, n in cc.most_common(2):
            cand.append((("class", c), n / args.topk / max(base_rates[f"class={CLASS_NAMES[c]}"], 1e-6), n / args.topk))
        tc = Counter(labels[disc][ts, tt].tolist())
        top_targets = tuple(t for t, _ in tc.most_common(3))
        cover_t = sum(tc[t] for t in top_targets) / args.topk
        br_t = float(np.isin(labels[ver], list(top_targets)).mean())
        cand.append((("target_in", top_targets), cover_t / max(br_t, 1e-6), cover_t))
        pc = Counter(prev_tok[disc][ts, tt].tolist())
        top_prev = tuple(t for t, _ in pc.most_common(3))
        cover_p = sum(pc[t] for t in top_prev) / args.topk
        br_p = float(np.isin(prev_tok[ver], list(top_prev)).mean())
        cand.append((("prev_in", top_prev), cover_p / max(br_p, 1e-6), cover_p))
        seen_rate = float(seen[disc][ts, tt].mean())
        cand.append((("seen", None), seen_rate / max(base_rates["seen_in_prev128"], 1e-6), seen_rate))
        early = float((tt < 64).mean())
        cand.append((("pos<64", None), early / max(base_rates["pos<64"], 1e-6), early))
        # pick hypothesis: highest lift among candidates covering >= 25% of top-K
        cand = [c for c in cand if c[2] >= 0.25]
        cand.sort(key=lambda c: -c[1])
        hyp = cand[0][0] if cand else None
        # verification on held-out half: effect-mass coverage / base rate
        verified, v_cover, v_enrich, v_base = False, 0.0, 0.0, 0.0
        if hyp is not None:
            m = rule_mask(hyp, ver)
            pm = pos_mass[ver]
            v_cover = float(pm[m].sum() / (pm.sum() + 1e-9))
            v_base = float(m.mean())
            v_enrich = v_cover / max(v_base, 1e-6)
            verified = v_enrich >= args.min_enrich and v_cover >= args.min_cover
        name = rule_name(hyp) if hyp is not None else "none"
        summary.append({"head": f"L{l}/h{h}", "mean_dnll": float(e.mean()),
                        "concentration_top200": conc, "hypothesis": name,
                        "verify_cover": v_cover, "verify_base_rate": v_base,
                        "verify_enrichment": v_enrich, "verified": bool(verified)})
        # card with snippets
        lines = [f"# Head L{l}/h{h}", f"mean dNLL {e.mean():+.4f}; top-200 share of positive effect mass {conc:.2f}",
                 f"hypothesis: {name}", f"verification (held-out half): coverage {v_cover:.2f}, base rate {v_base:.3f}, enrichment {v_enrich:.1f}x, verified={verified}", "",
                 "## top-30 tokens by effect (discovery half)  [context ... >>target<<]"]
        for s_, t_ in list(zip(ts, tt))[:30]:
            ctx = tok.decode(ids[s_, max(0, t_ - 14):t_ + 1].tolist()).replace("\n", "⏎")
            tgt = tok.decode([int(labels[s_, t_])]).replace("\n", "⏎")
            lines.append(f"- dNLL {e[s_, t_]:+.2f} (base {base[s_, t_]:.2f}) | {ctx[-90:]} >>{tgt}<<")
        (cards_dir / f"L{l}_h{h}.md").write_text("\n".join(lines))
        if (k + 1) % 20 == 0:
            print(f"[disc] {k + 1}/{len(heads)} heads processed", flush=True)

    summary.sort(key=lambda r: -r["verify_enrichment"])
    json.dump({"base_rates": base_rates, "heads": summary}, open(d / "discovery_summary.json", "w"), indent=1)
    nv = sum(r["verified"] for r in summary)
    print(f"[disc] verified names: {nv}/{len(summary)}")
    for r in summary[:40]:
        print(f"  {r['head']:8s} dNLL={r['mean_dnll']:+.4f} conc={r['concentration_top200']:.2f} "
              f"enrich={r['verify_enrichment']:.1f}x cover={r['verify_cover']:.2f} {'V' if r['verified'] else ' '} {r['hypothesis']}")


if __name__ == "__main__":
    main()
