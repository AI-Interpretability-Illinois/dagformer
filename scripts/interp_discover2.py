"""Discovery pass 2: broader rules + effect-contrast verification.

Candidate rules per head:
  (a) data-driven token set: all target tokens among the head's top-200
      discovery tokens with count >= 3 and lift >= 5 vs corpus base rate;
  (b) curated lexical classes (pronoun_3rd, pronoun_it, pronoun_1st,
      open_bracket, close_bracket, quote, newline, sentence_end, preposition,
      conjunction, determiner) matched on the target token string;
  (c) structural: target seen in previous 128 tokens; position < 64;
      target char-class.
Verification on the held-out half for each candidate:
  mean dNLL on rule-matching tokens (m1) vs non-matching tokens (m0),
  contrast = m1 / m0; base rate; effect-mass coverage.
Named if contrast >= 3 and m1 >= 0.01 nats and base rate >= 0.2%.
Best rule = highest contrast among those passing; ties -> larger coverage.
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from interp_common import load_elh
from interp_probe import token_charclass

LEX = {
    "pronoun_3rd": {"he", "she", "his", "her", "him", "hers", "himself", "herself"},
    "pronoun_it_they": {"it", "its", "they", "them", "their", "theirs", "itself", "themselves"},
    "pronoun_1st": {"i", "me", "my", "mine", "we", "us", "our", "ours", "myself"},
    "pronoun_2nd": {"you", "your", "yours", "yourself"},
    "open_bracket": {"(", "[", "{", "<"},
    "close_bracket": {")", "]", "}", ">", "),", ")."},
    "quote": {'"', "'", "“", "”", "‘", "’", '",', '."', ',"', "'s"},
    "newline": {"\n", "\n\n", ",\n", ".\n", ".\n\n"},
    "sentence_end": {".", "?", "!", "?\"", "!\"", ".\"", "...", ":"},
    "preposition": {"of", "in", "on", "at", "by", "for", "with", "from", "to", "into", "about", "over", "under", "between", "through", "during", "without", "within"},
    "conjunction": {"and", "or", "but", "nor", "so", "yet", "because", "although", "while", "then", "than", "if", "when"},
    "determiner": {"the", "a", "an", "this", "that", "these", "those", "some", "any", "each", "every", "another", "other"},
    "be_have_modal": {"is", "are", "was", "were", "be", "been", "has", "have", "had", "will", "would", "can", "could", "should", "may", "might", "do", "does", "did"},
}


def norm_tok(s: str) -> str:
    return s.replace("Ġ", "").replace("Ċ", "\n").strip().lower()


def main():
    d = Path("experiments/results/interp/token_effects")
    elh = load_elh()
    cfg = elh.load_config("configs/fourway_300m_dagformer_mmap.yaml")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])
    corpus = torch.load(d / "corpus.pt", map_location="cpu", weights_only=False)
    ids, labels = corpus["ids"].numpy(), corpus["labels"].numpy()
    N, T = ids.shape; half = N // 2
    parts = [torch.load(p, map_location="cpu", weights_only=False) for p in sorted(glob.glob(str(d / "effects_heads*.pt")))]
    heads = [h for p in parts for h in p["heads"]]
    eff = torch.cat([p["effects"] for p in parts], 0).float().numpy()

    uniq = np.unique(labels)
    raw = dict(zip(uniq.tolist(), tok.convert_ids_to_tokens(uniq.tolist())))
    normed = {i: norm_tok(s) for i, s in raw.items()}
    cls_of = {i: token_charclass(s) for i, s in raw.items()}
    tcls = np.vectorize(cls_of.get)(labels)
    lex_sets = {name: {i for i, s in normed.items() if s in words or raw[i].replace("Ġ", "") in words}
                for name, words in LEX.items()}
    pos = np.tile(np.arange(T), (N, 1))
    seen = np.zeros((N, T), dtype=bool)
    for s in range(N):
        row = ids[s]
        for t in range(T):
            seen[s, t] = labels[s, t] in row[max(0, t - 127):t + 1]
    counts = Counter(labels.reshape(-1).tolist())
    total = labels.size

    def masks_for(rule, sl):
        kind, val = rule
        if kind == "tokset":
            return np.isin(labels[sl], list(val))
        if kind == "lex":
            return np.isin(labels[sl], list(lex_sets[val]))
        if kind == "class":
            return tcls[sl] == val
        if kind == "seen":
            return seen[sl]
        if kind == "pos<64":
            return pos[sl] < 64
        raise ValueError(kind)

    def describe(rule):
        kind, val = rule
        if kind == "tokset":
            return "target ∈ {" + ", ".join(repr(raw[i]) for i in list(val)[:8]) + ("…" if len(val) > 8 else "") + "}"
        if kind == "lex":
            return f"target is a {val.replace('_', ' ')} token"
        if kind == "class":
            return f"target char-class = {['whitespace','punctuation','digit','capitalized','lowercase','mixed'][val]}"
        if kind == "seen":
            return "target occurred in previous 128 tokens"
        return "position < 64"

    disc, ver = slice(0, half), slice(half, N)
    summary = []
    for k, (l, h) in enumerate(heads):
        e = eff[k]
        flat = np.clip(e[disc], 0, None).reshape(-1)
        top = np.argsort(-flat)[:200]
        ts, tt = np.unravel_index(top, e[disc].shape)
        tc = Counter(labels[disc][ts, tt].tolist())
        tokset = {t for t, n in tc.items() if n >= 3 and (n / 200) / (counts[t] / total) >= 5}
        cands = [("lex", name) for name in LEX] + [("class", c) for c in range(6)] + [("seen", None), ("pos<64", None)]
        if tokset:
            cands.append(("tokset", frozenset(tokset)))
        ev = e[ver]
        best = None
        for rule in cands:
            m = masks_for(rule, ver)
            br = float(m.mean())
            if br < 0.002 or br > 0.6:
                continue
            m1, m0 = float(ev[m].mean()), float(ev[~m].mean())
            contrast = m1 / max(m0, 1e-6) if m0 > 0 else float("inf")
            cover = float(np.clip(ev, 0, None)[m].sum() / (np.clip(ev, 0, None).sum() + 1e-9))
            ok = contrast >= 3 and m1 >= 0.01
            cand = {"rule": describe(rule), "m1": m1, "m0": m0, "contrast": contrast,
                    "base_rate": br, "coverage": cover, "named": ok}
            if best is None or (cand["named"], cand["contrast"] if np.isfinite(contrast) else 1e9, cover) > \
                    (best["named"], best["contrast"] if np.isfinite(best["contrast"]) else 1e9, best["coverage"]):
                best = cand
        summary.append({"head": f"L{l}/h{h}", "mean_dnll": float(e.mean()), **(best or {})})
    named = [r for r in summary if r.get("named")]
    named.sort(key=lambda r: -r["contrast"])
    print(f"[disc2] named heads: {len(named)}/{len(summary)}")
    for r in named:
        print(f"  {r['head']:8s} on-rule {r['m1']:+.3f} vs off-rule {r['m0']:+.4f} ({r['contrast']:.0f}x), "
              f"base {r['base_rate']*100:.1f}%, covers {r['coverage']*100:.0f}% | {r['rule']}")
    fam = Counter(r["rule"].split(" ∈")[0] if "∈" in r["rule"] else r["rule"] for r in named)
    print("[disc2] rule families:", dict(fam))
    json.dump(summary, open(d / "discovery2_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
