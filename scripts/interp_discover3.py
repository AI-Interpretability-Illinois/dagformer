"""Discovery pass 3: merged corpus (1000 + 4000 windows for unnamed heads)
and an extended rule set. Same verification criterion as pass 2
(held-out contrast >= 3x, on-rule mean effect >= 0.01 nats).

Extended rules (all on the target token unless noted):
  lexical classes: pass-2 set + number words, comparatives, time words,
    quantifiers, negation, question/relative words, months/units, code symbols
  structural: word-internal continuation (no leading space, alphabetic),
    word-initial; prev token is punctuation / newline / digit / capitalized;
    target == prev token; target in previous 8 tokens; document boundary;
    sentence-initial capitalized (prev is sentence end); number continuation
    (prev digit and target digit)
  data-driven: target token set; preceding token set
"""
from __future__ import annotations

import glob
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from interp_common import load_elh
from interp_probe import token_charclass
from interp_discover2 import LEX as LEX2, norm_tok

LEX = dict(LEX2)
LEX.update({
    "number_word": {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "hundred", "thousand", "million", "billion", "first", "second", "third", "fourth", "fifth", "half", "dozen"},
    "comparative": {"than", "more", "less", "most", "least", "better", "worse", "greater", "larger", "smaller", "higher", "lower", "rather"},
    "time_word": {"year", "years", "day", "days", "month", "months", "week", "weeks", "hour", "hours", "minute", "minutes", "ago", "later", "before", "after", "today", "yesterday", "tomorrow", "now", "then", "century"},
    "quantifier": {"all", "many", "some", "few", "several", "most", "both", "each", "every", "any", "much", "more", "enough"},
    "negation": {"not", "no", "never", "n't", "none", "nothing", "nobody", "neither", "nor", "cannot"},
    "wh_word": {"what", "who", "where", "when", "why", "how", "which", "whom", "whose", "whether"},
    "month_unit": {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december", "km", "kg", "mm", "cm", "%", "$", "€", "£", "°"},
    "code_symbol": {"=", "==", ";", "{", "}", "->", "=>", "/", "\\", "#", "<", ">", "|", "&", "*", "+"},
})


def main():
    d = Path("experiments/results/interp/token_effects")
    elh = load_elh()
    cfg = elh.load_config("configs/fourway_300m_dagformer_mmap.yaml")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])
    eos = 100257

    c0 = torch.load(d / "corpus.pt", map_location="cpu", weights_only=False)
    c1 = torch.load(d / "corpus_more.pt", map_location="cpu", weights_only=False)
    ids_all = np.concatenate([c0["ids"].numpy(), c1["ids"].numpy()])
    lab_all = np.concatenate([c0["labels"].numpy(), c1["labels"].numpy()])
    base_all = np.concatenate([torch.load(d / "baseline_nll.pt", map_location="cpu").float().numpy(),
                               torch.load(d / "baseline_nll_more.pt", map_location="cpu").float().numpy()])
    N, T = ids_all.shape
    print(f"[disc3] merged corpus {N}x{T}")

    # effects: old (all heads, 1000) + new (unnamed heads, 4000)
    old = {}
    for p in sorted(glob.glob(str(d / "effects_heads*.pt"))):
        if "_more" in p: continue
        x = torch.load(p, map_location="cpu", weights_only=False)
        for k, hh in enumerate(x["heads"]):
            old[tuple(hh)] = x["effects"][k]
    new = {}
    for p in sorted(glob.glob(str(d / "effects_custom_*_more.pt"))):
        x = torch.load(p, map_location="cpu", weights_only=False)
        for k, hh in enumerate(x["heads"]):
            new[tuple(hh)] = x["effects"][k]
    print(f"[disc3] old effects for {len(old)} heads, extra 4000-window effects for {len(new)} heads")

    # token features on merged corpus
    uniq = np.unique(np.concatenate([lab_all.reshape(-1), ids_all.reshape(-1)]))
    raw = dict(zip(uniq.tolist(), tok.convert_ids_to_tokens(uniq.tolist())))
    normed = {i: norm_tok(s) for i, s in raw.items()}
    cls_of = {i: token_charclass(s) for i, s in raw.items()}
    tcls = np.vectorize(cls_of.get)(lab_all)
    pcls = np.vectorize(cls_of.get)(ids_all)   # class of the token preceding the target
    lex_sets = {name: np.array(sorted({i for i, s in normed.items() if s in words or raw[i].replace("Ġ", "") in words}))
                for name, words in LEX.items()}
    word_internal = np.vectorize(lambda i: (not raw[i].startswith("Ġ")) and raw[i].isalpha())(lab_all)
    word_initial = np.vectorize(lambda i: raw[i].startswith("Ġ") and raw[i][1:].isalpha())(lab_all)
    sent_end_ids = lex_sets["sentence_end"]
    prev_sent_end = np.isin(ids_all, sent_end_ids)
    prev_newline = np.isin(ids_all, np.array(sorted({i for i, s in raw.items() if "Ċ" in s})))
    same_as_prev = lab_all == ids_all
    in_prev8 = np.zeros((N, T), dtype=bool)
    for k in range(1, 9):
        in_prev8[:, k:] |= lab_all[:, k:] == ids_all[:, :-k]
    seen128 = np.zeros((N, T), dtype=bool)
    for s in range(N):
        row = ids_all[s]; lab = lab_all[s]
        for t in range(T):
            seen128[s, t] = lab[t] in row[max(0, t - 127):t + 1]
    doc_boundary = lab_all == eos
    pos = np.tile(np.arange(T), (N, 1))
    counts = Counter(lab_all.reshape(-1).tolist()); total = lab_all.size

    def rule_mask(rule, sl):
        kind, val = rule
        if kind == "lex": return np.isin(lab_all[sl], lex_sets[val])
        if kind == "class": return tcls[sl] == val
        if kind == "prev_class": return pcls[sl] == val
        if kind == "tokset": return np.isin(lab_all[sl], list(val))
        if kind == "prevset": return np.isin(ids_all[sl], list(val))
        if kind == "word_internal": return word_internal[sl]
        if kind == "word_initial": return word_initial[sl]
        if kind == "sent_initial_cap": return prev_sent_end[sl] & (tcls[sl] == 3)
        if kind == "prev_newline": return prev_newline[sl]
        if kind == "same_as_prev": return same_as_prev[sl]
        if kind == "in_prev8": return in_prev8[sl]
        if kind == "seen128": return seen128[sl]
        if kind == "doc_boundary": return doc_boundary[sl]
        if kind == "number_cont": return (pcls[sl] == 2) & (tcls[sl] == 2)
        if kind == "pos<64": return pos[sl] < 64
        raise ValueError(kind)

    CN = ["whitespace", "punctuation", "digit", "capitalized", "lowercase", "mixed"]
    def describe(rule):
        kind, val = rule
        return {"lex": lambda: f"target is a {val.replace('_', ' ')} token",
                "class": lambda: f"target char-class = {CN[val]}",
                "prev_class": lambda: f"preceding token char-class = {CN[val]}",
                "tokset": lambda: "target ∈ {" + ", ".join(repr(raw[i]) for i in list(val)[:8]) + ("…" if len(val) > 8 else "") + "}",
                "prevset": lambda: "preceding token ∈ {" + ", ".join(repr(raw.get(i, i)) for i in list(val)[:8]) + "}",
                "word_internal": lambda: "target continues a word (no leading space)",
                "word_initial": lambda: "target starts a new word",
                "sent_initial_cap": lambda: "sentence-initial capitalized word",
                "prev_newline": lambda: "target follows a newline",
                "same_as_prev": lambda: "target equals the preceding token",
                "in_prev8": lambda: "target occurred in the previous 8 tokens",
                "seen128": lambda: "target occurred in the previous 128 tokens",
                "doc_boundary": lambda: "target is the document boundary",
                "number_cont": lambda: "digit following a digit",
                "pos<64": lambda: "position < 64"}[kind]()

    base_cands = [("lex", n) for n in LEX] + [("class", c) for c in range(6)] + [("prev_class", c) for c in range(6)] + \
                 [(k, None) for k in ("word_internal", "word_initial", "sent_initial_cap", "prev_newline", "same_as_prev",
                                      "in_prev8", "seen128", "doc_boundary", "number_cont", "pos<64")]

    summary = []
    for (l, h), e_old in sorted(old.items()):
        if (l, h) in new:
            e = np.concatenate([e_old.float().numpy(), new[(l, h)].float().numpy()])
            n_here = N
        else:
            e = e_old.float().numpy(); n_here = e.shape[0]
        half = n_here // 2
        disc, ver = slice(0, half), slice(half, n_here)
        flat = np.clip(e[disc], 0, None).reshape(-1)
        top = np.argsort(-flat)[:200]
        ts, tt = np.unravel_index(top, e[disc].shape)
        tc = Counter(lab_all[disc][ts, tt].tolist())
        tokset = {t for t, n in tc.items() if n >= 3 and (n / 200) / (counts[t] / total) >= 5}
        pc = Counter(ids_all[disc][ts, tt].tolist())
        prevset = {t for t, n in pc.items() if n >= 3 and (n / 200) / (counts.get(t, 1) / total) >= 5}
        cands = list(base_cands)
        if tokset: cands.append(("tokset", frozenset(tokset)))
        if prevset: cands.append(("prevset", frozenset(prevset)))
        ev = e[ver]; pm = np.clip(ev, 0, None)
        passing = []
        for rule in cands:
            m = rule_mask(rule, ver)
            br = float(m.mean())
            if br < 0.002 or br > 0.6: continue
            m1, m0 = float(ev[m].mean()), float(ev[~m].mean())
            contrast = m1 / m0 if m0 > 1e-6 else (float("inf") if m1 > 0 else 0.0)
            cover = float(pm[m].sum() / (pm.sum() + 1e-9))
            if contrast >= 3 and m1 >= 0.01:
                kind, val = rule
                spec = [kind, (sorted(int(x) for x in val) if isinstance(val, frozenset) else val)]
                passing.append({"rule": describe(rule), "spec": spec, "m1": m1, "m0": m0, "contrast": contrast,
                                "base_rate": br, "coverage": cover, "mask": m})
        passing.sort(key=lambda c: (-min(c["contrast"], 1e9), -c["coverage"]))
        kept = []
        for c in passing:   # greedy non-redundant selection (Jaccard < 0.5 with all kept)
            if all((c["mask"] & k["mask"]).sum() / max((c["mask"] | k["mask"]).sum(), 1) < 0.5 for k in kept):
                kept.append(c)
        union = np.zeros_like(ev, dtype=bool)
        for k in kept: union |= k["mask"]
        total_cover = float(pm[union].sum() / (pm.sum() + 1e-9)) if kept else 0.0
        best = kept[0] if kept else None
        funcs = [{k2: v for k2, v in c.items() if k2 != "mask"} for c in kept]
        summary.append({"head": f"L{l}/h{h}", "n_windows": n_here, "mean_dnll": float(e.mean()),
                        "n_functions": len(kept), "total_coverage": total_cover, "functions": funcs,
                        **({k2: v for k2, v in best.items() if k2 != "mask"} | {"named": True} if best else {"named": False})})
    named = [r for r in summary if r.get("named")]
    print(f"[disc3] named heads: {len(named)}/{len(summary)}")
    prev2 = {r["head"] for r in json.load(open(d / "discovery2_summary.json")) if r.get("named")}
    newly = [r for r in named if r["head"] not in prev2]
    print(f"[disc3] newly named (were unnamed in pass 2): {len(newly)}")
    for r in sorted(newly, key=lambda r: -r["contrast"]):
        print(f"  {r['head']:8s} n={r['n_windows']} on {r['m1']:+.3f} vs off {r['m0']:+.4f} ({min(r['contrast'],9999):.0f}x) base {r['base_rate']*100:.1f}% cover {r['coverage']*100:.0f}% | {r['rule']}")
    fam = Counter(re.sub(r" ∈ .*", " ∈ {...}", r["rule"]) for r in named)
    print("[disc3] rule families:", json.dumps(dict(fam.most_common()), ensure_ascii=False))
    nf = [r["n_functions"] for r in summary]
    print(f"[disc3] functions per head: mean {np.mean(nf):.2f}, max {max(nf)}, heads with >=2: {sum(1 for x in nf if x >= 2)}, total verified functions: {sum(nf)}")
    tc = [r["total_coverage"] for r in summary if r["named"]]
    print(f"[disc3] total coverage of named heads' effect mass by all their functions: median {np.median(tc)*100:.0f}%, max {max(tc)*100:.0f}%")
    for r in sorted(summary, key=lambda r: -r["n_functions"])[:12]:
        print(f"  {r['head']:8s} {r['n_functions']} functions, total cover {r['total_coverage']*100:.0f}%: " + " || ".join(f["rule"][:60] for f in r["functions"][:5]))
    json.dump(summary, open(d / "discovery3_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
