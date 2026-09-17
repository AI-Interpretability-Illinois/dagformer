"""Circuit attribution over ALL connections (all 176 heads) and the FULL
generic rule library — functions are not derived from single-head naming.

For every rule in the library (lexical classes, char classes, preceding-token
classes, structural rules) with >= 200 held-out matching tokens, compute each
connection's mean effect on rule tokens vs others (verification half),
z-normalize per connection across rules, take the top-10 (s > 0) as the
circuit, and report how many of its connections come from heads that had no
single-head name. Output: circuits_all.json (same schema as
circuits_specific.json so interp_circuit_verify.py can consume it).
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
from interp_discover3 import LEX, norm_tok


def main():
    d = Path("experiments/results/interp/token_effects")
    elh = load_elh()
    cfg = elh.load_config("configs/fourway_300m_dagformer_mmap.yaml")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])
    corpus = torch.load(d / "corpus.pt", map_location="cpu", weights_only=False)
    ids, labels = corpus["ids"].numpy(), corpus["labels"].numpy()
    N, T = ids.shape; half = N // 2; ver = slice(half, N)
    lab, idv = labels[ver], ids[ver]

    shard_files = sorted(glob.glob(str(d / "effects_conn_*.pt")))
    conns = []
    for p in shard_files:
        x = torch.load(p, map_location="cpu", weights_only=False)
        conns += [tuple(c) for c in x["conns"]]; del x
    print(f"[all] {len(conns)} connections from {len(shard_files)} shards")
    named_heads = {r["head"] for r in json.load(open(d / "discovery3_summary.json")) if r.get("named")}

    uniq = np.unique(np.concatenate([lab.reshape(-1), idv.reshape(-1)]))
    raw = dict(zip(uniq.tolist(), tok.convert_ids_to_tokens(uniq.tolist())))
    normed = {i: norm_tok(s) for i, s in raw.items()}
    cls_of = {i: token_charclass(s) for i, s in raw.items()}
    tcls = np.vectorize(cls_of.get)(lab); pcls = np.vectorize(cls_of.get)(idv)
    lex_sets = {n: np.array(sorted({i for i, s in normed.items() if s in w or raw[i].replace("Ġ", "") in w})) for n, w in LEX.items()}
    word_internal = np.vectorize(lambda i: (not raw[i].startswith("Ġ")) and raw[i].isalpha())(lab)
    word_initial = np.vectorize(lambda i: raw[i].startswith("Ġ") and raw[i][1:].isalpha())(lab)
    prev_sent_end = np.isin(idv, lex_sets["sentence_end"])
    prev_newline = np.isin(idv, np.array(sorted({i for i, s in raw.items() if "Ċ" in s})))
    same_as_prev = lab == idv
    in_prev8 = np.zeros_like(lab, dtype=bool)
    for k in range(1, 9): in_prev8[:, k:] |= lab[:, k:] == idv[:, :-k]
    seen128 = np.zeros_like(lab, dtype=bool)
    for s in range(lab.shape[0]):
        for t in range(T): seen128[s, t] = lab[s, t] in idv[s, max(0, t - 127):t + 1]
    pos = np.tile(np.arange(T), (lab.shape[0], 1))
    CN = ["whitespace", "punctuation", "digit", "capitalized", "lowercase", "mixed"]
    rules = {}
    for n in LEX: rules[f"target is a {n.replace('_', ' ')} token"] = np.isin(lab, lex_sets[n])
    for c in range(6): rules[f"target char-class = {CN[c]}"] = tcls == c
    for c in range(6): rules[f"preceding token char-class = {CN[c]}"] = pcls == c
    rules.update({"target continues a word (no leading space)": word_internal, "target starts a new word": word_initial,
                  "sentence-initial capitalized word": prev_sent_end & (tcls == 3), "target follows a newline": prev_newline,
                  "target equals the preceding token": same_as_prev, "target occurred in the previous 8 tokens": in_prev8,
                  "target occurred in the previous 128 tokens": seen128, "target is the document boundary": lab == 100257,
                  "digit following a digit": (pcls == 2) & (tcls == 2), "position < 64": pos < 64})
    # add previously verified data-driven token sets
    tokset_specs = {}
    for r in json.load(open(d / "discovery3_summary.json")):
        for f in r.get("functions", []):
            if f["spec"][0] == "tokset":
                key = f"[{r['head']}] " + f["rule"]
                rules[key] = np.isin(lab, np.array(f["spec"][1]))
                tokset_specs[key] = [int(x) for x in f["spec"][1]]
    names = [n for n, m in rules.items() if 200 <= m.sum() <= 0.6 * m.size]
    M = np.stack([rules[n].reshape(-1) for n in names], 1).astype(np.float32)
    cnt1 = M.sum(0); cnt0 = M.shape[0] - cnt1
    print(f"[all] {len(names)} rules with >=200 held-out tokens")

    M1 = np.zeros((len(conns), len(names)), dtype=np.float64); tot = np.zeros(len(conns)); off = 0
    for p in shard_files:
        x = torch.load(p, map_location="cpu", weights_only=False)
        Es = x["effects"][:, ver].float().numpy().reshape(x["effects"].shape[0], -1)
        M1[off:off + Es.shape[0]] = Es @ M; tot[off:off + Es.shape[0]] = Es.sum(1); off += Es.shape[0]
        del x, Es
    m1 = M1 / cnt1[None, :]; m0 = (tot[:, None] - M1) / cnt0[None, :]
    s = m1 - m0
    zs = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)
    np.savez(d / "circuit_matrix_all.npz", m1=m1, m0=m0, conns=np.array(conns, dtype=object), rules=np.array(names))

    out = []
    for j, rule in enumerate(names):
        order = [i for i in np.argsort(-zs[:, j]) if s[i, j] > 0][:12]
        if not order: continue
        top = [{"conn": f"L{conns[i][0]}/h{conns[i][2]}/{conns[i][1]}<-src{conns[i][3]}", "layer": int(conns[i][0]), "head": int(conns[i][2]),
                "stream": conns[i][1], "src": int(conns[i][3]), "z": float(zs[i, j]), "effect_on_rule": float(m1[i, j]), "effect_off_rule": float(m0[i, j])} for i in order]
        heads = Counter(f"L{t['layer']}/h{t['head']}" for t in top[:10])
        from_unnamed = sum(v for h, v in heads.items() if h not in named_heads)
        out.append({"rule": rule, "top_connections": top, "n_heads_top10": len(heads), "n_layers_top10": len({t['layer'] for t in top[:10]}),
                    "unnamed_head_share_top10": from_unnamed / 10, "max_z": float(zs[order[0], j]), "top_effect_on_rule": float(m1[order[0], j]),
                    "tokset": tokset_specs.get(rule)})
    out.sort(key=lambda r: -r["max_z"])
    json.dump(out, open(d / "circuits_all.json", "w"), indent=1)
    print(f"[all] circuits for {len(out)} rules; mean share of top-10 connections from UNNAMED heads: {np.mean([r['unnamed_head_share_top10'] for r in out])*100:.0f}%")
    print(f"[all] rules whose circuit is >=50% unnamed-head connections: {sum(1 for r in out if r['unnamed_head_share_top10'] >= 0.5)}")
    for r in out[:30]:
        print(f"  z={r['max_z']:.1f} unnamed={r['unnamed_head_share_top10']*100:.0f}% heads={r['n_heads_top10']} {r['rule'][:55]:55s} | " + ", ".join(f"{t['conn']}({t['effect_on_rule']:+.2f})" for t in r["top_connections"][:3]))


if __name__ == "__main__":
    main()
