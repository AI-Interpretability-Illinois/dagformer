"""Circuit attribution: function x connection matrix.

For every verified function (head, rule) from discovery pass 3, compute each
scanned connection's mean per-token effect on rule-matching tokens (m1) vs
non-matching tokens (m0) on the verification half of the 1000-window corpus.
Rank connections by m1 - m0 -> circuit candidates (which may span many heads
and layers). Report, per function: top connections, number of distinct
heads/layers among the top-10, and the share of the top-10 that lie outside
the function's own head.
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

    shard_files = sorted(glob.glob(str(d / "effects_conn_g*.pt")))
    conns = []
    for p in shard_files:
        x = torch.load(p, map_location="cpu", weights_only=False)
        conns += [tuple(c) for c in x["conns"]]
        del x
    print(f"[circ] {len(conns)} connections in {len(shard_files)} shards (streamed)")

    # token features (verification half)
    uniq = np.unique(np.concatenate([labels.reshape(-1), ids.reshape(-1)]))
    raw = dict(zip(uniq.tolist(), tok.convert_ids_to_tokens(uniq.tolist())))
    normed = {i: norm_tok(s) for i, s in raw.items()}
    cls_of = {i: token_charclass(s) for i, s in raw.items()}
    lab, idv = labels[ver], ids[ver]
    tcls = np.vectorize(cls_of.get)(lab); pcls = np.vectorize(cls_of.get)(idv)
    lex_sets = {n: np.array(sorted({i for i, s in normed.items() if s in w or raw[i].replace("Ġ", "") in w})) for n, w in LEX.items()}
    word_internal = np.vectorize(lambda i: (not raw[i].startswith("Ġ")) and raw[i].isalpha())(lab)
    word_initial = np.vectorize(lambda i: raw[i].startswith("Ġ") and raw[i][1:].isalpha())(lab)
    prev_sent_end = np.isin(idv, lex_sets["sentence_end"])
    prev_newline = np.isin(idv, np.array(sorted({i for i, s in raw.items() if "Ċ" in s})))
    same_as_prev = lab == idv
    in_prev8 = np.zeros_like(lab, dtype=bool)
    for k in range(1, 9):
        in_prev8[:, k:] |= lab[:, k:] == idv[:, :-k]
    seen128 = np.zeros_like(lab, dtype=bool)
    for s in range(lab.shape[0]):
        for t in range(T):
            seen128[s, t] = lab[s, t] in idv[s, max(0, t - 127):t + 1]
    pos = np.tile(np.arange(T), (lab.shape[0], 1))

    def mask_of(spec):
        kind, val = spec
        return {"lex": lambda: np.isin(lab, lex_sets[val]), "class": lambda: tcls == val,
                "prev_class": lambda: pcls == val, "tokset": lambda: np.isin(lab, np.array(val)),
                "prevset": lambda: np.isin(idv, np.array(val)), "word_internal": lambda: word_internal,
                "word_initial": lambda: word_initial, "sent_initial_cap": lambda: prev_sent_end & (tcls == 3),
                "prev_newline": lambda: prev_newline, "same_as_prev": lambda: same_as_prev,
                "in_prev8": lambda: in_prev8, "seen128": lambda: seen128,
                "doc_boundary": lambda: lab == 100257, "number_cont": lambda: (pcls == 2) & (tcls == 2),
                "pos<64": lambda: pos < 64}[kind]()

    S = json.load(open(d / "discovery3_summary.json"))
    functions = [(r["head"], f) for r in S if r.get("named") for f in r["functions"]]
    functions = [(h, f) for h, f in functions if mask_of(f["spec"]).sum() >= 200]
    print(f"[circ] {len(functions)} verified functions (>=200 held-out tokens)")
    M = np.stack([mask_of(f["spec"]).reshape(-1) for _, f in functions], 1).astype(np.float32)  # [tok, n_func]
    cnt1 = M.sum(0); cnt0 = M.shape[0] - cnt1
    M1 = np.zeros((len(conns), len(functions)), dtype=np.float64)
    tot = np.zeros(len(conns), dtype=np.float64)
    off = 0
    for p in shard_files:
        x = torch.load(p, map_location="cpu", weights_only=False)
        Es = x["effects"][:, ver].float().numpy().reshape(x["effects"].shape[0], -1)   # [n, tok]
        M1[off:off + Es.shape[0]] = Es @ M
        tot[off:off + Es.shape[0]] = Es.sum(1)
        off += Es.shape[0]
        del x, Es
        print(f"[circ] streamed {off}/{len(conns)}", flush=True)
    m1_all = M1 / cnt1[None, :]
    m0_all = (tot[:, None] - M1) / cnt0[None, :]
    np.savez(d / "circuit_matrix.npz", m1=m1_all, m0=m0_all, conns=np.array(conns, dtype=object),
             heads=np.array([h for h, _ in functions]), rules=np.array([f["rule"] for _, f in functions]),
             specs=np.array([json.dumps(f["spec"]) for _, f in functions]))
    out = []
    for j, (head, f) in enumerate(functions):
        m1 = m1_all[:, j]; m0 = m0_all[:, j]
        score = m1 - m0
        m = M[:, j].astype(bool)
        order = np.argsort(-score)[:10]
        top = [{"conn": f"L{conns[i][0]}/h{conns[i][2]}/{conns[i][1]}<-src{conns[i][3]}", "m1": float(m1[i]), "m0": float(m0[i])} for i in order]
        heads_in_top = {f"L{conns[i][0]}/h{conns[i][2]}" for i in order}
        layers_in_top = {conns[i][0] for i in order}
        own = sum(1 for i in order if f"L{conns[i][0]}/h{conns[i][2]}" == head)
        out.append({"head": head, "rule": f["rule"], "spec": f["spec"], "n_tokens": int(m.sum()),
                    "top_connections": top, "n_heads_top10": len(heads_in_top), "n_layers_top10": len(layers_in_top),
                    "own_head_share_top10": own / 10, "top1_score": float(score[order[0]])})
    out.sort(key=lambda r: -r["top1_score"])
    json.dump(out, open(d / "circuits.json", "w"), indent=1)
    nh = [r["n_heads_top10"] for r in out]; own = [r["own_head_share_top10"] for r in out]
    print(f"[circ] functions analysed: {len(out)}; distinct heads among top-10 connections: mean {np.mean(nh):.1f}; "
          f"share of top-10 inside the function's own head: mean {np.mean(own)*100:.0f}%")
    for r in out[:15]:
        print(f"  {r['head']:8s} {r['rule'][:55]:55s} top: " + ", ".join(f"{c['conn']}({c['m1']:+.2f})" for c in r["top_connections"][:4])
              + f" | heads in top10: {r['n_heads_top10']}, own-head share {r['own_head_share_top10']*100:.0f}%")


if __name__ == "__main__":
    main()
