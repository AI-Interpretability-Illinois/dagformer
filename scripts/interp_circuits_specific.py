"""Specificity-normalized circuit attribution.

raw score  s[c,f] = m1[c,f] - m0[c,f]  (connection c's mean effect on function
f's tokens minus on other tokens). Large-effect connections (the copy circuit)
dominate raw rankings for any copy-prone token class. Normalize per connection
across functions:  z[c,f] = (s[c,f] - mean_f s[c,.]) / std_f s[c,.]  — how
unusually strongly connection c responds to function f. Functions are
deduplicated by rule spec. Output circuits_specific.json with, per unique
function, the top connections by z (requiring s > 0), the heads/layers they
span, and a compact circuit description.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    d = Path("experiments/results/interp/token_effects")
    z = np.load(d / "circuit_matrix.npz", allow_pickle=True)
    m1, m0 = z["m1"], z["m0"]
    conns = [tuple(c) for c in z["conns"]]
    rules, specs = list(z["rules"]), list(z["specs"])
    s = m1 - m0                                            # [n_conn, n_func]
    # dedupe functions by spec
    seen, keep = {}, []
    for j, sp in enumerate(specs):
        if sp not in seen:
            seen[sp] = j; keep.append(j)
    s = s[:, keep]; m1 = m1[:, keep]; rules = [rules[j] for j in keep]
    zs = (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)
    out = []
    for j, rule in enumerate(rules):
        order = [i for i in np.argsort(-zs[:, j]) if s[i, j] > 0][:12]
        if not order: continue
        top = [{"conn": f"L{conns[i][0]}/h{conns[i][2]}/{conns[i][1]}<-src{conns[i][3]}",
                "layer": int(conns[i][0]), "head": int(conns[i][2]), "stream": conns[i][1], "src": int(conns[i][3]),
                "z": float(zs[i, j]), "effect_on_rule": float(m1[i, j]), "effect_off_rule": float(m0[i, keep[j]])} for i in order]
        heads = Counter(f"L{t['layer']}/h{t['head']}" for t in top[:10])
        streams = Counter(t["stream"] for t in top[:10])
        srcs = Counter(t["src"] for t in top[:10])
        out.append({"rule": rule, "top_connections": top, "n_heads_top10": len(heads),
                    "n_layers_top10": len({t['layer'] for t in top[:10]}),
                    "stream_mix": dict(streams), "src_mix": {str(k): v for k, v in srcs.items()},
                    "max_z": float(zs[order[0], j])})
    out.sort(key=lambda r: -r["max_z"])
    json.dump(out, open(d / "circuits_specific.json", "w"), indent=1)
    print(f"[circ2] {len(out)} unique functions; heads among top-10 specific connections: mean {np.mean([r['n_heads_top10'] for r in out]):.1f}")
    for r in out[:25]:
        print(f"  z={r['max_z']:.1f} {r['rule'][:52]:52s} heads={r['n_heads_top10']} streams={r['stream_mix']} | "
              + ", ".join(f"{t['conn']}({t['effect_on_rule']:+.2f})" for t in r["top_connections"][:4]))


if __name__ == "__main__":
    main()
