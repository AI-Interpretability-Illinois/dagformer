"""Atlas pass 2: relative (z-scored) specialization with copy-first priority,
plus static wiring signatures for ALL heads from the predictor output, and
clustering of wiring signatures into families.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from interp_common import alpha_layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default="experiments/results/interp/atlas_step9000.json")
    ap.add_argument("--dump-dir", default="experiments/results/interp/alpha_dumps")
    ap.add_argument("--out", default="experiments/results/interp")
    ap.add_argument("--fig", default="experiments/figures")
    ap.add_argument("--z-thr", type=float, default=2.5)
    ap.add_argument("--copy-thr", type=float, default=5.0)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()

    atlas = json.load(open(args.atlas))
    base = atlas["baseline"]
    block = atlas["wire"]
    heads = list(block)
    keys = [k for k in base if k != "nll_overall"]
    overall = np.array([block[h]["nll_overall"] - base["nll_overall"] for h in heads])
    E = np.zeros((len(heads), len(keys)))
    for i, h in enumerate(heads):
        for j, k in enumerate(keys):
            if k.startswith("copy_acc"):
                E[i, j] = 100 * (base[k] - block[h][k])
            else:
                E[i, j] = (block[h][k] - base[k]) - overall[i]
    Z = (E - np.median(E, 0)) / (1.4826 * np.median(np.abs(E - np.median(E, 0)), 0) + 1e-9)

    labels = {}
    for i, h in enumerate(heads):
        copy_js = [j for j, k in enumerate(keys) if k.startswith("copy_acc")]
        jc = max(copy_js, key=lambda j: E[i, j])
        if E[i, jc] >= args.copy_thr:
            labels[h] = f"copy (period {keys[jc].split('_p')[1]}, -{E[i, jc]:.0f} pts)"
            continue
        other = [j for j in range(len(keys)) if j not in copy_js]
        jz = max(other, key=lambda j: Z[i, j])
        if Z[i, jz] >= args.z_thr and E[i, jz] > 0:
            labels[h] = f"{keys[jz]} (z={Z[i, jz]:.1f}, +{E[i, jz]:.3f})"
        elif overall[i] >= 0.02:
            labels[h] = f"general (+{overall[i]:.3f})"
        else:
            labels[h] = "no distinct single-head effect"
    n_named = sum(1 for v in labels.values() if not v.startswith("no distinct"))
    print(f"[pass2] heads with a behavioral label at z>={args.z_thr}: {n_named}/{len(heads)}")
    from collections import Counter
    fam = Counter(v.split(" (")[0] for v in labels.values())
    print("[pass2] label families:", dict(fam))

    # ---- wiring signatures for all heads (predictor output, eval corpus) ----
    L = int(max(int(h[1:].split("/")[0]) for h in heads)) + 1
    H = int(max(int(h.split("/h")[1]) for h in heads)) + 1
    a = torch.load(Path(args.dump_dir) / "alpha_eval_step9000.pt", map_location="cpu").float()
    mean_alpha = a.reshape(-1, a.shape[-1]).mean(0).numpy()
    layout = alpha_layout(L, H)
    idx = {}
    for d, lab in enumerate(layout):
        idx[(lab["stream"], lab["layer"], lab["head"], lab["src"])] = d

    def profile(l, h, s):
        return np.array([mean_alpha[idx[(s, l, h, src)]] for src in range(l + 1)])

    # fixed-length descriptor per head: for q,k,v: [w_emb, w_early(mean src1..l-2), w_prev(src l-1), w_last(src l)]
    def desc(l, h):
        out = []
        for s in ("q", "k", "v"):
            p = profile(l, h, s)
            emb = p[0]
            last = p[-1]
            prev = p[-2] if l >= 1 else 0.0
            early = p[1:-2].mean() if l >= 3 else 0.0
            out += [emb, early, prev, last]
        return np.array(out)
    Dm = np.stack([desc(int(h[1:].split("/")[0]), int(h.split("/h")[1])) for h in heads])

    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=args.k, n_init=20, random_state=0).fit(Dm)
    fam_names = []
    for c in range(args.k):
        cen = km.cluster_centers_[c].reshape(3, 4)
        parts = []
        for si, s in enumerate("QKV"):
            emb, early, prev, last = cen[si]
            tag = f"{s}:last{last:+.1f}"
            if emb <= -0.2: tag += f",emb{emb:+.1f}"
            elif emb >= 0.2: tag += f",emb{emb:+.1f}"
            if early <= -0.2 or early >= 0.2: tag += f",early{early:+.1f}"
            parts.append(tag)
        fam_names.append(" | ".join(parts))
    cl = km.labels_
    print("\n[pass2] wiring families (k-means on Q/K/V source profiles):")
    for c in range(args.k):
        members = [heads[i] for i in range(len(heads)) if cl[i] == c]
        lab_counts = Counter(labels[m].split(" (")[0] for m in members)
        print(f"  family {c} (n={len(members)}): {fam_names[c]}")
        print(f"      behaviors: {dict(lab_counts)}")

    cards = {h: {"d_overall": float(overall[i]), "label": labels[h],
                 "wiring_family": int(cl[i]), "wiring_family_desc": fam_names[cl[i]],
                 "wiring_q": profile(int(h[1:].split('/')[0]), int(h.split('/h')[1]), "q").round(2).tolist(),
                 "wiring_k": profile(int(h[1:].split('/')[0]), int(h.split('/h')[1]), "k").round(2).tolist(),
                 "wiring_v": profile(int(h[1:].split('/')[0]), int(h.split('/h')[1]), "v").round(2).tolist()}
             for i, h in enumerate(heads)}
    json.dump({"labels": labels, "families": fam_names, "cards": cards},
              open(Path(args.out) / "atlas_cards.json", "w"), indent=1)

    named = [(h, labels[h], overall[i]) for i, h in enumerate(heads) if not labels[h].startswith("no distinct")]
    named.sort(key=lambda x: -x[2])
    print("\n[pass2] named heads (behavioral):")
    for h, lab, o in named:
        print(f"  {h:8s} dNLL={o:+.4f}  {lab}  | family {cl[heads.index(h)]}")


if __name__ == "__main__":
    main()
