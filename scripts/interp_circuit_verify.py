"""Circuit-level verification and steering.

For selected functions, take the top-k specific connections (a circuit that
may span several heads/layers) and:
  ablate   replace each connection by its layer mean (k = 3, 10) — measure the
           function's on-rule effect vs an equal-size random-connection control
  amplify  scale each connection's deviation from the layer mean by gamma
           (2, 4) — measure on-rule NLL change and the function's class rate in
           sampled text (Golden-Gate-style test at circuit level)
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interp_common import load_elh, save_json, unflatten_alpha, flatten_alpha
from interp_editing import layer_chunks, stream_slices
from interp_discover3 import LEX, norm_tok
from interp_probe import token_charclass


def apply_conn_set(chunk, l, H, edits):
    """edits: list of (stream, head, src, gamma) for layer l; gamma=0 -> layer mean."""
    out = chunk.clone()
    for stream, h, src, gamma in edits:
        sl, n = stream_slices(l, H)
        s0, s1 = sl[stream]
        seg = out[:, :, s0:s1].view(*out.shape[:2], H, n)
        m = seg[:, :, :, src].mean(dim=2)
        seg[:, :, h, src] = m + gamma * (seg[:, :, h, src] - m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--functions", default="pronoun 3rd,open bracket,close bracket,digit,whitespace,continues a word,quote,sentence end")
    ap.add_argument("--out", default="experiments/results/interp/token_effects")
    args = ap.parse_args()
    d = Path(args.out)
    elh = load_elh()
    cfg = elh.load_config("configs/fourway_300m_dagformer_mmap.yaml")
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    chunks = layer_chunks(L, H)
    fourway, predictor = elh.load_fourway("checkpoints/fourway_300m_dagformer_mmap/checkpoint_step9000.pt", cfg, device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])
    corpus = torch.load(d / "corpus.pt", map_location="cpu", weights_only=False)
    ids, labels = corpus["ids"], corpus["labels"]
    N, T = ids.shape
    ver_ids, ver_lab = ids[N // 2:][:200], labels[N // 2:][:200]
    base = torch.load(d / "baseline_nll.pt", map_location="cpu").float()[N // 2:][:200]
    circuits = json.load(open(d / "circuits_specific.json"))

    state = {"edits": {}}   # layer -> list of (stream, head, src, gamma)
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            def hook(m, inp, o):
                e = state["edits"].get(i + 1)
                return apply_conn_set(o, i + 1, H, e) if e else o
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))

    @torch.no_grad()
    def run(rows):
        routing = predictor(rows.to(device))
        if state["edits"]:
            flat = flatten_alpha(routing).float()
            parts = []
            for li, (a, b) in enumerate(chunks):
                e = state["edits"].get(li + 1)
                parts.append(apply_conn_set(flat[:, :, a:b], li + 1, H, e) if e else flat[:, :, a:b])
            routing = {s: [t.to(device) for t in v] for s, v in unflatten_alpha(torch.cat(parts, -1), L, H).items()}
        return fourway(rows.to(device), routing)

    @torch.no_grad()
    def per_token_nll(rows, labs, batch=16):
        outs = []
        for i in range(0, rows.shape[0], batch):
            lg = run(rows[i:i + batch])
            outs.append(F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), labs[i:i + batch].to(device).reshape(-1),
                                        reduction="none").view(-1, T).cpu())
        return torch.cat(outs)

    uniq = np.unique(np.concatenate([ver_lab.numpy().reshape(-1), ver_ids.numpy().reshape(-1)]))
    raw = dict(zip(uniq.tolist(), tok.convert_ids_to_tokens(uniq.tolist())))
    lab_np = ver_lab.numpy()
    def mask_for(rule):
        if "continues a word" in rule:
            return np.vectorize(lambda i: (not raw[i].startswith("Ġ")) and raw[i].isalpha())(lab_np)
        for name in LEX:
            if rule == f"target is a {name.replace('_', ' ')} token":
                ids_set = {i for i, s2 in raw.items() if norm_tok(s2) in LEX[name] or s2.replace("Ġ", "") in LEX[name]}
                return np.isin(lab_np, list(ids_set))
        for c, cn in enumerate(["whitespace", "punctuation", "digit", "capitalized", "lowercase", "mixed"]):
            if rule == f"target char-class = {cn}":
                return np.vectorize(lambda i: token_charclass(raw[i]) == c)(lab_np)
        return None

    def set_edits(conn_list, gamma):
        state["edits"] = {}
        for t in conn_list:
            state["edits"].setdefault(t["layer"], []).append((t["stream"], t["head"], t["src"], gamma))

    all_conns = [(int(c["layer"]), c["stream"], int(c["head"]), int(c["src"])) for r in circuits for c in r["top_connections"]]
    rng = random.Random(0)
    results = {}
    wanted = [w.strip() for w in args.functions.split(",")]
    for r in circuits:
        if not any(w in r["rule"] for w in wanted): continue
        m = mask_for(r["rule"])
        if m is None or m.sum() < 100: continue
        m_t = torch.from_numpy(m)
        rec = {"rule": r["rule"], "n_rule_tokens": int(m.sum())}
        for k in (3, 10):
            circ = r["top_connections"][:k]
            set_edits(circ, 0.0)
            nll = per_token_nll(ver_ids, ver_lab)
            d_on, d_off = float((nll - base)[m_t].mean()), float((nll - base)[~m_t].mean())
            rnd = rng.sample(all_conns, k)
            set_edits([{"layer": l, "stream": s, "head": h, "src": sr} for l, s, h, sr in rnd], 0.0)
            nll_r = per_token_nll(ver_ids, ver_lab)
            rec[f"ablate_top{k}"] = {"on_rule": d_on, "off_rule": d_off,
                                     "random_k_on_rule": float((nll_r - base)[m_t].mean()),
                                     "random_k_off_rule": float((nll_r - base)[~m_t].mean())}
            print(f"[verify] {r['rule'][:45]:45s} ablate top{k}: on {d_on:+.4f} off {d_off:+.4f} | random{k}: on {rec[f'ablate_top{k}']['random_k_on_rule']:+.4f}", flush=True)
        for gamma in (2.0, 4.0):
            set_edits(r["top_connections"][:10], gamma)
            nll = per_token_nll(ver_ids, ver_lab)
            rec[f"amplify_top10_x{gamma}"] = {"on_rule": float((nll - base)[m_t].mean()), "off_rule": float((nll - base)[~m_t].mean())}
            print(f"[verify] {r['rule'][:45]:45s} amplify x{gamma}: on {rec[f'amplify_top10_x{gamma}']['on_rule']:+.4f} off {rec[f'amplify_top10_x{gamma}']['off_rule']:+.4f}", flush=True)
        state["edits"] = {}
        results[r["rule"]] = rec
    for hk in hooks:
        hk.remove()
    save_json(results, d / "circuit_verify.json")
    print("[verify] DONE")


if __name__ == "__main__":
    main()
