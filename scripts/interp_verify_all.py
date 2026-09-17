"""Verify circuits discovered over ALL connections (circuits_all.json).

For each selected rule: ablate its top-k connection set (k=10) vs a
size-matched random set drawn from all scanned connections; amplify the set
(gamma 2, 4). Reports on-rule / off-rule NLL change on 200 held-out windows.
Selection favours rules whose circuit draws mostly on heads that had NO
single-head name (unnamed_head_share_top10 >= 0.5) — the trace-only
candidates — plus the strongest overall circuits for reference.
"""
from __future__ import annotations

import argparse, json, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interp_common import load_elh, save_json, unflatten_alpha, flatten_alpha
from interp_editing import layer_chunks
from interp_circuit_verify import apply_conn_set
from interp_discover3 import LEX, norm_tok
from interp_probe import token_charclass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trace-only", type=int, default=8)
    ap.add_argument("--n-strongest", type=int, default=6)
    args = ap.parse_args()
    d = Path("experiments/results/interp/token_effects")
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
    base_all = torch.load(d / "baseline_nll.pt", map_location="cpu").float()[N // 2:][:200]
    circuits = json.load(open(d / "circuits_all.json"))

    state = {"edits": {}}
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            def hook(m, inp, o):
                e = state["edits"].get(i + 1)
                return apply_conn_set(o, i + 1, H, e) if e else o
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))

    @torch.no_grad()
    def per_token_nll(batch=16):
        outs = []
        for i in range(0, ver_ids.shape[0], batch):
            rows = ver_ids[i:i + batch].to(device)
            routing = predictor(rows)
            if state["edits"]:
                flat = flatten_alpha(routing).float()
                parts = []
                for li, (a, b) in enumerate(chunks):
                    e = state["edits"].get(li + 1)
                    parts.append(apply_conn_set(flat[:, :, a:b], li + 1, H, e) if e else flat[:, :, a:b])
                routing = {s: list(v) for s, v in unflatten_alpha(torch.cat(parts, -1), L, H).items()}
            lg = fourway(rows, routing)
            outs.append(F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), ver_lab[i:i + batch].to(device).reshape(-1), reduction="none").view(-1, T).cpu())
        return torch.cat(outs)

    lab_np = ver_lab.numpy(); id_np = ver_ids.numpy()
    uniq = np.unique(np.concatenate([lab_np.reshape(-1), id_np.reshape(-1)]))
    raw = dict(zip(uniq.tolist(), tok.convert_ids_to_tokens(uniq.tolist())))
    normed = {i: norm_tok(s) for i, s in raw.items()}
    CN = ["whitespace", "punctuation", "digit", "capitalized", "lowercase", "mixed"]
    def mask_for(rule, tokset=None):
        if tokset:
            return np.isin(lab_np, np.array(tokset))
        for n in LEX:
            if rule == f"target is a {n.replace('_', ' ')} token":
                return np.isin(lab_np, list({i for i, s in normed.items() if s in LEX[n] or raw[i].replace("Ġ", "") in LEX[n]}))
        for c, cn in enumerate(CN):
            if rule == f"target char-class = {cn}":
                return np.vectorize(lambda i: token_charclass(raw[i]) == c)(lab_np)
            if rule == f"preceding token char-class = {cn}":
                return np.vectorize(lambda i: token_charclass(raw[i]) == c)(id_np)
        if rule == "target continues a word (no leading space)":
            return np.vectorize(lambda i: (not raw[i].startswith("Ġ")) and raw[i].isalpha())(lab_np)
        if rule == "target starts a new word":
            return np.vectorize(lambda i: raw[i].startswith("Ġ") and raw[i][1:].isalpha())(lab_np)
        if rule == "target follows a newline":
            return np.isin(id_np, np.array(sorted({i for i, s in raw.items() if "Ċ" in s})))
        if rule == "target equals the preceding token":
            return lab_np == id_np
        if rule == "target occurred in the previous 8 tokens":
            m = np.zeros_like(lab_np, dtype=bool)
            for k in range(1, 9): m[:, k:] |= lab_np[:, k:] == id_np[:, :-k]
            return m
        if rule == "target occurred in the previous 128 tokens":
            m = np.zeros_like(lab_np, dtype=bool)
            for s in range(lab_np.shape[0]):
                for t in range(T): m[s, t] = lab_np[s, t] in id_np[s, max(0, t - 127):t + 1]
            return m
        if rule == "digit following a digit":
            return np.vectorize(lambda i: token_charclass(raw[i]) == 2)(id_np) & np.vectorize(lambda i: token_charclass(raw[i]) == 2)(lab_np)
        if rule == "position < 64":
            return np.tile(np.arange(T), (lab_np.shape[0], 1)) < 64
        if rule.startswith("["):     # data-driven token set carries no ids here
            return None
        return None

    def set_edits(conns, gamma):
        state["edits"] = {}
        for c in conns:
            state["edits"].setdefault(c["layer"], []).append((c["stream"], c["head"], c["src"], gamma))

    all_conns = [(c["layer"], c["stream"], c["head"], c["src"]) for r in circuits for c in r["top_connections"]]
    rng = random.Random(0)
    trace_only = [r for r in circuits if r["unnamed_head_share_top10"] >= 0.5][:args.n_trace_only]
    rest = [r for r in circuits if r not in trace_only]
    strongest = rest[:args.n_strongest]
    mid = [r for r in rest if 0.2 <= r["unnamed_head_share_top10"] < 0.5][:args.n_strongest]
    state["edits"] = {}; base = per_token_nll()
    results = {}
    for tag, group in (("trace_only", trace_only), ("mixed", mid), ("strongest", strongest)):
        for r in group:
            m = mask_for(r["rule"], r.get("tokset"))
            if m is None or m.sum() < 100:
                print(f"[vall] skip {r['rule'][:50]} (no mask)"); continue
            mt = torch.from_numpy(m)
            circ = r["top_connections"][:10]
            set_edits(circ, 0.0); nll = per_token_nll()
            on, off = float((nll - base)[mt].mean()), float((nll - base)[~mt].mean())
            rnd = rng.sample(all_conns, 10)
            set_edits([{"layer": l, "stream": s, "head": h, "src": sr} for l, s, h, sr in rnd], 0.0)
            nr = per_token_nll(); ron = float((nr - base)[mt].mean())
            amp = {}
            for g in (2.0, 4.0):
                set_edits(circ, g); na = per_token_nll()
                amp[str(g)] = {"on": float((na - base)[mt].mean()), "off": float((na - base)[~mt].mean())}
            state["edits"] = {}
            results[r["rule"]] = {"group": tag, "unnamed_share": r["unnamed_head_share_top10"], "n_heads": r["n_heads_top10"],
                                  "heads": sorted({f"L{c['layer']}/h{c['head']}" for c in circ}),
                                  "ablate": {"on": on, "off": off, "random10_on": ron}, "amplify": amp}
            print(f"[vall/{tag}] {r['rule'][:44]:44s} unnamed {100*r['unnamed_head_share_top10']:.0f}% | ablate on {on:+.3f} off {off:+.3f} rnd {ron:+.3f} | amp2 {amp['2.0']['on']:+.3f} amp4 {amp['4.0']['on']:+.3f}", flush=True)
    save_json(results, d / "verify_all.json")
    for hk in hooks: hk.remove()
    print("[vall] DONE")


if __name__ == "__main__":
    main()
