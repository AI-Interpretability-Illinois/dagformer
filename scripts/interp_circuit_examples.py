"""For each verified circuit (function), compute three-state evidence:
per-token NLL on 200 held-out windows under gamma = 0 (circuit connections
set to layer mean), 1 (unchanged), 2 (deviation doubled); pick the 3
rule-matching positions where suppression hurts most; record context text,
target token and P(target) under the three states, plus aggregate deltas.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interp_common import load_elh, save_json, unflatten_alpha, flatten_alpha
from interp_editing import layer_chunks
from interp_circuit_verify import apply_conn_set
from interp_discover3 import LEX, norm_tok
from interp_probe import token_charclass

WANTED = ["target is a open bracket token", "target is a close bracket token", "target is a quote token",
          "target is a pronoun 3rd token", "target is a sentence end token", "target char-class = whitespace"]


def main():
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
    circuits = {r["rule"]: r for r in json.load(open(d / "circuits_specific.json"))}

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

    lab_np = ver_lab.numpy()
    uniq = np.unique(np.concatenate([lab_np.reshape(-1), ver_ids.numpy().reshape(-1)]))
    raw = dict(zip(uniq.tolist(), tok.convert_ids_to_tokens(uniq.tolist())))
    def mask_for(rule):
        for name in LEX:
            if rule == f"target is a {name.replace('_', ' ')} token":
                ids_set = {i for i, s2 in raw.items() if norm_tok(s2) in LEX[name] or s2.replace("Ġ", "") in LEX[name]}
                return np.isin(lab_np, list(ids_set))
        for c, cn in enumerate(["whitespace", "punctuation", "digit", "capitalized", "lowercase", "mixed"]):
            if rule == f"target char-class = {cn}":
                return np.vectorize(lambda i: token_charclass(raw[i]) == c)(lab_np)
        raise ValueError(rule)

    def set_edits(conns, gamma):
        state["edits"] = {}
        for c in conns:
            state["edits"].setdefault(c["layer"], []).append((c["stream"], c["head"], c["src"], gamma))

    results = {}
    for rule in WANTED:
        r = circuits[rule]; conns = r["top_connections"][:10]
        m = torch.from_numpy(mask_for(rule))
        nll = {}
        for g in (0.0, 1.0, 2.0):
            set_edits(conns, g) if g != 1.0 else state.__setitem__("edits", {})
            nll[g] = per_token_nll()
        state["edits"] = {}
        d0 = nll[0.0] - nll[1.0]; d2 = nll[2.0] - nll[1.0]
        agg = {"suppress_on": float(d0[m].mean()), "suppress_off": float(d0[~m].mean()),
               "amplify_on": float(d2[m].mean()), "amplify_off": float(d2[~m].mean()), "n_rule_tokens": int(m.sum())}
        # example positions: largest suppression damage among rule tokens with baseline P >= 0.2
        cand = (d0 * m.float()) * (nll[1.0] < 1.6).float()
        flat = cand.reshape(-1); top = torch.argsort(-flat)[:40].tolist()
        examples, seen_ctx = [], set()
        for idx in top:
            s, t = divmod(idx, T)
            ctx = tok.decode(ver_ids[s, max(0, t - 16):t + 1].tolist()).replace("\n", "⏎")
            if ctx[-40:] in seen_ctx: continue
            seen_ctx.add(ctx[-40:])
            examples.append({"context": ctx[-110:], "target": raw[int(lab_np[s, t])].replace("Ġ", "␣").replace("Ċ", "⏎"),
                             "p_suppressed": float(torch.exp(-nll[0.0][s, t])), "p_unchanged": float(torch.exp(-nll[1.0][s, t])),
                             "p_activated": float(torch.exp(-nll[2.0][s, t]))})
            if len(examples) == 3: break
        results[rule] = {"connections": conns, "aggregate": agg, "examples": examples}
        print(f"[ex] {rule}: {agg}")
        for e in examples:
            print(f"     {e['context'][-70:]} >>{e['target']}<<  P: {e['p_suppressed']:.2f} / {e['p_unchanged']:.2f} / {e['p_activated']:.2f}")
    save_json(results, d / "circuit_examples.json")
    for hk in hooks: hk.remove()
    print("[ex] DONE")


if __name__ == "__main__":
    main()
