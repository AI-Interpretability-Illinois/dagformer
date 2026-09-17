"""Free-generation three-state test at CIRCUIT level: for each verified
function's top-10 connection set, sample text with gamma in {0,1,2,4} and
measure visible statistics (function-class rate, mean sentence length,
newline rate, distinct-token ratio); store examples.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from interp_common import load_elh, save_json, unflatten_alpha, flatten_alpha
from interp_editing import layer_chunks
from interp_circuit_verify import apply_conn_set
from interp_discover3 import LEX, norm_tok
from interp_probe import token_charclass

RULES = ["target char-class = whitespace", "target is a quote token", "target is a preposition token",
         "target is a pronoun 3rd token", "target is a close bracket token"]
PROMPTS = ["The", "Yesterday", "In the morning,", "It was a quiet day in the village.", "Here is a short story:",
           "The report begins as follows.", "According to the article,", "Once upon a time", "She looked at him and",
           "When they arrived at the house,", "He opened the letter and read:", "In 1895, the town"]


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
    def logits_last(rows):
        routing = predictor(rows)
        if state["edits"]:
            flat = flatten_alpha(routing).float()
            parts = []
            for li, (a, b) in enumerate(chunks):
                e = state["edits"].get(li + 1)
                parts.append(apply_conn_set(flat[:, :, a:b], li + 1, H, e) if e else flat[:, :, a:b])
            routing = {s: list(v) for s, v in unflatten_alpha(torch.cat(parts, -1), L, H).items()}
        return fourway(rows, routing)[:, -1].float()

    @torch.no_grad()
    def sample(prompt_ids, n, new_tokens=80):
        rows = prompt_ids.unsqueeze(0).repeat(n, 1).to(device); gen = torch.zeros(n, 0, dtype=torch.long, device=device)
        for _ in range(new_tokens):
            lg = logits_last(torch.cat([rows, gen], 1)) / 0.9
            p = torch.softmax(lg, -1); sp, si = p.sort(-1, descending=True); cum = sp.cumsum(-1); sp[cum - sp > 0.95] = 0; sp /= sp.sum(-1, keepdim=True)
            gen = torch.cat([gen, si.gather(1, torch.multinomial(sp, 1))], 1)
        return gen.cpu()

    cache = {}
    def tstr(t):
        if t not in cache: cache[t] = tok.convert_ids_to_tokens([int(t)])[0]
        return cache[t]
    def matches(rule, t):
        s = tstr(t)
        for name in LEX:
            if rule == f"target is a {name.replace('_', ' ')} token":
                return norm_tok(s) in LEX[name] or s.replace("Ġ", "") in LEX[name]
        if rule == "target continues a word (no leading space)":
            return (not s.startswith("Ġ")) and s.isalpha()
        for c, cn in enumerate(["whitespace", "punctuation", "digit", "capitalized", "lowercase", "mixed"]):
            if rule == f"target char-class = {cn}": return token_charclass(s) == c
        return False

    def set_edits(conns, gamma):
        state["edits"] = {}
        for c in conns:
            state["edits"].setdefault(c["layer"], []).append((c["stream"], c["head"], c["src"], gamma))

    results = {}
    for rule in RULES:
        if rule not in circuits: print("[cgen] missing circuit for", rule); continue
        conns = circuits[rule]["top_connections"][:10]
        res = {"heads": sorted({f"L{c['layer']}/h{c['head']}" for c in conns})}
        for g in (0.0, 1.0, 2.0, 4.0):
            set_edits(conns, g) if g != 1.0 else state.__setitem__("edits", {})
            hits = tot = 0; sent_lens = []; nl = 0; distinct = []; texts = []
            for p in PROMPTS:
                pid = torch.tensor(tok(p, add_special_tokens=False)["input_ids"], dtype=torch.long)
                for gsample in sample(pid, 8):
                    toks = gsample.tolist()
                    hits += sum(matches(rule, t) for t in toks); tot += len(toks)
                    nl += sum("Ċ" in tstr(t) for t in toks)
                    distinct.append(len(set(toks)) / len(toks))
                    txt = tok.decode(toks)
                    sents = [s for s in re.split(r"[.!?]+", txt) if s.strip()]
                    sent_lens += [len(s.split()) for s in sents]
                    texts.append(p + txt)
            res[str(g)] = {"class_rate": hits / tot, "newline_rate": nl / tot, "mean_sentence_len": float(np.mean(sent_lens)) if sent_lens else 0.0,
                           "distinct_ratio": float(np.mean(distinct)), "examples": texts}
            print(f"[cgen] {rule[:34]:34s} g={g}: class {100*hits/tot:.2f}% newline {100*nl/tot:.2f}% sentlen {res[str(g)]['mean_sentence_len']:.1f} distinct {res[str(g)]['distinct_ratio']:.2f} | {texts[0][:80].replace(chr(10), '⏎')}", flush=True)
        state["edits"] = {}
        results[rule] = res
    save_json(results, d / "circuit_generation_full.json")
    for hk in hooks: hk.remove()
    print("[cgen] DONE")


if __name__ == "__main__":
    main()
