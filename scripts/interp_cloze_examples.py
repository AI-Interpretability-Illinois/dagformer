"""Show what the context-fidelity circuit does, item by item: the top
completions at the blank under gamma = 0 (circuit removed), 1 (trained model)
and 4 (amplified), for a handful of deceptive-frame cloze items."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from interp_common import load_elh, save_json, unflatten_alpha, flatten_alpha
from interp_editing import layer_chunks
from interp_circuit_verify import apply_conn_set
from interp_liar_cloze import build_items, prompt_for


def main():
    out = Path("experiments/results/interp/liar_cloze")
    elh = load_elh()
    cfg = elh.load_config("configs/fourway_300m_dagformer_mmap.yaml")
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    chunks = layer_chunks(L, H)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])
    fourway, predictor = elh.load_fourway(
        "checkpoints/fourway_300m_dagformer_mmap/checkpoint_step9000.pt", cfg, device)

    cj = json.load(open(out / "deception_conns.json"))
    cs = cj["conn_scores"]
    import re
    def parse(k):
        l = int(k[1:].split("/")[0]); h = int(k.split("/h")[1].split("/")[0])
        stream = k.split("/")[2].split("<-")[0]; src = int(k.split("src")[1])
        return l, stream, h, src
    circuit = [parse(k) for k in sorted(cs, key=lambda k: cs[k])[:10]]

    state = {"edits": {}}
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            def hook(m, inp, o):
                e = state["edits"].get(i + 1)
                return apply_conn_set(o, i + 1, H, e) if e else o
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))

    def set_gamma(g):
        state["edits"] = {}
        if g == 1.0: return
        for (l, stream, h, src) in circuit:
            state["edits"].setdefault(l, []).append((stream, h, src, g))

    @torch.no_grad()
    def topk(text, k=5):
        ids = torch.tensor(tok(text, add_special_tokens=False)["input_ids"],
                           dtype=torch.long, device=device).unsqueeze(0)
        routing = predictor(ids)
        if state["edits"]:
            flat = flatten_alpha(routing).float(); parts = []
            for li, (a, b) in enumerate(chunks):
                e = state["edits"].get(li + 1)
                parts.append(apply_conn_set(flat[:, :, a:b], li + 1, H, e) if e else flat[:, :, a:b])
            routing = {s: list(v) for s, v in unflatten_alpha(torch.cat(parts, -1), L, H).items()}
        p = torch.softmax(fourway(ids, routing)[0, -1].float(), -1)
        v, i = p.topk(k)
        return [(tok.decode([int(x)]).strip(), float(y)) for x, y in zip(i, v)]

    items = build_items(240)
    picked = [items[j] for j in (0, 3, 11, 27)]
    dump = []
    for it in picked:
        text = prompt_for(it, "deceptive")
        rec = {"prompt": text, "true": it["true"], "alts": it["alts"], "states": {}}
        print("\n" + "=" * 96)
        print("PROMPT: " + text)
        print(f"context value = '{it['true']}'   distractors = {it['alts']}")
        for g, lab in ((0.0, "circuit OFF  (γ=0)"), (1.0, "trained     (γ=1)"), (4.0, "amplified   (γ=4)")):
            set_gamma(g)
            tk = topk(text)
            rec["states"][str(g)] = tk
            marks = " ".join(f"{w}:{p:.2f}{'*' if w == it['true'] else ''}" for w, p in tk)
            print(f"  {lab}: {marks}")
        set_gamma(1.0)
        dump.append(rec)
    for hk in hooks: hk.remove()
    save_json(dump, out / "examples.json")
    print("\n(* = the value the context established)")
    print("[ex] DONE")


if __name__ == "__main__":
    main()
