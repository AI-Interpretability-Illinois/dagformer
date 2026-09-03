"""Generation demos, round 2.
 (1) Output-amplification lever for class heads: scale the head's attention
     output (o_proj input slice) by kappa and measure class rate in samples.
 (2) Phrase-recurrence demo for the copy circuit: prompts containing a
     distinctive phrase; amplify the 6 named copy heads' routing by gamma and
     count how many times the phrase recurs in the continuation.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from interp_common import load_elh, save_json, unflatten_alpha, flatten_alpha
from interp_editing import layer_chunks
from interp_localize import apply_spec_to_chunk
from interp_discover2 import LEX, norm_tok
from interp_probe import token_charclass

CLASS_HEADS = [
    (9, 3, "lex", "pronoun_3rd", "3rd-person pronoun head"),
    (8, 8, "lex", "pronoun_it_they", "it/they pronoun head"),
    (8, 0, "lex", "open_bracket", "opening-bracket head"),
    (11, 0, "class", 1, "quote/punctuation head"),
    (11, 7, "class", 0, "newline head"),
    (6, 11, "class", 2, "digit head"),
]
COPY_HEADS = [(4, 1), (3, 11), (6, 11), (3, 13), (3, 2), (4, 4)]
PHRASE_PROMPTS = [
    ("The Golden Gate Bridge is a suspension bridge in San Francisco.", "Golden Gate Bridge"),
    ("Professor Ellington explained the theory of quantum crystals to the class.", "quantum crystals"),
    ("The village of Aldermoor sits at the foot of the northern hills.", "Aldermoor"),
    ("Our new product, the SkyLoom 3000, launches next spring.", "SkyLoom 3000"),
]
NEUTRAL = ["The", "Yesterday", "In the morning,", "It was a quiet day in the village.",
           "Here is a short story:", "The report begins as follows.", "According to the article,", "Once upon a time"]


def main():
    out = Path("experiments/results/interp/generation"); out.mkdir(parents=True, exist_ok=True)
    elh = load_elh()
    cfg = elh.load_config("configs/fourway_300m_dagformer_mmap.yaml")
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    hd = cfg["hidden_size"] // H
    chunks = layer_chunks(L, H)
    fourway, predictor = elh.load_fourway("checkpoints/fourway_300m_dagformer_mmap/checkpoint_step9000.pt", cfg, device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])

    state = {"spec": {}, "scale": None}   # scale = (layer, head, kappa)
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            def hook(m, inp, o):
                return apply_spec_to_chunk(o, i + 1, H, state["spec"]) if state["spec"] else o
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))
    for l in range(L):
        def make_pre(l):
            def pre(m, a):
                sc = state["scale"]
                if sc is None or sc[0] != l:
                    return None
                x = a[0].clone()
                x[..., sc[1] * hd:(sc[1] + 1) * hd] *= sc[2]
                return (x,)
            return pre
        hooks.append(fourway.olmo.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_pre(l)))

    @torch.no_grad()
    def logits_for(rows):
        routing = predictor(rows)
        if state["spec"]:
            flat = flatten_alpha(routing).float()
            parts = [apply_spec_to_chunk(flat[:, :, a:b], li + 1, H, state["spec"]) for li, (a, b) in enumerate(chunks)]
            routing = {s: list(v) for s, v in unflatten_alpha(torch.cat(parts, dim=-1), L, H).items()}
        return fourway(rows, routing)[:, -1].float()

    @torch.no_grad()
    def sample(prompt_ids, n, new_tokens=80, temperature=0.9, top_p=0.95):
        rows = prompt_ids.unsqueeze(0).repeat(n, 1).to(device)
        gen = torch.zeros(n, 0, dtype=torch.long, device=device)
        for _ in range(new_tokens):
            lg = logits_for(torch.cat([rows, gen], 1)) / temperature
            probs = torch.softmax(lg, -1)
            sp, si = probs.sort(-1, descending=True)
            cum = sp.cumsum(-1); sp[cum - sp > top_p] = 0; sp = sp / sp.sum(-1, keepdim=True)
            gen = torch.cat([gen, si.gather(1, torch.multinomial(sp, 1))], 1)
        return gen.cpu()

    def in_rule(kind, val, tid):
        s = tok.convert_ids_to_tokens([int(tid)])[0]
        if kind == "lex":
            return norm_tok(s) in LEX[val] or s.replace("Ġ", "") in LEX[val]
        return token_charclass(s) == val

    results = {"output_amplification": {}, "phrase_recurrence": {}}
    for (l, h, kind, val, label) in CLASS_HEADS:
        res = {}
        for kappa in (0.0, 1.0, 2.0, 4.0, 8.0):
            state["spec"] = {}; state["scale"] = None if kappa == 1.0 else (l, h, kappa)
            hits = total = 0; texts = []
            for p in NEUTRAL:
                pid = torch.tensor(tok(p, add_special_tokens=False)["input_ids"], dtype=torch.long)
                for g in sample(pid, 2):
                    ids_ = g.tolist()
                    hits += sum(in_rule(kind, val, t) for t in ids_); total += len(ids_)
                    texts.append(p + tok.decode(ids_))
            res[str(kappa)] = {"class_rate": hits / total, "examples": texts[:2]}
            print(f"[gen2/out] L{l}/h{h} ({label}) kappa={kappa}: class rate {100*hits/total:.1f}% | "
                  + texts[0][:160].replace("\n", "⏎"), flush=True)
        state["scale"] = None
        results["output_amplification"][f"L{l}/h{h}"] = {"label": label, **res}

    def spec_for(gamma):
        spec = {}
        for l, h in COPY_HEADS:
            spec.setdefault(l, {"q": ([], gamma), "k": ([], gamma), "v": ([], gamma)})
            for s in ("q", "k", "v"):
                spec[l][s][0].append(h)
        return spec
    for gamma in (0.0, 1.0, 1.5, 2.0, 3.0, 4.0):
        state["spec"] = {} if gamma == 1.0 else spec_for(gamma); state["scale"] = None
        rec = {}
        for prompt, phrase in PHRASE_PROMPTS:
            pid = torch.tensor(tok(prompt, add_special_tokens=False)["input_ids"], dtype=torch.long)
            counts, texts = [], []
            for g in sample(pid, 4, new_tokens=100):
                txt = tok.decode(g.tolist())
                counts.append(txt.count(phrase)); texts.append(txt)
            rec[phrase] = {"mean_recurrences": sum(counts) / len(counts), "example": texts[0]}
            print(f"[gen2/phrase] gamma={gamma} '{phrase}': recurs {sum(counts)/len(counts):.2f}x | "
                  + texts[0][:160].replace("\n", "⏎"), flush=True)
        results["phrase_recurrence"][str(gamma)] = rec
    state["spec"] = {}
    for hk in hooks:
        hk.remove()
    save_json(results, out / "generation_demos2.json")
    print("[gen2] DONE")


if __name__ == "__main__":
    main()
