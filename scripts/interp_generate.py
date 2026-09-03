"""Golden-Gate-style demonstration: amplify one named head's routing and
sample text. For each (head, gamma) we sample continuations of neutral
prompts with the head's Q/K/V routing deviation from the layer mean scaled by
gamma (applied to predictor + corrections), and measure how often the head's
target class appears in the generated tokens.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from interp_common import load_elh, save_json, unflatten_alpha, flatten_alpha
from interp_editing import layer_chunks
from interp_localize import apply_spec_to_chunk
from interp_discover2 import LEX, norm_tok
from interp_probe import token_charclass

DEMOS = [  # (layer, head, rule_kind, rule_value, label)
    (9, 3, "lex", "pronoun_3rd", "3rd-person pronoun head (he/she/her)"),
    (8, 8, "lex", "pronoun_it_they", "it/they/them pronoun head"),
    (9, 11, "lex", "pronoun_1st", "1st-person pronoun head (I/me/my/we)"),
    (8, 0, "lex", "open_bracket", "opening-bracket head"),
    (8, 9, "lex", "close_bracket", "closing-bracket head"),
    (11, 0, "class", 1, "quote/punctuation head"),
    (11, 7, "class", 0, "newline/whitespace head"),
    (6, 11, "class", 2, "digit head"),
    (10, 10, "lex", "preposition", "preposition head"),
    (4, 1, "repeat", None, "copy head (repeats earlier tokens)"),
]
PROMPTS = [
    "The", "Yesterday", "In the morning,", "It was a quiet day in the village.",
    "Here is a short story:", "The report begins as follows.",
    "According to the article,", "Once upon a time",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/fourway_300m_dagformer_mmap.yaml")
    ap.add_argument("--ckpt", default="checkpoints/fourway_300m_dagformer_mmap/checkpoint_step9000.pt")
    ap.add_argument("--out", default="experiments/results/interp/generation")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0, 1, 2, 4, 8])
    ap.add_argument("--new-tokens", type=int, default=80)
    ap.add_argument("--samples-per-prompt", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    elh = load_elh()
    cfg = elh.load_config(args.config)
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    chunks = layer_chunks(L, H)
    fourway, predictor = elh.load_fourway(args.ckpt, cfg, device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])

    state = {"spec": {}}
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            def hook(m, inp, o):
                return apply_spec_to_chunk(o, i + 1, H, state["spec"]) if state["spec"] else o
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))

    @torch.no_grad()
    def logits_for(rows):
        routing = predictor(rows)
        if state["spec"]:
            flat = flatten_alpha(routing).float()
            parts = [apply_spec_to_chunk(flat[:, :, a:b], li + 1, H, state["spec"])
                     for li, (a, b) in enumerate(chunks)]
            routing = {s: [t for t in v] for s, v in unflatten_alpha(torch.cat(parts, dim=-1), L, H).items()}
        return fourway(rows, routing)[:, -1].float()

    @torch.no_grad()
    def sample(prompt_ids, n):
        rows = prompt_ids.unsqueeze(0).repeat(n, 1).to(device)
        gen = torch.zeros(n, 0, dtype=torch.long, device=device)
        for _ in range(args.new_tokens):
            lg = logits_for(torch.cat([rows, gen], 1)) / args.temperature
            probs = torch.softmax(lg, -1)
            sp, si = probs.sort(-1, descending=True)
            cum = sp.cumsum(-1)
            sp[cum - sp > args.top_p] = 0
            sp = sp / sp.sum(-1, keepdim=True)
            nxt = si.gather(1, torch.multinomial(sp, 1))
            gen = torch.cat([gen, nxt], 1)
        return gen.cpu()

    lex_ids = {}
    def in_rule(kind, val, tid, prev_gen):
        s = tok.convert_ids_to_tokens([int(tid)])[0]
        if kind == "lex":
            return norm_tok(s) in LEX[val] or s.replace("Ġ", "") in LEX[val]
        if kind == "class":
            return token_charclass(s) == val
        if kind == "repeat":
            return int(tid) in prev_gen
        raise ValueError(kind)

    results = {}
    for (l, h, kind, val, label) in DEMOS:
        results[f"L{l}/h{h}"] = {"label": label, "gammas": {}}
        for gamma in args.gammas:
            state["spec"] = {} if gamma == 1 else {l: {"q": ((h,), gamma), "k": ((h,), gamma), "v": ((h,), gamma)}}
            hits, total, texts, distinct = 0, 0, [], []
            for p in PROMPTS:
                pid = torch.tensor(tok(p, add_special_tokens=False)["input_ids"], dtype=torch.long)
                gen = sample(pid, args.samples_per_prompt)
                for g in gen:
                    ids_ = g.tolist()
                    for j, t in enumerate(ids_):
                        hits += int(in_rule(kind, val, t, set(ids_[:j]) if kind == "repeat" else None))
                        total += 1
                    distinct.append(len(set(ids_)) / len(ids_))
                    texts.append(p + tok.decode(ids_))
            rate = hits / total
            results[f"L{l}/h{h}"]["gammas"][str(gamma)] = {
                "class_rate": rate, "distinct_ratio": sum(distinct) / len(distinct),
                "examples": texts[:3]}
            print(f"[gen] L{l}/h{h} ({label}) gamma={gamma}: class rate {rate*100:.1f}%  "
                  f"distinct {sum(distinct)/len(distinct):.2f}", flush=True)
            print("      e.g.: " + texts[0][:200].replace("\n", "⏎"), flush=True)
        state["spec"] = {}
    for hk in hooks:
        hk.remove()
    save_json(results, out / "generation_demos.json")
    print("[gen] DONE")


if __name__ == "__main__":
    main()
