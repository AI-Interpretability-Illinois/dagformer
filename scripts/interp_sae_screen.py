"""Screen all auto-named SAE features for bidirectional controllability on
the NLL readout (alpha = -4, +4), then run large-sample generation for the
top bidirectional features with class-appropriate prompts.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interp_common import load_elh, save_json
from interp_editing import layer_chunks
from interp_routing_sae import SAE
from interp_token_effects import build_corpus
from interp_discover3 import LEX, norm_tok
from interp_probe import token_charclass

GEN_PROMPTS = ["The", "Yesterday", "In the morning,", "It was a quiet day in the village.", "Here is a short story:",
               "The report begins as follows.", "According to the article,", "Once upon a time", "She looked at him and",
               "When they arrived at the house,", "He opened the letter and read:", "In 1895, the town"]


def main():
    out = Path("experiments/results/interp/routing_sae")
    elh = load_elh()
    cfg = elh.load_config("configs/fourway_300m_dagformer_mmap.yaml")
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]; T = cfg["seq_len"]
    chunks = layer_chunks(L, H); D = chunks[-1][1]
    fourway, predictor = elh.load_fourway("checkpoints/fourway_300m_dagformer_mmap/checkpoint_step9000.pt", cfg, device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])
    ck = torch.load(out / "sae.pt", map_location="cpu")
    sae = SAE(D, ck["sae"]["enc.weight"].shape[0], 32).to(device); sae.load_state_dict(ck["sae"]); sae.eval()
    sd = ck["sd"].to(device)
    feats = json.load(open(out / "features.json"))["features"]

    state = {"delta": None}
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            a, b = chunks[i]
            def hook(m, inp, o):
                dl = state["delta"]
                return o if dl is None else o + dl[a:b].to(o.device, o.dtype)
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))

    ids_nll, lab_nll, _ = build_corpus("/work/hdd/bfqt/data/pretok/dolma_v1_7_12b", T, 100, 42, 1024, 5500000, offset_seq=0)
    lab_np = lab_nll.numpy()
    uniq = np.unique(lab_np)
    raw = dict(zip(uniq.tolist(), tok.convert_ids_to_tokens(uniq.tolist())))
    lex_ids = {n: {i for i, s in raw.items() if norm_tok(s) in w or s.replace("Ġ", "") in w} for n, w in LEX.items()}
    def rule_mask(rule):
        kind, val = rule
        if kind == "lex": return np.isin(lab_np, list(lex_ids[val]))
        if kind == "class": return np.vectorize(lambda i: token_charclass(raw[i]) == val)(lab_np)
        return np.isin(lab_np, list(val))

    @torch.no_grad()
    def per_token_nll():
        outs = []
        for i in range(0, ids_nll.shape[0], 16):
            rows = ids_nll[i:i + 16]
            lg = fourway(rows.to(device), predictor(rows.to(device)))
            outs.append(F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), lab_nll[i:i + 16].to(device).reshape(-1), reduction="none").view(rows.shape[0], T).cpu())
        return torch.cat(outs)
    state["delta"] = None
    base = per_token_nll()

    screen = []
    for r in feats:
        m = torch.tensor(rule_mask(r["rule"]))
        if m.sum() < 50: continue
        d_f = (sae.dec.weight[:, r["feature"]] * sd).detach()
        rec = {"feature": r["feature"], "rule": r["rule"], "freq": r["freq"], "n_rule": int(m.sum())}
        for a in (-4.0, 4.0):
            state["delta"] = a * r["mean_act"] * d_f
            dn = per_token_nll() - base
            rec[f"on{a:+.0f}"] = float(dn[m].mean()); rec[f"off{a:+.0f}"] = float(dn[~m].mean())
        state["delta"] = None
        # bidirectional score: opposite signs on-rule, both large relative to off-rule
        s1, s2 = rec["on-4"], rec["on+4"]
        rec["bidir"] = float(abs(s1 - s2)) if s1 * s2 < 0 else 0.0
        rec["collateral"] = float(max(abs(rec["off-4"]), abs(rec["off+4"])))
        screen.append(rec)
        print(f"[screen] f{r['feature']:5d} {str(r['rule'])[:36]:36s} on-4 {s1:+.3f} on+4 {s2:+.3f} off {rec['collateral']:.3f} bidir {rec['bidir']:.3f}", flush=True)
    screen.sort(key=lambda r: -(r["bidir"] - 2 * r["collateral"]))
    save_json(screen, out / "screen.json")
    top = [r for r in screen if r["bidir"] > 0.04 and r["collateral"] < 0.03][:8]
    print(f"[screen] bidirectional & specific: {len(top)}")

    # ---- large-sample generation for the winners ----
    def tstr(t): return norm_tok(tok.convert_ids_to_tokens([int(t)])[0])
    def matches(rule, t):
        kind, val = rule
        if kind == "lex": return tstr(t) in LEX[val]
        if kind == "class": return token_charclass(tok.convert_ids_to_tokens([int(t)])[0]) == val
        return int(t) in set(val)
    @torch.no_grad()
    def sample(prompt_ids, n, new_tokens=60):
        rows = prompt_ids.unsqueeze(0).repeat(n, 1).to(device); gen = torch.zeros(n, 0, dtype=torch.long, device=device)
        for _ in range(new_tokens):
            full = torch.cat([rows, gen], 1)
            lg = fourway(full, predictor(full))[:, -1].float() / 0.9
            p = torch.softmax(lg, -1); sp, si = p.sort(-1, descending=True); cum = sp.cumsum(-1); sp[cum - sp > 0.95] = 0; sp /= sp.sum(-1, keepdim=True)
            gen = torch.cat([gen, si.gather(1, torch.multinomial(sp, 1))], 1)
        return gen.cpu()
    gen_res = {}
    for r in top:
        d_f = (sae.dec.weight[:, r["feature"]] * sd).detach()
        rec = {"rule": r["rule"], "decoded": [tok.convert_ids_to_tokens([t])[0] for t in r["rule"][1]][:10] if r["rule"][0] == "tokset" else r["rule"][1]}
        for a in (-4.0, -2.0, 0.0, 2.0, 4.0):
            state["delta"] = a * next(f["mean_act"] for f in feats if f["feature"] == r["feature"]) * d_f if a != 0 else None
            hits = tot = 0; ex = []
            for p in GEN_PROMPTS:
                pid = torch.tensor(tok(p, add_special_tokens=False)["input_ids"], dtype=torch.long)
                for g in sample(pid, 8):
                    toks = g.tolist(); hits += sum(matches(r["rule"], t) for t in toks); tot += len(toks)
                    ex.append(p + tok.decode(toks))
            rec[str(a)] = {"class_rate": hits / tot, "n_hits": hits, "examples": ex[:3]}
            print(f"[gen] f{r['feature']} {str(rec['decoded'])[:40]} alpha={a:+.0f}: rate {100*hits/tot:.2f}% ({hits}/{tot}) | {ex[0][:90].replace(chr(10), '⏎')}", flush=True)
        state["delta"] = None
        gen_res[f"f{r['feature']}"] = rec
    save_json(gen_res, out / "screen_generation.json")
    for hk in hooks: hk.remove()
    print("[screen] DONE")


if __name__ == "__main__":
    main()
