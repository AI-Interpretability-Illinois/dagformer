"""Search for heads/circuits whose effect is VISIBLE in free generation.

Stage A (this script, --mode heads): for every head, ablate its Q/K/V routing
(layer mean) and amplify it (gamma 4), generate text, and measure visible
statistics. Score = how much each statistic moves monotonically across
gamma in {0, 1, 4}. Reports the heads with the largest movement per statistic.

Stage B (--mode circuits): same measurement for connection-level circuits
listed in circuits_all.json (top-10 sets).

Statistics per sample: newlines, mid-sentence line breaks, quotes, brackets,
digits, capitalised words, mean sentence length, distinct-token ratio,
repeated-token ratio, ALL-CAPS ratio.
"""
from __future__ import annotations

import argparse, json, re
from pathlib import Path

import numpy as np
import torch

from interp_common import load_elh, save_json, unflatten_alpha, flatten_alpha
from interp_editing import layer_chunks, stream_slices
from interp_localize import apply_spec_to_chunk
from interp_circuit_verify import apply_conn_set
from interp_token_effects import build_corpus

PROMPTS = ["The", "Yesterday", "In the morning,", "It was a quiet day in the village.",
           "Here is a short story:", "According to the article,", "She looked at him and", "In 1895, the town"]


def stats(texts):
    out = {}
    def per(f): return float(np.mean([f(t) for t in texts]))
    out["newlines"] = per(lambda t: t.count("\n"))
    out["midsent_breaks"] = per(lambda t: len(re.findall(r"[a-z,]\n[a-zA-Z]", t)))
    out["quotes"] = per(lambda t: t.count('"') + t.count("“") + t.count("”"))
    out["brackets"] = per(lambda t: t.count("(") + t.count(")"))
    out["digits"] = per(lambda t: sum(c.isdigit() for c in t) / max(len(t), 1) * 100)
    out["caps_words"] = per(lambda t: len(re.findall(r"\b[A-Z][a-z]+", t)))
    out["allcaps"] = per(lambda t: len(re.findall(r"\b[A-Z]{2,}\b", t)))
    out["sent_len"] = per(lambda t: float(np.mean([len(s.split()) for s in re.split(r"[.!?]+", t) if s.strip()] or [0])))
    out["distinct"] = per(lambda t: len(set(t.split())) / max(len(t.split()), 1))
    out["repeat3gram"] = per(lambda t: 1 - len(set(zip(t.split(), t.split()[1:], t.split()[2:]))) / max(len(t.split()) - 2, 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="heads", choices=["heads", "circuits"])
    ap.add_argument("--n-per-prompt", type=int, default=4)
    ap.add_argument("--new-tokens", type=int, default=70)
    ap.add_argument("--gammas", type=float, nargs="+", default=[-4.0, -1.0, 0.0, 1.0, 4.0, 8.0, 16.0])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="experiments/results/interp/gen_search")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    elh = load_elh()
    cfg = elh.load_config("configs/fourway_300m_dagformer_mmap.yaml")
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    chunks = layer_chunks(L, H)
    fourway, predictor = elh.load_fourway("checkpoints/fourway_300m_dagformer_mmap/checkpoint_step9000.pt", cfg, device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])

    state = {"spec": {}, "edits": {}}
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            def hook(m, inp, o):
                if state["spec"]: return apply_spec_to_chunk(o, i + 1, H, state["spec"])
                e = state["edits"].get(i + 1)
                return apply_conn_set(o, i + 1, H, e) if e else o
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))

    @torch.no_grad()
    def gen_texts():
        texts = []
        for p in PROMPTS:
            pid = torch.tensor(tok(p, add_special_tokens=False)["input_ids"], dtype=torch.long)
            rows = pid.unsqueeze(0).repeat(args.n_per_prompt, 1).to(device)
            gen = torch.zeros(args.n_per_prompt, 0, dtype=torch.long, device=device)
            for _ in range(args.new_tokens):
                full = torch.cat([rows, gen], 1)
                routing = predictor(full)
                if state["spec"] or state["edits"]:
                    flat = flatten_alpha(routing).float(); parts = []
                    for li, (a, b) in enumerate(chunks):
                        seg = flat[:, :, a:b]
                        if state["spec"]: seg = apply_spec_to_chunk(seg, li + 1, H, state["spec"])
                        elif state["edits"].get(li + 1): seg = apply_conn_set(seg, li + 1, H, state["edits"][li + 1])
                        parts.append(seg)
                    routing = {s: list(v) for s, v in unflatten_alpha(torch.cat(parts, -1), L, H).items()}
                lg = fourway(full, routing)[:, -1].float() / 0.9
                pr = torch.softmax(lg, -1); sp, si = pr.sort(-1, descending=True); cum = sp.cumsum(-1); sp[cum - sp > 0.95] = 0; sp /= sp.sum(-1, keepdim=True)
                gen = torch.cat([gen, si.gather(1, torch.multinomial(sp, 1))], 1)
            texts += [p + tok.decode(g.tolist()) for g in gen.cpu()]
        return texts

    def reset(): state["spec"] = {}; state["edits"] = {}

    ids_nll, lab_nll, _ = build_corpus("/work/hdd/bfqt/data/pretok/dolma_v1_7_12b", cfg["seq_len"], 24, 42, 1024, 5500000, offset_seq=0)
    import torch.nn.functional as Fn
    @torch.no_grad()
    def nll_now():
        vals = []
        for i in range(0, ids_nll.shape[0], 12):
            rows = ids_nll[i:i + 12].to(device)
            routing = predictor(rows)
            if state["spec"] or state["edits"]:
                flat = flatten_alpha(routing).float(); parts = []
                for li, (a, b) in enumerate(chunks):
                    seg = flat[:, :, a:b]
                    if state["spec"]: seg = apply_spec_to_chunk(seg, li + 1, H, state["spec"])
                    elif state["edits"].get(li + 1): seg = apply_conn_set(seg, li + 1, H, state["edits"][li + 1])
                    parts.append(seg)
                routing = {s2: list(v) for s2, v in unflatten_alpha(torch.cat(parts, -1), L, H).items()}
            lg = fourway(rows, routing)
            vals.append(Fn.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), lab_nll[i:i + 12].to(device).reshape(-1)).item())
        return float(np.mean(vals))

    reset(); base_texts = gen_texts(); base = stats(base_texts); base_nll = nll_now()
    print(f"[gs] baseline NLL {base_nll:.3f}", flush=True)
    print("[gs] baseline:", {k: round(v, 2) for k, v in base.items()}, flush=True)
    results = {"baseline": base}
    items = []
    if args.mode == "heads":
        items = [(f"L{l}/h{h}", ("head", l, h)) for l in range(1, L) for h in range(H)]
    else:
        for r in json.load(open("experiments/results/interp/token_effects/circuits_all.json")):
            items.append((r["rule"], ("circuit", r["top_connections"][:10], None)))
    if args.limit: items = items[:args.limit]
    for name, spec in items:
        rec = {}
        for g in args.gammas:
            if g == 1.0: reset()
            elif spec[0] == "head":
                _, l, h = spec; state["edits"] = {}; state["spec"] = {l: {"q": ((h,), g), "k": ((h,), g), "v": ((h,), g)}}
            else:
                state["spec"] = {}; state["edits"] = {}
                for c in spec[1]: state["edits"].setdefault(c["layer"], []).append((c["stream"], c["head"], c["src"], g))
            st = stats(gen_texts()); st["nll"] = nll_now()
            # fluency gate: reject degenerate states (token-locked text or large cost)
            st["fluent"] = bool(st["repeat3gram"] <= max(0.05, 3 * base["repeat3gram"]) and st["distinct"] >= 0.6
                                and st["nll"] <= base_nll + 0.35)
            rec[str(g)] = st
        reset()
        # movement score per statistic: |value(g_max) - value(g_min)| normalised by baseline scale
        ok = [g for g in args.gammas if rec[str(g)]["fluent"]]
        mv = {}
        if len(ok) < 3:
            rec["movement"] = {}; rec["best_stat"] = "none"; rec["best_score"] = 0.0
            results[name] = rec
            print(f"[gs] {name:34s} SKIP (only {len(ok)}/{len(args.gammas)} fluent states)", flush=True)
            save_json(results, out / f"search_{args.mode}.json")
            continue
        for k in base:
            vals = np.array([rec[str(g)][k] for g in ok], dtype=float)
            scale = max(abs(base[k]), 1e-3)
            span = (vals.max() - vals.min()) / scale
            # rank correlation with gamma order (monotone trend, robust to scale)
            gr = np.argsort(np.argsort(np.array(ok, dtype=float)))
            vr = np.argsort(np.argsort(vals))
            rho = float(np.corrcoef(gr, vr)[0, 1]) if len(vals) > 2 else 0.0
            mv[k] = span * max(abs(rho), 0.25)
        rec["movement"] = mv; rec["best_stat"] = max(mv, key=mv.get); rec["best_score"] = mv[rec["best_stat"]]
        results[name] = rec
        print(f"[gs] {name:34s} best={rec['best_stat']:14s} score={rec['best_score']:.2f} | " +
              f"fluent {len(ok)}/{len(args.gammas)} | " +
              " ".join(f"{k}:{'/'.join(f'{rec[str(g)][k]:.1f}' for g in ok)}" for k in ("newlines", "quotes", "caps_words", "sent_len")), flush=True)
        save_json(results, out / f"search_{args.mode}.json")
    for hk in hooks: hk.remove()
    top = sorted([k for k in results if k != "baseline"], key=lambda k: -results[k]["best_score"])[:15]
    print("\n[gs] top movers:")
    for k in top:
        print(f"  {k:34s} {results[k]['best_stat']:14s} score {results[k]['best_score']:.2f}")
    print("[gs] DONE")


if __name__ == "__main__":
    main()
