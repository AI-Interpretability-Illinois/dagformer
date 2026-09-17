"""Find the circuit that decides truth vs falsehood under a FIXED deceptive frame.

Design (Xiaocong): hold the framing constant — every item tells the model the
character wants to deceive — and search for the circuit that controls whether
the blank is filled with the context-established value (truthful report) or a
contradicting one.  Cue sensitivity is not required; the contrast is carried by
the intervention, not by the prompt.

Measurement at the blank position: p_true = P(context value) renormalised over
{true value} U {alternatives}.  Baseline p_true is high, so a circuit that
*lowers* it makes the model contradict what it was told, and one that raises it
makes the model stick to the context.

Modes:
  heads  ablate each head's Q/K/V routing (-> layer mean); rank by delta p_true
  conns  connection-level refinement over the top heads
  steer  bidirectional gamma sweep on the discovered set, with a
         size-matched random-connection control
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from interp_common import load_elh, save_json, unflatten_alpha, flatten_alpha
from interp_editing import layer_chunks, stream_slices
from interp_localize import apply_spec_to_chunk
from interp_circuit_verify import apply_conn_set
from interp_liar_cloze import build_items, prompt_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="heads", choices=["heads", "conns", "steer"])
    ap.add_argument("--config", default="configs/fourway_300m_dagformer_mmap.yaml")
    ap.add_argument("--ckpt", default="checkpoints/fourway_300m_dagformer_mmap/checkpoint_step9000.pt")
    ap.add_argument("--n-items", type=int, default=240)
    ap.add_argument("--cue", default="deceptive")
    ap.add_argument("--top-heads", type=int, default=8)
    ap.add_argument("--gammas", type=float, nargs="+", default=[-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--out", default="experiments/results/interp/liar_cloze")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    elh = load_elh()
    cfg = elh.load_config(args.config)
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    chunks = layer_chunks(L, H)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])
    fourway, predictor = elh.load_fourway(args.ckpt, cfg, device)

    # ---- items, bucketed by prompt length so we can batch without padding ----
    items = build_items(args.n_items)
    def tid(w): return tok(" " + w, add_special_tokens=False)["input_ids"][0]
    buckets = defaultdict(list)
    for it in items:
        ids = tok(prompt_for(it, args.cue), add_special_tokens=False)["input_ids"]
        cand = [tid(it["true"])] + [tid(v) for v in it["alts"]]
        buckets[len(ids)].append((ids, cand))
    print(f"[dec] {len(items)} items in {len(buckets)} length buckets "
          f"(sizes {sorted(buckets, key=lambda k: -len(buckets[k]))[:5]})", flush=True)

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
    def p_true() -> float:
        """Mean renormalised P(context value) at the blank, over all items."""
        vals = []
        for n, group in buckets.items():
            for s in range(0, len(group), 32):
                chunk_g = group[s:s + 32]
                rows = torch.tensor([g[0] for g in chunk_g], dtype=torch.long, device=device)
                routing = predictor(rows)
                if state["spec"] or state["edits"]:
                    flat = flatten_alpha(routing).float(); parts = []
                    for li, (a, b) in enumerate(chunks):
                        seg = flat[:, :, a:b]
                        if state["spec"]: seg = apply_spec_to_chunk(seg, li + 1, H, state["spec"])
                        elif state["edits"].get(li + 1): seg = apply_conn_set(seg, li + 1, H, state["edits"][li + 1])
                        parts.append(seg)
                    routing = {k: list(v) for k, v in unflatten_alpha(torch.cat(parts, -1), L, H).items()}
                lg = fourway(rows, routing)[:, -1].float()
                p = torch.softmax(lg, -1)
                for r, (_, cand) in enumerate(chunk_g):
                    ps = p[r, torch.tensor(cand, device=device)]
                    vals.append((ps[0] / ps.sum()).item())
        return float(np.mean(vals))

    def reset(): state["spec"] = {}; state["edits"] = {}
    reset(); base = p_true()
    print(f"[dec] baseline p_true under '{args.cue}' frame = {base:.4f}", flush=True)
    results = {"baseline_p_true": base, "cue": args.cue}

    if args.mode == "heads":
        scores = {}
        for l in range(1, L):
            for h in range(H):
                state["edits"] = {}
                state["spec"] = {l: {"q": ((h,), 0.0), "k": ((h,), 0.0), "v": ((h,), 0.0)}}
                v = p_true(); scores[f"L{l}/h{h}"] = v
                d = v - base
                if abs(d) > 0.01:
                    print(f"[dec] L{l}/h{h}: p_true {v:.4f} ({d:+.4f})", flush=True)
        reset()
        results["head_scores"] = scores
        rank = sorted(scores, key=lambda k: scores[k])
        print("\n[dec] heads whose removal makes the model CONTRADICT context most:")
        for k in rank[:10]: print(f"   {k:9s} p_true {scores[k]:.4f} ({scores[k]-base:+.4f})")
        print("[dec] heads whose removal makes it stick to context MORE:")
        for k in rank[-5:]: print(f"   {k:9s} p_true {scores[k]:.4f} ({scores[k]-base:+.4f})")

    elif args.mode == "conns":
        hs = json.load(open(out / "deception_heads.json"))["head_scores"]
        b = json.load(open(out / "deception_heads.json"))["baseline_p_true"]
        top = sorted(hs, key=lambda k: abs(hs[k] - b), reverse=True)[:args.top_heads]
        print(f"[dec] refining connections in {top}", flush=True)
        cs = {}
        for hk in top:
            l = int(hk[1:].split("/")[0]); h = int(hk.split("/h")[1])
            for stream in ("q", "k", "v"):
                for src in range(l + 1):
                    state["spec"] = {}
                    state["edits"] = {l: [(stream, h, src, 0.0)]}
                    v = p_true(); cs[f"L{l}/h{h}/{stream}<-src{src}"] = v
            print(f"[dec] done {hk}", flush=True)
        reset()
        results["conn_scores"] = cs
        rank = sorted(cs, key=lambda k: cs[k])
        print("\n[dec] connections whose removal most lowers p_true:")
        for k in rank[:12]: print(f"   {k:26s} p_true {cs[k]:.4f} ({cs[k]-base:+.4f})")
        print("[dec] connections whose removal most raises p_true:")
        for k in rank[-8:]: print(f"   {k:26s} p_true {cs[k]:.4f} ({cs[k]-base:+.4f})")

    else:  # steer
        cj = json.load(open(out / "deception_conns.json"))  # circuit found under the deceptive frame
        cs, b = cj["conn_scores"], cj["baseline_p_true"]
        rank = sorted(cs, key=lambda k: cs[k])
        def parse(k):
            l = int(k[1:].split("/")[0]); h = int(k.split("/h")[1].split("/")[0])
            stream = k.split("/")[2].split("<-")[0]; src = int(k.split("src")[1])
            return l, stream, h, src
        circuit = [parse(k) for k in rank[:10]]
        print(f"[dec] circuit (10 connections): {rank[:10]}", flush=True)
        import random as _r
        # Two controls: (a) connections drawn from the SAME top heads -- a
        # deliberately strong control, since those heads already matter; and
        # (b) connections drawn from the whole model, the honest null.
        allc = [parse(k) for k in cs]
        allmodel = [(l, st, h, src) for l in range(1, L) for h in range(H)
                    for st in ("q", "k", "v") for src in range(l + 1)]
        for g in args.gammas:
            state["spec"] = {}; state["edits"] = {}
            for (l, stream, h, src) in circuit:
                state["edits"].setdefault(l, []).append((stream, h, src, g))
            v = p_true()
            results[f"gamma{g}"] = v
            print(f"[dec/steer] gamma={g:+.1f}: p_true {v:.4f} ({v-base:+.4f})", flush=True)
        for pool_name, pool in (("tophead", allc), ("wholemodel", allmodel)):
            vs = []
            for seed in range(5):
                rng2 = _r.Random(1000 + seed)
                state["spec"] = {}; state["edits"] = {}
                for (l, stream, h, src) in rng2.sample(pool, 10):
                    state["edits"].setdefault(l, []).append((stream, h, src, 0.0))
                v = p_true(); vs.append(v)
                results[f"random10_{pool_name}_seed{seed}"] = v
                print(f"[dec/steer] random10 from {pool_name} (seed {seed}): p_true {v:.4f} ({v-base:+.4f})", flush=True)
            import numpy as _np
            print(f"[dec/steer] random10 {pool_name}: mean {_np.mean(vs):.4f} "
                  f"worst {min(vs):.4f} (circuit gamma=0 gives {results.get('gamma0.0', float('nan')):.4f})", flush=True)
        reset()

    for hk_ in hooks: hk_.remove()
    tag = args.mode if args.cue == "deceptive" else f"{args.mode}_{args.cue}"
    save_json(results, out / f"deception_{tag}.json")
    print("[dec] DONE")


if __name__ == "__main__":
    main()
