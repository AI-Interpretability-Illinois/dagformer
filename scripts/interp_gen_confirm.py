"""Confirm the generation-visible movers: (a) save example texts per state,
(b) control for generic degradation by applying a random-direction routing
perturbation of matched magnitude to the same head, (c) measure held-out NLL
so we can report the cost of each state.
"""
from __future__ import annotations

import json, re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interp_common import load_elh, save_json, unflatten_alpha, flatten_alpha
from interp_editing import layer_chunks, stream_slices
from interp_localize import apply_spec_to_chunk
from interp_token_effects import build_corpus

HEADS = [(3, 12), (2, 10), (1, 15), (4, 1), (1, 8), (3, 6)]
GAMMAS = [-4.0, -1.0, 0.0, 1.0, 4.0, 16.0]
PROMPTS = ["The", "Yesterday", "In the morning,", "It was a quiet day in the village.",
           "Here is a short story:", "According to the article,", "She looked at him and", "In 1895, the town"]


def stats(texts):
    f = lambda fn: float(np.mean([fn(t) for t in texts]))
    return {"newlines": f(lambda t: t.count("\n")),
            "sent_len": f(lambda t: float(np.mean([len(s.split()) for s in re.split(r"[.!?]+", t) if s.strip()] or [0]))),
            "distinct": f(lambda t: len(set(t.split())) / max(len(t.split()), 1)),
            "repeat3gram": f(lambda t: 1 - len(set(zip(t.split(), t.split()[1:], t.split()[2:]))) / max(len(t.split()) - 2, 1)),
            "allcaps": f(lambda t: len(re.findall(r"\b[A-Z]{2,}\b", t)))}


def main():
    out = Path("experiments/results/interp/gen_search"); out.mkdir(parents=True, exist_ok=True)
    elh = load_elh()
    cfg = elh.load_config("configs/fourway_300m_dagformer_mmap.yaml")
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]; T = cfg["seq_len"]
    chunks = layer_chunks(L, H)
    fourway, predictor = elh.load_fourway("checkpoints/fourway_300m_dagformer_mmap/checkpoint_step9000.pt", cfg, device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])
    ids_nll, lab_nll, _ = build_corpus("/work/hdd/bfqt/data/pretok/dolma_v1_7_12b", T, 60, 42, 1024, 5500000, offset_seq=0)

    state = {"spec": {}, "noise": None}   # noise = (layer, head, vector) added to that head's routing rows
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            def hook(m, inp, o):
                o2 = apply_spec_to_chunk(o, i + 1, H, state["spec"]) if state["spec"] else o
                nz = state["noise"]
                if nz and nz[0] == i + 1:
                    l, h, vec = nz
                    sl, n = stream_slices(l, H)
                    o2 = o2.clone()
                    off = 0
                    for s in ("q", "k", "v"):
                        s0, s1 = sl[s]
                        seg = o2[:, :, s0:s1].view(*o2.shape[:2], H, n)
                        seg[:, :, h, :] = seg[:, :, h, :] + vec[off:off + n].to(seg.device, seg.dtype)
                        off += n
                return o2
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))

    @torch.no_grad()
    def run_full(rows):
        routing = predictor(rows)
        if state["spec"] or state["noise"]:
            flat = flatten_alpha(routing).float(); parts = []
            for li, (a, b) in enumerate(chunks):
                seg = flat[:, :, a:b]
                if state["spec"]: seg = apply_spec_to_chunk(seg, li + 1, H, state["spec"])
                nz = state["noise"]
                if nz and nz[0] == li + 1:
                    l, h, vec = nz; sl, n = stream_slices(l, H); seg = seg.clone(); off = 0
                    for s in ("q", "k", "v"):
                        s0, s1 = sl[s]; sg = seg[:, :, s0:s1].view(*seg.shape[:2], H, n)
                        sg[:, :, h, :] = sg[:, :, h, :] + vec[off:off + n].to(sg.device, sg.dtype); off += n
                parts.append(seg)
            routing = {s: list(v) for s, v in unflatten_alpha(torch.cat(parts, -1), L, H).items()}
        return fourway(rows, routing)

    @torch.no_grad()
    def gen():
        texts = []
        for p in PROMPTS:
            pid = torch.tensor(tok(p, add_special_tokens=False)["input_ids"], dtype=torch.long)
            rows = pid.unsqueeze(0).repeat(4, 1).to(device); g = torch.zeros(4, 0, dtype=torch.long, device=device)
            for _ in range(70):
                lg = run_full(torch.cat([rows, g], 1))[:, -1].float() / 0.9
                pr = torch.softmax(lg, -1); sp, si = pr.sort(-1, descending=True); cum = sp.cumsum(-1); sp[cum - sp > 0.95] = 0; sp /= sp.sum(-1, keepdim=True)
                g = torch.cat([g, si.gather(1, torch.multinomial(sp, 1))], 1)
            texts += [p + tok.decode(x.tolist()) for x in g.cpu()]
        return texts

    @torch.no_grad()
    def nll():
        vals = []
        for i in range(0, ids_nll.shape[0], 12):
            rows = ids_nll[i:i + 12].to(device)
            lg = run_full(rows)
            vals.append(F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), lab_nll[i:i + 12].to(device).reshape(-1)).item())
        return float(np.mean(vals))

    results = {}
    rng = torch.Generator().manual_seed(0)
    for (l, h) in HEADS:
        rec = {}
        for g in GAMMAS:
            state["noise"] = None
            state["spec"] = {} if g == 1.0 else {l: {"q": ((h,), g), "k": ((h,), g), "v": ((h,), g)}}
            texts = gen(); rec[str(g)] = {**stats(texts), "nll": nll(), "examples": texts[:4]}
            print(f"[cf] L{l}/h{h} g={g:+.0f}: sent {rec[str(g)]['sent_len']:.1f} rep3 {rec[str(g)]['repeat3gram']:.2f} "
                  f"dist {rec[str(g)]['distinct']:.2f} allcaps {rec[str(g)]['allcaps']:.2f} nll {rec[str(g)]['nll']:.3f}", flush=True)
        # random-direction control: same head, perturbation magnitude matched to gamma=-4 deviation
        state["spec"] = {}
        n_src = l + 1
        for scale_tag, scale in (("small", 1.0), ("large", 4.0)):
            vec = torch.randn(3 * n_src, generator=rng) * scale
            state["noise"] = (l, h, vec)
            texts = gen(); rec[f"random_{scale_tag}"] = {**stats(texts), "nll": nll(), "examples": texts[:2]}
            print(f"[cf] L{l}/h{h} random {scale_tag}: sent {rec[f'random_{scale_tag}']['sent_len']:.1f} "
                  f"rep3 {rec[f'random_{scale_tag}']['repeat3gram']:.2f} dist {rec[f'random_{scale_tag}']['distinct']:.2f} "
                  f"nll {rec[f'random_{scale_tag}']['nll']:.3f}", flush=True)
        state["noise"] = None; state["spec"] = {}
        results[f"L{l}/h{h}"] = rec
    save_json(results, out / "confirm.json")
    for hk in hooks: hk.remove()
    print("[cf] DONE")


if __name__ == "__main__":
    main()
