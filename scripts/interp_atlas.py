"""Head function atlas: ablate every head two ways and measure a battery of
behavioral metrics, producing a [heads x metrics] effect matrix.

Ablations per head (layer l, head h):
  wire  replace the head's Q/K/V routing (predictor + correction) by the
        layer mean  -> removes the head's distinct cross-layer wiring
        (routed layers 1..L-1 only)
  out   zero the head's attention output (o_proj input slice) -> classic
        head ablation (all layers incl. 0)

Metrics (from one forward per sequence):
  held-out text: overall NLL; NLL by target char-class, frequency quartile,
    position quartile, baseline-hardness quartile; NLL on targets that occur
    in the previous 128 tokens vs not
  synthetic repeats with period 32 / 128 / 512: copy accuracy
  domain windows: NLL on code / prose / latex
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interp_common import load_elh, save_json, unflatten_alpha, flatten_alpha
from interp_editing import layer_chunks
from interp_localize import apply_spec_to_chunk
from interp_probe import token_charclass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump-dir", default="experiments/results/interp/alpha_dumps")
    ap.add_argument("--out", default="experiments/results/interp")
    ap.add_argument("--n-calib", type=int, default=25)
    ap.add_argument("--n-rep", type=int, default=8)
    args = ap.parse_args()

    elh = load_elh()
    cfg = elh.load_config(args.config)
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    hd = cfg["hidden_size"] // H
    chunks = layer_chunks(L, H)
    dump = Path(args.dump_dir)
    fourway, predictor = elh.load_fourway(args.ckpt, cfg, device)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])

    corp = torch.load(dump / "corpora.pt", map_location="cpu", weights_only=False)
    ids, labels = corp["eval_ids"], corp["eval_labels"]
    nc = args.n_calib
    test_ids, test_lab = ids[nc:], labels[nc:]
    n_test, T = test_ids.shape
    domains = {k: v for k, v in corp["domains"].items()}

    # synthetic repeats at three periods
    g = torch.Generator().manual_seed(11)
    flat_corpus = ids.reshape(-1)
    def sample(n):
        return flat_corpus[torch.randint(0, flat_corpus.shape[0], (n,), generator=g)]
    reps = {p: torch.stack([sample(p).repeat(T // p)[:T] for _ in range(args.n_rep)])
            for p in (32, 128, 512)}

    # ---- target-side masks on held-out text ----
    lab_np = test_lab.numpy()
    uniq = np.unique(lab_np)
    cls_map = {int(i): token_charclass(s) for i, s in zip(uniq, tok.convert_ids_to_tokens(uniq.tolist()))}
    charcls = np.vectorize(lambda i: cls_map[int(i)])(lab_np)
    counts = Counter(ids.reshape(-1).tolist())
    occ = np.vectorize(lambda i: counts[int(i)])(lab_np)
    freqq = np.digitize(occ, np.quantile(occ, [0.25, 0.5, 0.75]))
    posq = np.tile(np.digitize(np.arange(T), [T // 4, T // 2, 3 * T // 4]), (n_test, 1))
    ids_np = test_ids.numpy()
    copyable = np.zeros((n_test, T), dtype=bool)
    for s in range(n_test):
        for t in range(T):
            copyable[s, t] = lab_np[s, t] in ids_np[s, max(0, t - 127):t + 1]

    # ---- ablation machinery ----
    state = {"spec": {}, "zero": None}  # zero = (layer, head) for output ablation
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            def hook(m, inp, out):
                return apply_spec_to_chunk(out, i + 1, H, state["spec"]) if state["spec"] else out
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))
    for l in range(L):
        def make_pre(l):
            def pre(m, args_):
                z = state["zero"]
                if z is None or z[0] != l:
                    return None
                x = args_[0].clone()
                x[..., z[1] * hd:(z[1] + 1) * hd] = 0
                return (x,)
            return pre
        hooks.append(fourway.olmo.model.layers[l].self_attn.o_proj.register_forward_pre_hook(make_pre(l)))

    @torch.no_grad()
    def run(row):
        routing = predictor(row.to(device))
        if state["spec"]:
            flat = flatten_alpha(routing).float()
            parts = [apply_spec_to_chunk(flat[:, :, a:b], li + 1, H, state["spec"])
                     for li, (a, b) in enumerate(chunks)]
            routing = {s: [t.to(device) for t in v] for s, v in
                       unflatten_alpha(torch.cat(parts, dim=-1), L, H).items()}
        return fourway(row.to(device), routing)

    @torch.no_grad()
    def per_token_nll(rows, labs):
        out = []
        for i in range(rows.shape[0]):
            lg = run(rows[i:i + 1])
            out.append(F.cross_entropy(lg.float().view(-1, lg.shape[-1]),
                                       labs[i:i + 1].to(device).view(-1),
                                       reduction="none").cpu())
        return torch.stack(out).numpy()  # [n, T]

    @torch.no_grad()
    def copy_acc(rows, period):
        m = torch.zeros(T - 1, dtype=torch.bool); m[period - 1:] = True
        accs = []
        for i in range(rows.shape[0]):
            row = rows[i:i + 1]
            pred = run(row)[:, :-1].argmax(-1).cpu().view(-1)
            accs.append((pred == row[:, 1:].view(-1))[m].float().mean().item())
        return float(np.mean(accs))

    @torch.no_grad()
    def domain_nll(w):
        vals = []
        for i in range(w.shape[0]):
            row = w[i:i + 1]
            lg = run(row)[:, :-1]
            vals.append(F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]),
                                        row[:, 1:].to(device).reshape(-1)).item())
        return float(np.mean(vals))

    hard_q = None

    def battery():
        nll = per_token_nll(test_ids, test_lab)
        m = {"nll_overall": float(nll.mean())}
        for c, name in enumerate(["ws", "punct", "digit", "cap", "lower"]):
            sel = charcls == c
            if sel.sum() > 50:
                m[f"nll_class_{name}"] = float(nll[sel].mean())
        for q in range(4):
            m[f"nll_freq_q{q}"] = float(nll[freqq == q].mean())
            m[f"nll_pos_q{q}"] = float(nll[posq == q].mean())
            if hard_q is not None:
                m[f"nll_hard_q{q}"] = float(nll[hard_q == q].mean())
        m["nll_copyable"] = float(nll[copyable].mean())
        m["nll_noncopyable"] = float(nll[~copyable].mean())
        for p, rows in reps.items():
            m[f"copy_acc_p{p}"] = copy_acc(rows, p)
        for name, w in domains.items():
            m[f"nll_domain_{name}"] = domain_nll(w)
        return m, nll

    state["spec"] = {}; state["zero"] = None
    base, base_nll = battery()
    hard_q = np.digitize(base_nll, np.quantile(base_nll, [0.25, 0.5, 0.75]))
    base, _ = battery()  # recompute with hardness quartiles available
    print("[atlas] baseline:", json.dumps({k: round(v, 4) for k, v in base.items()}))

    results = {"baseline": base, "wire": {}, "out": {}}
    total = (L - 1) * H + L * H
    done = 0
    for l in range(L):
        for h in range(H):
            if l >= 1:
                state["spec"] = {l: {"q": ((h,), 0.0), "k": ((h,), 0.0), "v": ((h,), 0.0)}}
                state["zero"] = None
                m, _ = battery()
                results["wire"][f"L{l}/h{h}"] = m
                done += 1
                print(f"[atlas] wire L{l}/h{h}: dNLL={m['nll_overall'] - base['nll_overall']:+.4f} "
                      f"copy128={m['copy_acc_p128']:.3f} ({done}/{total})", flush=True)
            state["spec"] = {}
            state["zero"] = (l, h)
            m, _ = battery()
            results["out"][f"L{l}/h{h}"] = m
            done += 1
            print(f"[atlas] out  L{l}/h{h}: dNLL={m['nll_overall'] - base['nll_overall']:+.4f} "
                  f"copy128={m['copy_acc_p128']:.3f} ({done}/{total})", flush=True)
            state["zero"] = None
            if done % 32 == 0:
                save_json(results, Path(args.out) / "atlas_step9000.json")

    for hk in hooks:
        hk.remove()
    save_json(results, Path(args.out) / "atlas_step9000.json")
    print("[atlas] DONE")


if __name__ == "__main__":
    main()
