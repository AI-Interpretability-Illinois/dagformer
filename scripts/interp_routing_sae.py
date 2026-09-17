"""Dictionary learning on the routing interface (the SAE route to
activation + suppression steering).

Stage 1: train a sparse autoencoder on per-token correction vectors
         (the content channel, 3773 dims) streamed from unseen windows.
Stage 2: interpret features: top-activating contexts, enrichment-based
         auto-name (target token class / token set), activation frequency.
Stage 3: steer: add  alpha * a_f * d_f  to the correction outputs at every
         position (d_f = unit decoder direction, a_f = feature's mean active
         value) for alpha in {-3,-1,0,+1,+3}; measure the named class rate
         and repetition in sampled text, and on-rule NLL on held-out text.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interp_common import load_elh, save_json
from interp_editing import layer_chunks
from interp_discover3 import LEX, norm_tok
from interp_probe import token_charclass
from interp_token_effects import build_corpus


class SAE(torch.nn.Module):
    """Top-k sparse autoencoder: exactly k active features per token."""
    def __init__(self, d, f, k):
        super().__init__()
        self.k = k
        self.enc = torch.nn.Linear(d, f)
        self.dec = torch.nn.Linear(f, d, bias=True)
        with torch.no_grad():
            self.dec.weight.copy_(F.normalize(torch.randn(d, f), dim=0))
            self.enc.weight.copy_(self.dec.weight.t())
    def forward(self, x):
        pre = torch.relu(self.enc(x))
        v, i = pre.topk(self.k, dim=-1)
        a = torch.zeros_like(pre).scatter_(-1, i, v)
        return self.dec(a), a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-feat", type=int, default=8192)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--n-nll-seq", type=int, default=100)
    ap.add_argument("--l1", type=float, default=2e-3)
    ap.add_argument("--n-train-seq", type=int, default=4000)
    ap.add_argument("--n-interp-seq", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n-steer", type=int, default=12)
    ap.add_argument("--out", default="experiments/results/interp/routing_sae")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    elh = load_elh()
    cfg = elh.load_config("configs/fourway_300m_dagformer_mmap.yaml")
    device = torch.device("cuda")
    L, H = cfg["num_hidden_layers"], cfg["num_attention_heads"]
    T = cfg["seq_len"]
    chunks = layer_chunks(L, H); D = chunks[-1][1]
    fourway, predictor = elh.load_fourway("checkpoints/fourway_300m_dagformer_mmap/checkpoint_step9000.pt", cfg, device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"])

    ids_tr, _, _ = build_corpus("/work/hdd/bfqt/data/pretok/dolma_v1_7_12b", T, args.n_train_seq, 42, 1024, 5500000, offset_seq=1000)
    ids_in, lab_in, _ = build_corpus("/work/hdd/bfqt/data/pretok/dolma_v1_7_12b", T, args.n_interp_seq, 42, 1024, 5500000, offset_seq=0)

    cap = {}
    state = {"delta": None}   # [D] vector added to corrections at every position
    hooks = []
    for i, mlp in enumerate(fourway.correction_mlps):
        def make(i):
            a, b = chunks[i]
            def hook(m, inp, o):
                cap[i] = o.detach()
                dl = state["delta"]
                return o if dl is None else o + dl[a:b].to(o.device, o.dtype)
            return hook
        hooks.append(mlp.register_forward_hook(make(i)))

    @torch.no_grad()
    def corr_vectors(rows):
        cap.clear()
        logits = fourway(rows.to(device), predictor(rows.to(device)))
        return torch.cat([cap[i].float() for i in range(len(chunks))], -1), logits  # [B,T,D]

    # ---- normalization stats ----
    with torch.no_grad():
        xs = torch.cat([corr_vectors(ids_tr[i:i + 16])[0].reshape(-1, D) for i in range(0, 64, 16)])
        mu, sd = xs.mean(0), xs.std(0) + 1e-6
    sae = SAE(D, args.n_feat, args.topk).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-4)
    step = 0
    for ep in range(args.epochs):
        perm = torch.randperm(ids_tr.shape[0])
        for i in range(0, ids_tr.shape[0], 16):
            rows = ids_tr[perm[i:i + 16]]
            with torch.no_grad():
                x = ((corr_vectors(rows)[0].reshape(-1, D) - mu) / sd)
            x = x[torch.randperm(x.shape[0])]
            for j in range(0, x.shape[0], 4096):
                xb = x[j:j + 4096]
                rec, a = sae(xb)
                loss = F.mse_loss(rec, xb)
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    sae.dec.weight.data = F.normalize(sae.dec.weight.data, dim=0)
                step += 1
            if (i // 16) % 50 == 0:
                with torch.no_grad():
                    r2 = 1 - F.mse_loss(rec, xb) / xb.var()
                    l0 = (a > 0).float().sum(1).mean()
                print(f"[sae] ep{ep} seq{i} step{step} R2={r2:.3f} L0={l0:.1f}", flush=True)
    torch.save({"sae": sae.state_dict(), "mu": mu.cpu(), "sd": sd.cpu()}, out / "sae.pt")

    # ---- interpretation ----
    K = 50
    top_val = torch.full((args.n_feat, K), -1.0, device=device); top_idx = torch.zeros((args.n_feat, K, 2), dtype=torch.long, device=device)
    freq = torch.zeros(args.n_feat, device=device); mean_act = torch.zeros(args.n_feat, device=device)
    n_tok = 0
    with torch.no_grad():
        for i in range(0, ids_in.shape[0], 16):
            rows = ids_in[i:i + 16]
            x = (corr_vectors(rows)[0] - mu) / sd
            _, a = sae(x.reshape(-1, D))
            a = a.view(rows.shape[0], T, -1)
            freq += (a > 0).float().sum((0, 1)); mean_act += a.sum((0, 1)); n_tok += rows.shape[0] * T
            flat = a.reshape(-1, args.n_feat).t()   # [F, B*T]
            v, ix = flat.topk(min(K, flat.shape[1]), dim=1)
            allv = torch.cat([top_val, v], 1); alli = torch.cat([top_idx, torch.stack([ix // T + i, ix % T], -1)], 1)
            sel = allv.topk(K, dim=1).indices
            top_val = allv.gather(1, sel); top_idx = alli.gather(1, sel.unsqueeze(-1).expand(-1, -1, 2))
    freq = (freq / n_tok).cpu(); mean_act = (mean_act / freq.clamp(min=1e-9).to(device) / n_tok).cpu()
    lab_np = lab_in.numpy()
    uniq = np.unique(np.concatenate([lab_np.reshape(-1), ids_in.numpy().reshape(-1)]))
    raw = dict(zip(uniq.tolist(), tok.convert_ids_to_tokens(uniq.tolist())))
    counts = Counter(lab_np.reshape(-1).tolist()); total = lab_np.size
    lex_ids = {n: {i for i, s in raw.items() if norm_tok(s) in w or s.replace("Ġ", "") in w} for n, w in LEX.items()}
    base_lex = {n: sum(counts[i] for i in ids_) / total for n, ids_ in lex_ids.items()}
    base_cls = {c: float(np.mean(np.vectorize(lambda i: token_charclass(raw[i]) == c)(lab_np))) for c in range(6)}

    def auto_name(f):
        idx = top_idx[f].cpu().numpy(); vals = top_val[f].cpu().numpy()
        good = idx[vals > 0]
        if len(good) < 10: return None, None
        tg = [int(lab_np[s, t]) for s, t in good]
        best = (None, 0.0)
        for n, ids_ in lex_ids.items():
            r = np.mean([t in ids_ for t in tg]) / max(base_lex[n], 1e-4)
            if r > best[1] and np.mean([t in ids_ for t in tg]) >= 0.3: best = (("lex", n), r)
        for c in range(6):
            r = np.mean([token_charclass(raw[t]) == c for t in tg]) / max(base_cls[c], 1e-4)
            if r > best[1] and np.mean([token_charclass(raw[t]) == c for t in tg]) >= 0.5: best = (("class", c), r)
        tc = Counter(tg)
        tset = [t for t, n in tc.items() if n >= 3 and (n / len(tg)) / max(counts[t] / total, 1e-7) >= 5]
        if len(tset) >= 3:
            cov = sum(tc[t] for t in tset) / len(tg)
            r = cov / max(sum(counts[t] for t in tset) / total, 1e-4)
            if r > best[1] and cov >= 0.4: best = (("tokset", tset), r)
        return best
    feats = []
    for f in range(args.n_feat):
        if freq[f] < 5e-4 or freq[f] > 0.03: continue
        rule, lift = auto_name(f)
        if rule is None: continue
        idx = top_idx[f].cpu().numpy()
        snippets = [(tok.decode(ids_in[s, max(0, t - 12):t + 1].tolist()).replace("\n", "⏎")[-70:], raw[int(lab_np[s, t])]) for s, t in idx[:8]]
        feats.append({"feature": f, "freq": float(freq[f]), "mean_act": float(mean_act[f]), "rule": rule, "lift": float(lift), "snippets": snippets})
    feats.sort(key=lambda r: -r["lift"] * min(r["freq"], 0.05))
    print(f"[sae] features alive {(freq > 0).sum().item()}/{args.n_feat}; auto-named {len(feats)}")
    for r in feats[:30]:
        print(f"  f{r['feature']:5d} freq={r['freq']*100:.2f}% lift={r['lift']:.1f} rule={r['rule']} | " + " || ".join(f"{c} >>{t}<<" for c, t in r["snippets"][:3]))
    save_json({"features": feats[:200]}, out / "features.json")

    # ---- steering ----
    PROMPTS = ["The", "Yesterday", "In the morning,", "It was a quiet day in the village.", "Here is a short story:", "The report begins as follows.", "According to the article,", "Once upon a time"]
    def matches(rule, tid):
        kind, val = rule
        if kind == "lex": return tid in lex_ids[val]
        if kind == "class": return token_charclass(raw.get(tid, tok.convert_ids_to_tokens([tid])[0])) == val
        return tid in val
    @torch.no_grad()
    def sample(prompt_ids, n, new_tokens=60):
        rows = prompt_ids.unsqueeze(0).repeat(n, 1).to(device); gen = torch.zeros(n, 0, dtype=torch.long, device=device)
        for _ in range(new_tokens):
            lg = fourway(torch.cat([rows, gen], 1), predictor(torch.cat([rows, gen], 1)))[:, -1].float() / 0.9
            p = torch.softmax(lg, -1); sp, si = p.sort(-1, descending=True); cum = sp.cumsum(-1); sp[cum - sp > 0.95] = 0; sp /= sp.sum(-1, keepdim=True)
            gen = torch.cat([gen, si.gather(1, torch.multinomial(sp, 1))], 1)
        return gen.cpu()
    nll_ids, nll_lab = ids_in[:args.n_nll_seq], lab_in[:args.n_nll_seq]
    @torch.no_grad()
    def per_token_nll():
        outs = []
        for i in range(0, nll_ids.shape[0], 16):
            rows = nll_ids[i:i + 16]
            lg = fourway(rows.to(device), predictor(rows.to(device)))
            outs.append(F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), nll_lab[i:i + 16].to(device).reshape(-1), reduction="none").view(rows.shape[0], T).cpu())
        return torch.cat(outs)
    state["delta"] = None
    base_nll = per_token_nll()
    steer = {}
    for r in feats[:args.n_steer]:
        rule_mask = torch.tensor(np.vectorize(lambda t: matches(r["rule"], int(t)))(nll_lab.numpy()))
        f = r["feature"]; d_f = (sae.dec.weight[:, f] * sd).detach()   # back to raw correction units
        res = {}
        for alpha in (-3.0, -1.0, 0.0, 1.0, 3.0):
            state["delta"] = alpha * r["mean_act"] * d_f if alpha != 0 else None
            hits = tot = 0; ex = []
            for p in PROMPTS:
                pid = torch.tensor(tok(p, add_special_tokens=False)["input_ids"], dtype=torch.long)
                for g in sample(pid, 2):
                    for t in g.tolist():
                        raw.setdefault(int(t), tok.convert_ids_to_tokens([int(t)])[0])
                        hits += int(matches(r["rule"], int(t))); tot += 1
                    ex.append(p + tok.decode(g.tolist()))
            dn = per_token_nll() - base_nll
            on, off = float(dn[rule_mask].mean()), float(dn[~rule_mask].mean())
            res[str(alpha)] = {"class_rate": hits / tot, "on_rule_dnll": on, "off_rule_dnll": off, "example": ex[0][:200]}
            print(f"[steer] f{f} {str(r['rule'])[:40]} alpha={alpha:+.0f}: gen class rate {100*hits/tot:.1f}% | on-rule dNLL {on:+.3f} off {off:+.3f} | {ex[0][:90].replace(chr(10), '⏎')}", flush=True)
        state["delta"] = None
        steer[f"f{f}"] = {"rule": str(r["rule"]), **res}
    save_json(steer, out / "steering.json")
    for hk in hooks: hk.remove()
    print("[sae] DONE")


if __name__ == "__main__":
    main()
