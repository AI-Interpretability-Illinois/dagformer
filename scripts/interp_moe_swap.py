"""The same routing-intervention surgery, applied to an open MoE model
(OLMoE-1B-7B): is expert routing content-driven, position-driven, or
merely lexical (current-token-id driven)?

Each OLMoE layer routes tokens to 8 of 64 experts via a linear gate over the
hidden state. We hook every gate and replace its logits with:
  live            real gates (sanity)
  pos_table       per-position calib mean (content-free schedule)
  static_mean     global calib mean (constant)
  swap_context    another sequence's gate logits (content misaligned)
  shuffle_time    per-seq position permutation
  token_id        per-current-token-id calib mean (lexical router)

Contrast column for the DAGFormer anatomy: our external predictor was
position-equivalent; our corrections were content-essential. Where does a
classic MoE router fall? Literature suggests heavily lexical.

Also: variance decomposition + linear probes on the live router logits
(16 layers x 64 experts = 1024-dim per-token feature), reusing the probe
battery so the numbers are directly comparable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interp_common import REPO, save_json

MODEL_ID = "allenai/OLMoE-1B-7B-0125-Instruct"


def build_windows(seq_len: int) -> torch.Tensor:
    """Slice held-out dolma raw text (from the trainer's eval cache) into
    fixed windows under OLMoE's own tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    batches = torch.load(REPO / "checkpoints/fourway_300m_dagformer_mmap/eval_cache.pt",
                         map_location="cpu", weights_only=False)
    texts = [t for b in batches for t in b["raw_text"]]
    ids = tok("\n\n".join(texts), add_special_tokens=False)["input_ids"]
    n = len(ids) // seq_len
    return torch.tensor([ids[i * seq_len:(i + 1) * seq_len] for i in range(n)],
                        dtype=torch.long), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results/interp")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--n-calib", type=int, default=25)
    args = ap.parse_args()

    device = torch.device("cuda")
    windows, tok = build_windows(args.seq_len)
    N, T = windows.shape
    nc = min(args.n_calib, N // 2)
    n_test = N - nc
    print(f"[moe] windows {N}x{T}, calib={nc} test={n_test}")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    gates = [layer.mlp.gate for layer in model.model.layers]
    L, E = len(gates), model.config.num_experts
    print(f"[moe] {L} layers x {E} experts, top-{model.config.num_experts_per_tok}")

    # --- capture live gate logits ---
    captured: dict[int, torch.Tensor] = {}
    replacement: list[torch.Tensor | None] = [None] * L
    hooks = []
    for i, g in enumerate(gates):
        def make(i):
            def hook(m, inp, out):
                captured[i] = out.detach()
                r = replacement[i]
                return out if r is None else r.to(device=out.device, dtype=out.dtype)
            return hook
        hooks.append(g.register_forward_hook(make(i)))

    @torch.no_grad()
    def forward_nll(row: torch.Tensor) -> float:
        logits = model(row.to(device)).logits[:, :-1]
        lab = row[:, 1:].to(device)
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                               lab.reshape(-1)).item()

    @torch.no_grad()
    def dump_gates(rows: torch.Tensor) -> torch.Tensor:
        outs = []
        for j in range(rows.shape[0]):
            captured.clear()
            model(rows[j:j + 1].to(device))
            flat = torch.cat([captured[i].float().view(T, E) for i in range(L)], dim=-1)
            outs.append(flat.half().cpu())
        return torch.stack(outs)                       # [n, T, L*E]

    print("[moe] dumping live gate logits...")
    glogits = dump_gates(windows)                      # [N, T, L*E]
    torch.save(glogits, Path(args.out) / "moe_gate_logits.pt")

    calib = glogits[:nc].float()
    pos_table = calib.mean(0)                          # [T, L*E]
    static_vec = calib.reshape(-1, calib.shape[-1]).mean(0)

    # token-id table from calib
    ids_flat = windows[:nc].reshape(-1).numpy()
    X = calib.reshape(-1, calib.shape[-1]).numpy()
    uniq, inv = np.unique(ids_flat, return_inverse=True)
    sums = np.zeros((len(uniq), X.shape[1]), dtype=np.float64)
    np.add.at(sums, inv, X)
    means = (sums / np.bincount(inv)[:, None]).astype(np.float32)
    lookup = {int(u): k for k, u in enumerate(uniq)}
    gmean = X.mean(0)

    def token_id_flat(row_ids: torch.Tensor) -> torch.Tensor:
        M = np.array([means[lookup[int(i)]] if int(i) in lookup else gmean
                      for i in row_ids.view(-1)], dtype=np.float32)
        return torch.from_numpy(M)                     # [T, L*E]

    g = torch.Generator().manual_seed(0)
    perms = [torch.randperm(T, generator=g) for _ in range(n_test)]

    def set_replacement(flat: torch.Tensor | None):
        for i in range(L):
            replacement[i] = None if flat is None else flat[:, i * E:(i + 1) * E]

    def flat_for(mode: str, i: int) -> torch.Tensor | None:
        ti = nc + i
        if mode == "live":
            return None
        if mode == "pos_table":
            return pos_table
        if mode == "static_mean":
            return static_vec.view(1, -1).expand(T, -1)
        if mode == "swap_context":
            return glogits[nc + (i + 1) % n_test].float()
        if mode == "shuffle_time":
            return glogits[ti].float()[perms[i]]
        if mode == "token_id":
            return token_id_flat(windows[ti])
        raise ValueError(mode)

    results = {}
    for mode in ("live", "token_id", "pos_table", "static_mean",
                 "swap_context", "shuffle_time"):
        vals = []
        for i in range(n_test):
            set_replacement(flat_for(mode, i))
            vals.append(forward_nll(windows[nc + i:nc + i + 1]))
        set_replacement(None)
        results[mode] = sum(vals) / len(vals)
        print(f"[moe] {mode:14s} NLL = {results[mode]:.4f}")

    for h in hooks:
        h.remove()

    # --- variance decomposition + probes on live logits ---
    a = glogits.float().numpy()
    pos_var = float(a.mean(axis=0).var(axis=0).mean())
    cont_var = float(a.var(axis=0).mean(axis=0).mean())
    vd = {"pos_var": pos_var, "content_var": cont_var,
          "pos_share": pos_var / (pos_var + cont_var)}
    print(f"[moe] variance decomposition: {vd}")

    import interp_probe as ip
    targets = ip.build_targets(windows, tok)
    tr_seq = np.arange(N) < int(0.8 * N)
    probes = {}
    for tname in ("posbucket", "isrepeat64", "charclass", "freqbucket"):
        y = targets[tname]
        Xtr = a[tr_seq].reshape(-1, a.shape[-1])
        Xte = a[~tr_seq].reshape(-1, a.shape[-1])
        ytr, yte = y[tr_seq].reshape(-1), y[~tr_seq].reshape(-1)
        itr = windows.numpy()[tr_seq].reshape(-1)
        ite = windows.numpy()[~tr_seq].reshape(-1)
        acc, _, _ = ip.fit_probe(Xtr, ytr, Xte, yte)
        residualize = ip.per_id_mean_alpha(itr, Xtr)
        acc_r, _, _ = ip.fit_probe(residualize(itr, Xtr), ytr,
                                   residualize(ite, Xte), yte)
        probes[tname] = {"gate": acc, "gate_resid": acc_r,
                         "token_id": ip.token_id_baseline(itr, ytr, ite, yte),
                         "majority": ip.majority_baseline(ytr, yte)}
        print(f"[moeprobe] {tname:12s} gate={acc:.3f} resid={acc_r:.3f} "
              f"token_id={probes[tname]['token_id']:.3f} "
              f"majority={probes[tname]['majority']:.3f}")

    save_json({"model": MODEL_ID, "modes_nll": results,
               "variance_decomposition": vd, "probes": probes,
               "n_calib": nc, "n_test": n_test},
              Path(args.out) / "moe_swap_olmoe.json")
    print("[moe] DONE")


if __name__ == "__main__":
    main()
