"""Experiment 2: does the global predictor coordinate routing across layers?

DAGFormer's routing has two channels with the same coordinates:

    pred   FourWayPredictor(input_ids): ONE module emits every layer's alpha
           before the backbone runs (the "global topology" claim)
    corr   correction_mlps[l](hidden_l): a per-layer router reading its own
           layer's residual (an mHC / MUDDFormer-style local router)

If the global channel really plans a *joint* topology, the routing it emits
for layer 8 and for layer 3 of the same token should belong together. The
test: keep every layer's routing distribution intact but break the
cross-layer pairing.

    intact       the model's own routing
    coherent     channel c of ALL layers taken from another window pi(b) at the
                 same positions (a consistent foreign topology)
    incoherent   channel c of layer l taken from window pi_l(b), an independent
                 permutation per layer (same per-layer marginals as coherent,
                 cross-layer pairing destroyed)
    posmean      channel c replaced by its per-position mean over windows

    coherence gap = NLL(incoherent) - NLL(coherent)

measured per channel (pred, corr) and for both at once. A channel whose
layers are coordinated has a positive gap; a stack of independent routers has
none. `coherent - intact` is the token-specificity of the channel (how much the
routing is about *this* token at all), which is the number the interp study
found to be near zero for `pred` on the 300M checkpoint -- so the gap is
reported alongside it and only means something when specificity is non-zero.

Also reported, descriptively: the correlation across layers of each token's
routing-deviation norm (do tokens that deviate from the position mean in one
layer deviate in the others?), for each channel.

Usage:
    python experiments/coherence/cross_layer_patch.py \
        --config /work/hdd/bfqt/shared/dagformer-models/150m-dagformer/config.yaml \
        --ckpt   /work/hdd/bfqt/shared/dagformer-models/150m-dagformer/checkpoint.pt \
        --eval-cache /work/hdd/bfqt/xiaocong/dagformer_pruning/data/mathinstruct/eval_cache.pt \
        --out experiments/coherence/results/150m_math.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from scripts.eval_lm_harness import load_config, load_fourway  # noqa: E402

STREAMS = ("q", "k", "v", "r")


def load_windows(path: str, n: int, seq_len: int) -> torch.Tensor:
    batches = torch.load(path, map_location="cpu", weights_only=False)
    ids = torch.cat([b["olmo_ids"] for b in batches], dim=0)
    assert ids.shape[1] >= seq_len, (ids.shape, seq_len)
    return ids[:n, :seq_len].clone()


class Patcher:
    """Runs the routed model with per-layer donor substitution on either channel."""

    def __init__(self, fourway, predictor, device, batch_size: int):
        self.fw, self.pred, self.device, self.bs = fourway, predictor, device, batch_size
        self.L = fourway.num_layers
        self.has_corr = getattr(fourway, "use_local_correction", False)
        self.corr_store: list[torch.Tensor] | None = None     # per layer: [N, T, out] fp16 cpu
        self.corr_donor: list[torch.Tensor] | None = None     # per layer: [B, T, out] on device, or None
        self._capture: list[torch.Tensor] | None = None
        self._hooks = []
        if self.has_corr:
            for i, mlp in enumerate(fourway.correction_mlps):
                self._hooks.append(mlp.register_forward_hook(self._make_hook(i)))

    def _make_hook(self, i: int):
        def hook(_m, _inp, out):
            if self._capture is not None:
                self._capture[i] = out.detach().half().cpu()
            if self.corr_donor is not None and self.corr_donor[i] is not None:
                return self.corr_donor[i].to(out.dtype)
            return out
        return hook

    @torch.no_grad()
    def capture_corr(self, ids: torch.Tensor) -> None:
        """One intact pass over all windows, storing every layer's correction output."""
        if not self.has_corr:
            return
        store: list[list[torch.Tensor]] = [[] for _ in range(self.L - 1)]
        for s in range(0, ids.shape[0], self.bs):
            b = ids[s:s + self.bs].to(self.device)
            self._capture = [None] * (self.L - 1)
            self.fw(b, self.pred(b))
            for i in range(self.L - 1):
                store[i].append(self._capture[i])
            self._capture = None
        self.corr_store = [torch.cat(x, dim=0) for x in store]

    @torch.no_grad()
    def nll(self, ids: torch.Tensor, pred_perm: np.ndarray | None, corr_perm: np.ndarray | None,
            pred_mean: bool = False, corr_mean: bool = False) -> np.ndarray:
        """Per-window mean NLL [N] under the given substitutions.

        pred_perm / corr_perm: [L-1, N] donor window index per layer (None = intact).
        *_mean: replace the channel with its per-position mean over windows.
        """
        N = ids.shape[0]
        out = np.zeros(N)
        pred_mean_rw = self._pred_position_mean(ids) if pred_mean else None
        corr_mean_store = [c.float().mean(dim=0, keepdim=True) for c in self.corr_store] if corr_mean else None
        for s in range(0, N, self.bs):
            idx = np.arange(s, min(s + self.bs, N))
            b = ids[idx].to(self.device)
            rw = self.pred(b)
            if pred_mean_rw is not None:
                rw = {k: [t[:1].expand(len(idx), *t.shape[1:]).clone() for t in pred_mean_rw[k]] for k in STREAMS}
            elif pred_perm is not None:
                # layer l's routing comes from the donor windows' own prediction
                for i in range(self.L - 1):
                    donor = self.pred(ids[pred_perm[i, idx]].to(self.device))
                    for k in STREAMS:
                        rw[k][i] = donor[k][i]
            if self.has_corr and (corr_perm is not None or corr_mean):
                self.corr_donor = []
                for i in range(self.L - 1):
                    if corr_mean:
                        d = corr_mean_store[i].expand(len(idx), -1, -1)
                    else:
                        d = self.corr_store[i][corr_perm[i, idx]].float()
                    self.corr_donor.append(d.to(self.device))
            logits = self.fw(b, rw).float()
            tgt = b[:, 1:]
            lp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                 reduction="none").view(len(idx), -1)
            out[idx] = lp.mean(dim=1).cpu().numpy()
            self.corr_donor = None
        return out

    @torch.no_grad()
    def _pred_position_mean(self, ids: torch.Tensor) -> dict:
        acc = None
        n = 0
        for s in range(0, ids.shape[0], self.bs):
            rw = self.pred(ids[s:s + self.bs].to(self.device))
            part = {k: [t.float().sum(dim=0, keepdim=True) for t in rw[k]] for k in STREAMS}
            acc = part if acc is None else {k: [a + p for a, p in zip(acc[k], part[k])] for k in STREAMS}
            n += rw["q"][0].shape[0]
        return {k: [t / n for t in acc[k]] for k in STREAMS}

    @torch.no_grad()
    def deviation_norms(self, ids: torch.Tensor) -> dict[str, np.ndarray]:
        """Per-token norm of routing deviation from the position mean, per layer:
        {'pred': [N*T, L-1], 'corr': [N*T, L-1]}."""
        out = {}
        mean_rw = self._pred_position_mean(ids)
        parts = []
        for s in range(0, ids.shape[0], self.bs):
            rw = self.pred(ids[s:s + self.bs].to(self.device))
            per_layer = []
            for i in range(self.L - 1):
                dev = torch.cat([(rw[k][i].float() - mean_rw[k][i]).flatten(2) for k in STREAMS], dim=-1)
                per_layer.append(dev.norm(dim=-1))                       # [B, T]
            parts.append(torch.stack(per_layer, dim=-1).reshape(-1, self.L - 1).cpu())
        out["pred"] = torch.cat(parts).numpy()
        if self.has_corr:
            per_layer = []
            for i in range(self.L - 1):
                c = self.corr_store[i].float()
                per_layer.append((c - c.mean(dim=0, keepdim=True)).norm(dim=-1).reshape(-1))
            out["corr"] = torch.stack(per_layer, dim=-1).numpy()
        return out


def offdiag_corr(x: np.ndarray) -> float:
    c = np.corrcoef(x.T)
    m = ~np.eye(c.shape[0], dtype=bool)
    return float(np.nanmean(c[m]))


def paired(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_boot: int = 2000) -> dict:
    d = a - b
    boots = [rng.choice(d, size=d.size, replace=True).mean() for _ in range(n_boot)]
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "se": float(d.std(ddof=1) / np.sqrt(d.size))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--eval-cache", required=True)
    p.add_argument("--n-windows", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-perms", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    fw, pred = load_fourway(args.ckpt, cfg, device)
    ids = load_windows(args.eval_cache, args.n_windows, args.seq_len)
    N = ids.shape[0]
    patcher = Patcher(fw, pred, device, args.batch_size)
    patcher.capture_corr(ids)
    rng = np.random.default_rng(args.seed)
    L1 = fw.num_layers - 1
    channels = ["pred"] + (["corr", "both"] if patcher.has_corr else [])

    intact = patcher.nll(ids, None, None)
    res: dict = {"config": args.config, "ckpt": args.ckpt, "eval_cache": args.eval_cache,
                 "n_windows": N, "seq_len": args.seq_len, "n_perms": args.n_perms,
                 "intact_nll": float(intact.mean()), "channels": {}}
    print(f"intact NLL {intact.mean():.4f}  (N={N}, T={args.seq_len}, corr={patcher.has_corr})")

    for ch in channels:
        coh, inc = [], []
        for r in range(args.n_perms):
            same = rng.permutation(N)
            coherent = np.tile(same, (L1, 1))
            incoherent = np.stack([rng.permutation(N) for _ in range(L1)])
            pp_c = coherent if ch in ("pred", "both") else None
            cp_c = coherent if ch in ("corr", "both") else None
            pp_i = incoherent if ch in ("pred", "both") else None
            cp_i = incoherent if ch in ("corr", "both") else None
            coh.append(patcher.nll(ids, pp_c, cp_c))
            inc.append(patcher.nll(ids, pp_i, cp_i))
        coh_m, inc_m = np.mean(coh, axis=0), np.mean(inc, axis=0)
        mean_nll = patcher.nll(ids, None, None, pred_mean=ch in ("pred", "both"),
                               corr_mean=ch in ("corr", "both"))
        entry = {
            "coherent_nll": float(coh_m.mean()), "incoherent_nll": float(inc_m.mean()),
            "posmean_nll": float(mean_nll.mean()),
            "specificity(coherent-intact)": paired(coh_m, intact, rng),
            "coherence_gap(incoherent-coherent)": paired(inc_m, coh_m, rng),
            "posmean-intact": paired(mean_nll, intact, rng),
        }
        res["channels"][ch] = entry
        print(f"[{ch:4s}] coherent {coh_m.mean():.4f}  incoherent {inc_m.mean():.4f}  posmean {mean_nll.mean():.4f}"
              f"  | specificity {entry['specificity(coherent-intact)']['mean']:+.4f}"
              f"  gap {entry['coherence_gap(incoherent-coherent)']['mean']:+.4f}"
              f" ci {entry['coherence_gap(incoherent-coherent)']['ci95']}")

    dev = patcher.deviation_norms(ids)
    res["deviation_norm_cross_layer_corr"] = {k: offdiag_corr(v) for k, v in dev.items()}
    res["deviation_norm_mean_per_layer"] = {k: v.mean(axis=0).tolist() for k, v in dev.items()}
    print("cross-layer correlation of deviation norms:", res["deviation_norm_cross_layer_corr"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
