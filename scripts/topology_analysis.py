"""Topology analysis of a trained FourWay DAGFormer (J1 / J3 / J4 of the NAACL analysis plan).

Three questions, one checkpoint load:

  dump    What did the predictor learn? Mean and per-token spread of alpha over windows of the
          model's own held-out cache, per layer / stream / head / source, for the predictor channel
          and (corrected variant) the local correction channel. -> <out>_topology.npz + summary JSON.
  static  Is the topology static? Held-out NLL on the given eval caches with the predictor's output
          replaced by (a) its global mean (one constant alpha tensor per layer), (b) its per-position
          mean, and the correction channel replaced by its global mean, and both at once (a model with
          NO input-dependent routing at all). Means are estimated on the own cache, evaluated on all.
  prune   How sparse is the learned graph? With both channels static, zero the hyperconnection entries
          (sources other than the previous layer) of smallest |alpha|, keeping a fraction of them, and
          re-evaluate; also "previous layer only" (all skips removed).

    python scripts/topology_analysis.py --config CFG --ckpt CKPT --own-cache OWN.pt \
        --eval wikitext2=W.pt --eval mathinstruct=M.pt --eval gsm8k=G.pt --eval dolma21b=OWN.pt \
        --out experiments/topology/300m_corrected [--n-windows 128] [--seq-len 1024] [--skip-prune]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO)

from scripts.eval_lm_harness import load_config, load_fourway  # noqa: E402

STREAMS = ("q", "k", "v", "r")


def load_ids(path: str, n: int | None, seq_len: int) -> torch.Tensor:
    batches = torch.load(path, map_location="cpu", weights_only=False)
    ids = torch.cat([b["olmo_ids"] for b in batches], dim=0)
    ids = ids[:, :seq_len]
    return ids if n is None else ids[:n].clone()


class Analyzer:
    def __init__(self, fw, pred, device, bs: int):
        self.fw, self.pred, self.device, self.bs = fw, pred, device, bs
        self.L, self.H = fw.num_layers, fw.num_heads
        self.has_corr = bool(getattr(fw, "use_local_correction", False))
        self.corr_override: list[torch.Tensor | None] | None = None   # per layer [1 or B, T or 1, out]
        self.corr_mask: list[torch.Tensor | None] | None = None       # per layer [out] multiplicative
        self._capture: list | None = None
        if self.has_corr:
            for i, mlp in enumerate(fw.correction_mlps):
                mlp.register_forward_hook(self._hook(i))

    def _hook(self, i):
        def hook(_m, _inp, out):
            if self._capture is not None:
                self._capture[i] = out.detach().float()
            if self.corr_override is not None and self.corr_override[i] is not None:
                out = self.corr_override[i].to(device=out.device, dtype=out.dtype).expand_as(out)
            if self.corr_mask is not None and self.corr_mask[i] is not None:
                out = out * self.corr_mask[i].to(device=out.device, dtype=out.dtype)
            return out
        return hook

    # ---- statistics of both channels over windows -------------------------------------------
    @torch.no_grad()
    def stats(self, ids: torch.Tensor) -> dict:
        """Returns per layer: pred mean/var per (stream, head, src), corr mean/var in the flat layout,
        and the per-position pred mean. Means are over windows and positions."""
        L, H = self.L, self.H
        acc = {}
        n_tok = 0
        pos_sum = None
        for s in range(0, ids.shape[0], self.bs):
            b = ids[s:s + self.bs].to(self.device)
            self._capture = [None] * (L - 1)
            rw = self.pred(b)
            self.fw(b, rw)
            B, T = b.shape
            for i in range(L - 1):
                a = torch.cat([rw[k][i].float().reshape(B, T, -1) for k in STREAMS], dim=-1)   # [B,T,out]
                c = self._capture[i] if self.has_corr else torch.zeros_like(a)
                d = acc.setdefault(i, {"p1": 0, "p2": 0, "c1": 0, "c2": 0})
                d["p1"] = d["p1"] + a.sum((0, 1)); d["p2"] = d["p2"] + (a * a).sum((0, 1))
                d["c1"] = d["c1"] + c.sum((0, 1)); d["c2"] = d["c2"] + (c * c).sum((0, 1))
                if pos_sum is None:
                    pos_sum = {}
                pos_sum[i] = pos_sum.get(i, 0) + a.sum(0)                                       # [T,out]
            n_tok += B * T
            self._capture = None
        out = {}
        for i in range(L - 1):
            d = acc[i]
            pm, cm = d["p1"] / n_tok, d["c1"] / n_tok
            out[i] = {"pred_mean": pm.cpu(), "pred_var": (d["p2"] / n_tok - pm * pm).clamp_min(0).cpu(),
                      "corr_mean": cm.cpu(), "corr_var": (d["c2"] / n_tok - cm * cm).clamp_min(0).cpu(),
                      "pred_posmean": (pos_sum[i] / (n_tok // ids.shape[1])).cpu()}
        return out

    def split(self, flat: torch.Tensor, i: int) -> dict:
        """flat [..., out] of routed layer i (layer index i+1) -> dict of q/k/v [...,H,n_src], r [...,n_src]."""
        n_src = i + 2
        H = self.H
        q = flat[..., :H * n_src].reshape(*flat.shape[:-1], H, n_src)
        k = flat[..., H * n_src:2 * H * n_src].reshape(*flat.shape[:-1], H, n_src)
        v = flat[..., 2 * H * n_src:3 * H * n_src].reshape(*flat.shape[:-1], H, n_src)
        r = flat[..., 3 * H * n_src:]
        return {"q": q, "k": k, "v": v, "r": r}

    # ---- NLL under substitutions -----------------------------------------------------------
    @torch.no_grad()
    def nll(self, ids: torch.Tensor, pred_const: dict | None = None, pred_pos: dict | None = None,
            corr_const: list | None = None, pred_mask: list | None = None, corr_mask: list | None = None) -> float:
        total, n = 0.0, 0
        self.corr_override = corr_const
        self.corr_mask = corr_mask
        for s in range(0, ids.shape[0], self.bs):
            b = ids[s:s + self.bs].to(self.device)
            B, T = b.shape
            if pred_const is not None:
                rw = {k: [pred_const[k][i].to(self.device).expand(B, T, *pred_const[k][i].shape[2:]) for i in range(self.L - 1)] for k in STREAMS}
            elif pred_pos is not None:
                rw = {k: [pred_pos[k][i][:, :T].to(self.device).expand(B, T, *pred_pos[k][i].shape[2:]) for i in range(self.L - 1)] for k in STREAMS}
            else:
                rw = self.pred(b)
            if pred_mask is not None:
                rw = {k: [rw[k][i] * pred_mask[k][i].to(rw[k][i].dtype).to(self.device) for i in range(self.L - 1)] for k in STREAMS}
            logits = self.fw(b, rw).float()
            total += F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1), reduction="sum").item()
            n += B * (T - 1)
        self.corr_override = None
        self.corr_mask = None
        return total / n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--own-cache", required=True, help="cache used to estimate the static topology")
    p.add_argument("--eval", action="append", default=[], help="name=eval_cache.pt (evaluated on all windows)")
    p.add_argument("--n-windows", type=int, default=128, help="windows of the own cache for the statistics")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--keep", default="0.75,0.5,0.25,0.1,0.0", help="fractions of hyperconnection entries kept (prune)")
    p.add_argument("--skip-prune", action="store_true")
    p.add_argument("--out", required=True, help="output prefix")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    cfg = load_config(args.config)
    fw, pred = load_fourway(args.ckpt, cfg, device)
    fw.eval(); pred.eval()
    an = Analyzer(fw, pred, device, args.batch_size)
    L, H = an.L, an.H
    own = load_ids(args.own_cache, args.n_windows, args.seq_len)
    evals = {e.split("=")[0]: load_ids(e.split("=", 1)[1], None, args.seq_len) for e in args.eval}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    res = {"config": args.config, "ckpt": args.ckpt, "n_windows_stats": int(own.shape[0]), "layers": L, "heads": H,
           "has_corr": an.has_corr}

    # ---- J3: dump ------------------------------------------------------------------------
    st = an.stats(own)
    npz = {}
    summ = []
    for i in range(L - 1):
        pm, pv, cm, cv = st[i]["pred_mean"], st[i]["pred_var"], st[i]["corr_mean"], st[i]["corr_var"]
        eff = pm + cm
        sp, sv, sc = an.split(pm, i), an.split(pv, i), an.split(cm, i)
        se = an.split(eff, i)
        for k in STREAMS:
            npz[f"L{i+1}_{k}_pred_mean"] = sp[k].numpy(); npz[f"L{i+1}_{k}_pred_std"] = sv[k].sqrt().numpy()
            npz[f"L{i+1}_{k}_corr_mean"] = sc[k].numpy(); npz[f"L{i+1}_{k}_eff_mean"] = se[k].numpy()
        npz[f"L{i+1}_pred_posmean"] = st[i]["pred_posmean"].numpy()
        # summary: share of |alpha| mass on the previous layer vs skips, and spread
        row = {"layer": i + 1}
        for k in STREAMS:
            e = se[k].abs()
            prev = e[..., -1].sum().item(); allm = e.sum().item()
            row[f"{k}_prev_share"] = prev / max(allm, 1e-9)
            row[f"{k}_n_active_per_head"] = float((e > 0.05 * e.max()).float().sum(-1).mean().item())
            row[f"{k}_pred_tok_std_over_mean"] = float((sv[k].sqrt().mean() / (sp[k].abs().mean() + 1e-9)).item())
            if an.has_corr:
                row[f"{k}_corr_tok_std_over_mean"] = float((an.split(cv, i)[k].sqrt().mean() / (sp[k].abs().mean() + 1e-9)).item())
        summ.append(row)
    np.savez_compressed(args.out + "_topology.npz", **npz)
    res["topology_summary"] = summ
    print("dump done", flush=True)

    # ---- J1: static substitution ----------------------------------------------------------
    pred_const = {k: [an.split(st[i]["pred_mean"], i)[k].view(1, 1, *an.split(st[i]["pred_mean"], i)[k].shape) for i in range(L - 1)] for k in STREAMS}
    pred_pos = {k: [an.split(st[i]["pred_posmean"], i)[k].unsqueeze(0) for i in range(L - 1)] for k in STREAMS}
    corr_const = [st[i]["corr_mean"].view(1, 1, -1) for i in range(L - 1)] if an.has_corr else None
    static = {}
    for name, ids in evals.items():
        r = {"intact": an.nll(ids), "pred_mean": an.nll(ids, pred_const=pred_const), "pred_posmean": an.nll(ids, pred_pos=pred_pos)}
        if an.has_corr:
            r["corr_mean"] = an.nll(ids, corr_const=corr_const)
            r["both_mean"] = an.nll(ids, pred_const=pred_const, corr_const=corr_const)
        static[name] = r
        print(name, {k: round(v, 4) for k, v in r.items()}, flush=True)
    res["static"] = static

    # ---- J4: prune hyperconnections of the static topology --------------------------------
    if not args.skip_prune:
        eff = [st[i]["pred_mean"] + (st[i]["corr_mean"] if an.has_corr else 0) for i in range(L - 1)]
        # hyperconnection = any source except the previous layer (last index of each src axis)
        is_hyper = []
        for i in range(L - 1):
            m = torch.ones_like(eff[i], dtype=torch.bool)
            sp = an.split(m, i)
            for k in STREAMS:
                sp[k][..., -1] = False
            is_hyper.append(torch.cat([sp[k].reshape(-1) for k in STREAMS]))
        mags = torch.cat([eff[i].abs()[is_hyper[i]] for i in range(L - 1)])
        prune = {}
        ev_name = "wikitext2" if "wikitext2" in evals else next(iter(evals))
        for frac in [float(x) for x in args.keep.split(",")]:
            thr = float("inf") if frac <= 0 else torch.quantile(mags, 1 - frac).item() if frac < 1 else -1.0
            masks = []
            for i in range(L - 1):
                keep = (~is_hyper[i]) | (eff[i].abs() >= thr)
                masks.append(keep.float())
            pm = [{k: v for k, v in an.split(masks[i], i).items()} for i in range(L - 1)]
            pred_mask = {k: [pm[i][k].view(1, 1, *pm[i][k].shape) for i in range(L - 1)] for k in STREAMS}
            prune[str(frac)] = an.nll(evals[ev_name], pred_mask=pred_mask, corr_mask=masks if an.has_corr else None)
            print(f"keep {frac}: {ev_name} nll {prune[str(frac)]:.4f}", flush=True)
        res["prune"] = {"eval": ev_name, "n_hyper_entries": int(mags.numel()), "nll_by_fraction_kept": prune,
                        "note": "intact per-token routing; entries ranked by mean |alpha| (pred+corr); previous-layer entries always kept"}

    with open(args.out + "_analysis.json", "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", args.out + "_analysis.json", flush=True)


if __name__ == "__main__":
    main()
