"""Stage 0 — can the routing weights move anything at all?

This runs *before* circuit discovery and can invalidate it, in the same way
that `extract_contrast.py`'s behaviour gap can.  Discovery and verification
both assume that editing alpha is a way to change the model's computation.
That assumption is not free, and this script measures it three ways.

**Is routing used?**  The sequential path (`s == l`, bias-initialised to 1.0)
and the hyperconnections (`s < l`) are reported separately.  If training left
almost all the mass on the sequential path, the model is a vanilla transformer
wearing a DAG and there are no hyperconnection circuits to find.

**Is routing dynamic?**  Alpha is compared across token positions and across
prompts.  A predictor that emits near-constant weights has learned a static
architecture, not a per-token topology, and "the routing the model chose for
this token" is not a meaningful object.

**Does routing have leverage?**  This is the one that explains the rest.  Alpha
does not inject signal; it re-weights a mixture of things the model already
computed:

    y = sum_s alpha_s x_s       x_s = W_stream^(l) X_s   (per head, in head_dim)

so d y / d alpha_s = x_s exactly, and the reachable set is the span of the
sources.  If the x_s are nearly collinear, changing alpha rescales y without
rotating it -- and for q/k that surviving magnitude change is then partly
removed by q_norm/k_norm, which are applied *after* mixing.  Two numbers per
edge follow directly:

    dir_leverage  ||(I - y_hat y_hat^T) x_s|| / ||y||   rotation per unit alpha
    mag_leverage  |y_hat . x_s| / ||y||                 rescale per unit alpha

With `--delta`, the measured behavioural difference is pushed through the same
algebra to answer the question the verification stage keeps running into:
*how far does the instruction actually rotate each head's input?*  If that is
a fraction of a percent, a flat steering result is a property of the
architecture at this checkpoint, not a bug in the intervention.

Usage:
    python experiments/interp/routing_leverage.py \
        --config .../config.yaml --ckpt .../checkpoint.pt \
        --behavior domain_code \
        --delta experiments/results/interp/circuits/circuit_domain_code_pred_stats.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from behaviors import get_behavior
from circuit_common import (Edge, RoutingLayout, RoutingRunner, build_prompt_set,
                            load_models, load_tokenizer, save_json)
from interp_common import unflatten_alpha

DEFAULT_TOKENIZER = "/work/hdd/bfqt/shared/dagformer-models/tokenizer"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--behavior", default="domain_code",
                    help="supplies the prompts; any behaviour's content works")
    ap.add_argument("--no-corr", action="store_true",
                    help="analyse the predictor's alpha alone. Off by default: "
                         "the model mixes with pred + corr, so the geometry "
                         "below is only the model's if corr is included")
    ap.add_argument("--delta", default=None,
                    help="*_stats.npz; push its Delta through the leverage "
                         "algebra to get the induced rotation per head")
    ap.add_argument("--out", default="experiments/results/interp/circuits/leverage")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-items", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Reconstructing the layer outputs that alpha mixes
# ---------------------------------------------------------------------------

class SourceCapture:
    """Rebuilds `layer_outputs` from module hooks.

    `FourWayDAGFormer.forward` keeps the stack of prior layer outputs in a
    local list, so it cannot be read from outside.  It can be reconstructed
    exactly, though, because every term is produced by a module that *is*
    called as a module:

        X_0     = embed_tokens(ids)
        X_{l+1} = (embedding if l == 0 else R_l) + attn_out_l + mlp_out_l

    with `attn_out_l = post_attention_layernorm(...)`, `mlp_out_l =
    post_feedforward_layernorm(...)` and `R_l = sum_s alpha_r[l][s] X_s`, which
    is the r-stream mixture and is recomputed here from the routing dict.
    """

    def __init__(self, fourway):
        self.fw = fourway
        self.h: list = []
        self.emb: torch.Tensor | None = None
        self.attn: dict[int, torch.Tensor] = {}
        self.mlp: dict[int, torch.Tensor] = {}

    def __enter__(self):
        m = self.fw.olmo.model
        self.h.append(m.embed_tokens.register_forward_hook(
            lambda _m, _i, o: self._set_emb(o)))
        for l, layer in enumerate(m.layers):
            self.h.append(layer.post_attention_layernorm.register_forward_hook(
                lambda _m, _i, o, l=l: self.attn.__setitem__(l, o.detach())))
            self.h.append(layer.post_feedforward_layernorm.register_forward_hook(
                lambda _m, _i, o, l=l: self.mlp.__setitem__(l, o.detach())))
        return self

    def _set_emb(self, o):
        self.emb = o.detach()

    def __exit__(self, *a):
        for h in self.h:
            h.remove()
        self.h = []

    def layer_outputs(self, routing: dict) -> list[torch.Tensor]:
        """[X_0 .. X_L], each [B, T, D]."""
        outs = [self.emb]
        for l in range(self.fw.num_layers):
            if l == 0:
                base = self.emb
            else:
                a_r = routing["r"][l - 1].to(outs[0].dtype)        # [B, T, n_src]
                stack = torch.stack(outs, dim=0)                    # [n_src, B, T, D]
                base = torch.einsum("lbtd,btl->btd", stack, a_r)
            outs.append(base + self.attn[l] + self.mlp[l])
        return outs


def head_weights(fourway, l: int) -> dict:
    w = fourway._get_head_weight_views(l)
    return {"q": w["W_q"], "k": w["W_k"], "v": w["W_v"]}


# ---------------------------------------------------------------------------
# Leverage
# ---------------------------------------------------------------------------

def leverage_for_layer(outs: list[torch.Tensor], routing: dict, fourway,
                       l: int, lo: int, hi: int) -> dict:
    """Per-edge leverage at target layer `l`, averaged over the content span.

    Returns arrays keyed by stream: q/k/v are [H, n_src], r is [n_src].
    """
    n_src = l + 1
    stack = torch.stack(outs[:n_src], dim=0)[:, :, lo:hi]     # [S, B, t, D]
    W = head_weights(fourway, l)
    out: dict[str, np.ndarray] = {}
    for stream in ("q", "k", "v"):
        # project every source with this layer's per-head weights
        x = torch.einsum("sbtd,hod->sbtho", stack.float(), W[stream].float())
        a = routing[stream][l - 1][:, lo:hi].float()          # [B, t, H, S]
        a = a[..., :n_src].permute(3, 0, 1, 2)                # [S, B, t, H]
        y = (x * a.unsqueeze(-1)).sum(0)                      # [B, t, H, hd]
        out[stream] = _edge_stats(x, y)
    # r stream: mixing happens in model_dim, one weight per source (no heads)
    xr = stack.float().unsqueeze(3)                           # [S, B, t, 1, D]
    ar = routing["r"][l - 1][:, lo:hi].float()[..., :n_src]   # [B, t, S]
    yr = (xr * ar.permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1)).sum(0)
    out["r"] = _edge_stats(xr, yr)[0]
    out["_eff_rank"] = {s: _eff_rank(
        torch.einsum("sbtd,hod->sbtho", stack.float(), W[s].float())) for s in ("q", "k", "v")}
    out["_eff_rank"]["r"] = _eff_rank(xr)
    return out


def _edge_stats(x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
    """[S,B,t,H,d] sources and [B,t,H,d] mixture -> [H, S, 3] leverage stats.

    Columns: dir_leverage, mag_leverage, |x_s| / |y|.
    """
    ny = y.norm(dim=-1).clamp_min(1e-9)                       # [B, t, H]
    yhat = y / ny.unsqueeze(-1)
    proj = (x * yhat.unsqueeze(0)).sum(-1)                    # [S, B, t, H]
    perp = (x - proj.unsqueeze(-1) * yhat.unsqueeze(0)).norm(dim=-1)
    nx = x.norm(dim=-1)
    stats = torch.stack([perp / ny, proj.abs() / ny, nx / ny], dim=-1)
    return stats.mean(dim=(1, 2)).permute(1, 0, 2).cpu().numpy()   # [H, S, 3]


def _eff_rank(x: torch.Tensor) -> float:
    """Participation ratio of the source set's singular values, averaged.

    1.0 means every source projects to the same direction, so alpha can only
    rescale the mixture; n_src means the sources are mutually orthogonal and
    alpha has full control of the head's input within their span.
    """
    S = x.shape[0]
    if S < 2:
        return 1.0
    m = x.permute(1, 2, 3, 0, 4).reshape(-1, S, x.shape[-1])   # [N, S, d]
    m = m[torch.randperm(m.shape[0])[:512]]
    sv = torch.linalg.svdvals(m.double())
    pr = (sv.sum(-1) ** 2) / (sv ** 2).sum(-1).clamp_min(1e-30)
    return float(pr.mean())


# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg, fourway, predictor = load_models(args.config, args.ckpt, device, need_base=True)
    layout = RoutingLayout(cfg["num_hidden_layers"], cfg["num_attention_heads"])
    tokenizer = load_tokenizer(cfg, args.tokenizer)
    spec = get_behavior(args.behavior)
    ps = build_prompt_set(spec, tokenizer, max_items=args.max_items)
    lo, hi = ps.span
    runner = RoutingRunner(layout, predictor, fourway, device=device)
    ids = torch.tensor([p.ids for p in ps.prompts], dtype=torch.long)
    print(f"[leverage] {len(ps.prompts)} prompts, span [{lo},{hi}), "
          f"L={layout.L} H={layout.H} D={layout.D}")

    # --- alpha: how much mass, how much movement -------------------------
    use_corr = runner.has_corr and not args.no_corr
    chans: dict[str, list] = {"pred": [], "corr": []}
    for i in range(0, ids.shape[0], args.batch_size):
        chunk = ids[i:i + args.batch_size]
        chans["pred"].append(runner.alpha_pred(chunk))
        if runner.has_corr:
            chans["corr"].append(runner.forward_capture(chunk.to(device))[1])
    A_pred = torch.cat(chans["pred"], 0)
    A_corr = torch.cat(chans["corr"], 0) if runner.has_corr else torch.zeros_like(A_pred)
    seq = ~layout.hyper_arr
    res: dict = {"behavior": args.behavior, "with_corr": use_corr,
                 "n_prompts": int(ids.shape[0]), "span": [lo, hi],
                 "layout": {"L": layout.L, "H": layout.H, "D": layout.D}}
    res["mass"], res["dynamics"] = {}, {}
    for name, T in (("pred", A_pred), ("corr", A_corr), ("eff", A_pred + A_corr)):
        A = T[:, lo:hi].numpy()
        res["mass"][name] = {
            "mean_abs_sequential": float(np.abs(A[:, :, seq]).mean()),
            "mean_abs_hyper": float(np.abs(A[:, :, ~seq]).mean()),
            "hyper_share_of_mass": float(np.abs(A[:, :, ~seq]).sum()
                                         / max(np.abs(A).sum(), 1e-12))}
        scale = np.abs(A).mean(axis=(0, 1)) + 1e-12
        res["dynamics"][name] = {
            "across_tokens_over_scale": float((A.std(axis=1).mean(0) / scale).mean()),
            "across_prompts_over_scale": float((A.mean(1).std(0) / scale).mean())}
        print(f"[{name}] hyper share {res['mass'][name]['hyper_share_of_mass']:.4f}  "
              f"|a| seq {res['mass'][name]['mean_abs_sequential']:.4f} "
              f"hyper {res['mass'][name]['mean_abs_hyper']:.4f}  |  varies across "
              f"tokens {res['dynamics'][name]['across_tokens_over_scale']:.4f} "
              f"prompts {res['dynamics'][name]['across_prompts_over_scale']:.4f}")
    res["mass"]["n_sequential"] = int(seq.sum())
    res["mass"]["n_hyper"] = int((~seq).sum())
    res["dynamics"]["note"] = ("std of alpha relative to its own mean |alpha|; "
                               "~0 means a static architecture, not a per-token "
                               "topology")

    # --- leverage --------------------------------------------------------
    dir_lev = np.zeros(layout.D)
    mag_lev = np.zeros(layout.D)
    norm_ratio = np.zeros(layout.D)
    eff_rank: dict = {}
    nb, recon_err = 0, 0.0
    for i in range(0, ids.shape[0], args.batch_size):
        chunk = ids[i:i + args.batch_size].to(device)
        with SourceCapture(fourway) as cap, torch.no_grad():
            routing = runner.routing(chunk)
            logits = fourway(chunk, routing)
            if use_corr:
                corr = A_corr[i:i + args.batch_size].to(device)
                eff = unflatten_alpha(
                    torch.cat([runner.alpha_pred(chunk).to(device) + corr], 0),
                    layout.L, layout.H)
                routing = {s: [t.to(device) for t in v] for s, v in eff.items()}
            outs = cap.layer_outputs(routing)
            # the reconstruction is only worth anything if it reproduces the model
            chk = fourway.olmo.lm_head(fourway.olmo.model.norm(outs[-1]))
            recon_err = max(recon_err, float(
                (chk - logits).abs().max() / logits.abs().max().clamp_min(1e-6)))
        for l in range(1, layout.L):
            st = leverage_for_layer(outs, routing, fourway, l, lo, hi)
            for stream in ("q", "k", "v", "r"):
                arr = st[stream]
                for h in range(layout.H if stream != "r" else 1):
                    row = arr[h] if stream != "r" else arr
                    for s in range(l + 1):
                        j = layout.index_of(Edge(stream=stream, layer=l,
                                                 head=h if stream != "r" else -1, src=s))
                        dir_lev[j] += row[s, 0]
                        mag_lev[j] += row[s, 1]
                        norm_ratio[j] += row[s, 2]
            for stream, v in st["_eff_rank"].items():
                eff_rank.setdefault((l, stream), []).append(v)
        nb += 1
        print(f"  batch {nb} done", flush=True)
    dir_lev /= nb; mag_lev /= nb; norm_ratio /= nb
    print(f"[check] layer-output reconstruction vs model logits: "
          f"max rel err {recon_err:.2e}")
    assert recon_err < 5e-2, (
        f"reconstructed layer outputs do not reproduce the model "
        f"(rel err {recon_err:.3g}); the leverage numbers below would be wrong")

    res["reconstruction_rel_err"] = recon_err
    res["leverage"] = {
        "dir_mean": float(dir_lev.mean()), "mag_mean": float(mag_lev.mean()),
        "dir_mean_hyper": float(dir_lev[~seq].mean()),
        "dir_mean_sequential": float(dir_lev[seq].mean()),
        "by_stream": {s: {"dir": float(dir_lev[layout.stream_arr == s].mean()),
                          "mag": float(mag_lev[layout.stream_arr == s].mean())}
                      for s in ("q", "k", "v", "r")},
        "eff_rank_by_layer": {f"L{l}/{s}": float(np.mean(v))
                              for (l, s), v in sorted(eff_rank.items())},
        "note": "dir = rotation of the head input per unit alpha, relative to "
                "the mixture norm; mag = rescale per unit alpha"}
    print(f"[leverage] dir {res['leverage']['dir_mean']:.4f}  "
          f"mag {res['leverage']['mag_mean']:.4f}")

    # --- what the instruction actually does ------------------------------
    if args.delta:
        z = np.load(args.delta)
        delta = z["delta_all"].astype(np.float64)
        elig = z["eligible"].astype(bool)
        induced = np.abs(delta) * dir_lev
        per_head: dict[str, float] = {}
        for l in range(1, layout.L):
            for s in ("q", "k", "v"):
                m = (layout.layer_arr == l) & (layout.stream_arr == s)
                # edges into one head add in quadrature if independent
                vals = [np.sqrt((induced[m & (layout.head_arr == h)] ** 2).sum())
                        for h in range(layout.H)]
                per_head[f"L{l}/{s}"] = float(np.mean(vals))
        res["induced_rotation"] = {
            "mean_over_eligible": float(induced[elig].mean()),
            "rms_per_head_mean": float(np.mean(list(per_head.values()))),
            "rms_per_head_max": float(np.max(list(per_head.values()))),
            "per_layer_stream": per_head,
            "source": args.delta,
            "note": "relative rotation of a head's mixed input caused by the "
                    "measured instruction difference; 0.01 = 1%"}
        print(f"[induced] mean per-head rotation "
              f"{res['induced_rotation']['rms_per_head_mean']:.4f} "
              f"(max {res['induced_rotation']['rms_per_head_max']:.4f})")

    out = Path(args.out)
    np.savez(out.with_suffix(".npz"), dir_leverage=dir_lev, mag_leverage=mag_lev,
             norm_ratio=norm_ratio)
    save_json(res, out.with_suffix(".json"))


if __name__ == "__main__":
    main()
