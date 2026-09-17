"""E0 — would this pipeline find a circuit if one were there?

Every negative result in this directory rests on an assumption nobody has
checked: that the instruments have the power to detect a real effect of the
size a real behaviour would produce.  "We found nothing" and "we cannot see
anything this small" are very different claims, and until the second is ruled
out the first is not reportable.

So this plants effects of *known* size into the real measurement noise and
finds the floor at which each stage stops working.  Two floors, because the
pipeline has two places to fail:

`--part select`   Can the selection rule recover a known sparse Delta?
    The noise is real: balanced-split draws from the recorded contrast, which
    carry the actual item and paraphrase variation of the checkpoint.  A
    sparse signal of magnitude m is added on top, and precision/recall against
    the planted support are swept over m.  CPU only.

`--part sensitivity`   How big an alpha edit does the *behaviour* need?
    Random sparse edits of magnitude m are applied to the routing weights and
    the behaviour score is measured against the unedited run, together with
    the held-out NLL cost.  This is the power curve of the verification stage
    and needs the model.

Both floors are reported in the same unit as `routing_leverage.py`'s induced
rotation -- the relative rotation a perturbation causes in a head's mixed
input -- so they can be compared directly against what the real instruction
does.  On the 300M checkpoint that number is 0.0033 for `pred`, and the point
of this script is to say whether 0.0033 was ever detectable.

Usage:
    python experiments/interp/planted_circuit.py --part select \
        --contrast /tmp/interp_final/contrast_domain_code.pt \
        --stats    .../circuit_domain_code_pred_stats.npz \
        --leverage .../leverage_domain_code_pred.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from behaviors import get_behavior
from circuit_common import (Edit, RoutingLayout, RoutingRunner, build_prompt_set,
                            capability_nll, capability_windows, load_models,
                            load_tokenizer, save_json, score_behavior)
from discover_circuit import balanced_null, group_alpha, null_context, select

DEFAULT_TOKENIZER = "/work/hdd/bfqt/shared/dagformer-models/tokenizer"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="select", choices=("select", "sensitivity"))
    ap.add_argument("--stats", required=True, help="*_stats.npz (supplies `eligible`)")
    ap.add_argument("--layers", type=int, default=12,
                    help="fallback L when neither --contrast nor --config is given")
    ap.add_argument("--heads", type=int, default=16, help="fallback H")
    ap.add_argument("--leverage", default=None,
                    help="leverage_*.npz from routing_leverage.py; converts the "
                         "planted magnitude into induced head rotation")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    # -- select --
    ap.add_argument("--contrast", default=None, help="contrast_*.pt (select part)")
    ap.add_argument("--channel", default="pred", choices=("pred", "corr", "eff"))
    ap.add_argument("--k", type=int, nargs="+", default=[16, 64, 256],
                    help="planted circuit sizes")
    ap.add_argument("--mags", type=float, nargs="+",
                    default=[0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05],
                    help="planted per-edge magnitude, in raw alpha units")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--q", type=float, default=0.005)
    ap.add_argument("--n-est", type=int, default=24)
    ap.add_argument("--n-cal", type=int, default=24)
    # -- sensitivity --
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--behavior", default="domain_code")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-repeats", type=int, default=3,
                    help="random edge sets per (k, magnitude) cell")
    ap.add_argument("--cap-windows", type=int, default=4)
    ap.add_argument("--cap-seq-len", type=int, default=256)
    ap.add_argument("--max-nll-rise", type=float, default=0.05,
                    help="damage gate, matching verify_circuit.py")
    return ap.parse_args()


# ---------------------------------------------------------------------------

def plant(eligible: np.ndarray, k: int, mag: float,
          rng: np.random.Generator) -> np.ndarray:
    """A sparse signed vector of magnitude `mag` on `k` random eligible edges."""
    v = np.zeros(eligible.size)
    idx = rng.choice(np.flatnonzero(eligible), size=k, replace=False)
    v[idx] = mag * rng.choice([-1.0, 1.0], size=k)
    return v


def induced_rotation(layout: RoutingLayout, signal: np.ndarray,
                     dir_lev: np.ndarray | None) -> float:
    """Mean per-head rotation caused by `signal`, matching routing_leverage.py.

    Edges into the same head are summed in quadrature; the mean is taken over
    every (layer, stream) head group so the number is comparable across
    circuits of different shapes.
    """
    if dir_lev is None:
        return float("nan")
    induced = np.abs(signal) * dir_lev
    per = []
    for l in range(1, layout.L):
        for s in ("q", "k", "v"):
            m = (layout.layer_arr == l) & (layout.stream_arr == s)
            per += [np.sqrt((induced[m & (layout.head_arr == h)] ** 2).sum())
                    for h in range(layout.H)]
    return float(np.mean(per))


def part_select(args, layout: RoutingLayout, eligible: np.ndarray,
                dir_lev: np.ndarray | None, rng: np.random.Generator) -> dict:
    """Sweep planted magnitude; report recovery by the real selection rule."""
    assert args.contrast, "--part select needs --contrast"
    d = torch.load(args.contrast, map_location="cpu", weights_only=False)
    alpha, var, n_items = d["alpha"][args.channel], d["variants"], d["n_items"]
    pos = group_alpha(alpha, d["meta"], "pos", n_items, var["pos"])
    neg = group_alpha(alpha, d["meta"], "neg", n_items, var["neg"])
    tr = np.arange(n_items)

    need = args.n_est + args.n_cal + args.n_trials
    draws = balanced_null(pos, neg, need, rng)
    est, cal = draws[:args.n_est], draws[args.n_est:args.n_est + args.n_cal]
    noise = draws[args.n_est + args.n_cal:]
    ctx = null_context(est, cal, tr, eligible, args.q)
    sel_args = SimpleNamespace(select="null", q=args.q, min_abs=0.0)

    # what the rule does on the noise alone, for reference
    base = [int(select(n[tr].mean(0), None, None, None, sel_args, eligible, ctx)[0].sum())
            for n in noise]

    rows = []
    for k in args.k:
        for mag in args.mags:
            prec, rec, nsel = [], [], []
            for t in range(args.n_trials):
                sig = plant(eligible, k, mag, rng)
                X = noise[t] + sig                       # real noise + known signal
                chosen, _ = select(X[tr].mean(0), None, None, None,
                                   sel_args, eligible, ctx)
                truth = sig != 0
                hit = int((chosen & truth).sum())
                prec.append(hit / max(int(chosen.sum()), 1))
                rec.append(hit / k)
                nsel.append(int(chosen.sum()))
            rows.append({"k": k, "magnitude": mag,
                         "induced_rotation": induced_rotation(layout, plant(
                             eligible, k, mag, np.random.default_rng(0)), dir_lev),
                         "precision": float(np.mean(prec)),
                         "recall": float(np.mean(rec)),
                         "n_selected": float(np.mean(nsel))})
            print(f"  k={k:4d} mag={mag:<7g} rot={rows[-1]['induced_rotation']:.5f} "
                  f"recall={rows[-1]['recall']:.2f} prec={rows[-1]['precision']:.2f} "
                  f"nsel={rows[-1]['n_selected']:.0f}", flush=True)
    return {"rows": rows, "n_selected_on_noise_alone": float(np.mean(base)),
            "s_cutoff": ctx["thr"], "channel": args.channel,
            "note": "recall is the fraction of planted edges the rule recovers; "
                    "the detection floor is the smallest induced rotation with "
                    "recall >= 0.5"}


def part_sensitivity(args, layout: RoutingLayout, eligible: np.ndarray,
                     dir_lev: np.ndarray | None, rng: np.random.Generator) -> dict:
    """Sweep planted magnitude; report the behaviour shift the model shows."""
    assert args.config and args.ckpt, "--part sensitivity needs --config/--ckpt"
    device = torch.device(args.device)
    cfg, fourway, predictor = load_models(args.config, args.ckpt, device)
    assert (cfg["num_hidden_layers"], cfg["num_attention_heads"]) == (layout.L, layout.H), \
        f"--layers/--heads {layout.L}/{layout.H} disagree with the checkpoint config"
    tokenizer = load_tokenizer(cfg, args.tokenizer)
    spec = get_behavior(args.behavior)
    ps = build_prompt_set(spec, tokenizer, max_items=args.max_items)
    runner = RoutingRunner(layout, predictor, fourway, device=device)
    neutral = ps.by("neutral")
    windows = capability_windows(tokenizer, args.cap_seq_len, args.cap_windows)

    base = score_behavior(runner, spec, neutral, tokenizer, ps.filler_id,
                          batch_size=args.batch_size)
    nll_ref = capability_nll(runner, windows)
    print(f"[sensitivity] {len(neutral)} neutral prompts; baseline score "
          f"{base['score']:.4f} (sem {base['sem']:.4f}) nll {nll_ref:.4f}", flush=True)

    rows = []
    for k in args.k:
        for mag in args.mags:
            shifts, nlls, rots = [], [], []
            for _ in range(args.n_repeats):
                sig = plant(eligible, k, mag, rng)
                edit = Edit(op="add", mask=torch.from_numpy(sig != 0),
                            delta=torch.from_numpy(sig).float(), lam=1.0)
                with runner.edited(pred=edit):
                    s = score_behavior(runner, spec, neutral, tokenizer,
                                       ps.filler_id, batch_size=args.batch_size)
                    nlls.append(capability_nll(runner, windows))
                shifts.append(s["score"] - base["score"])
                rots.append(induced_rotation(layout, sig, dir_lev))
            rows.append({"k": k, "magnitude": mag,
                         "induced_rotation": float(np.mean(rots)),
                         "abs_shift": float(np.mean(np.abs(shifts))),
                         "shift_over_se": float(np.mean(np.abs(shifts)) / max(base["sem"], 1e-9)),
                         "nll_rise": float(np.mean(nlls)) - nll_ref})
            print(f"  k={k:4d} mag={mag:<7g} rot={rows[-1]['induced_rotation']:.5f} "
                  f"|shift|={rows[-1]['abs_shift']:.4f} "
                  f"({rows[-1]['shift_over_se']:.1f} se)  "
                  f"dNLL={rows[-1]['nll_rise']:+.3f}", flush=True)
    return {"rows": rows, "baseline_score": base["score"], "baseline_se": base["sem"],
            "baseline_nll": nll_ref, "behavior": args.behavior,
            "max_nll_rise": args.max_nll_rise,
            "note": "the floor is the smallest induced rotation whose behaviour "
                    "shift clears 2 se without a material NLL rise"}


def report(res: dict, part: str) -> list[str]:
    L = [f"# Planted-circuit power analysis ({part})", ""]
    if part == "select":
        L += [f"Selection rule admits {res['n_selected_on_noise_alone']:.0f} edges "
              f"from noise alone (S cutoff {res['s_cutoff']:.2f}).", "",
              "| k | magnitude | induced rotation | recall | precision | selected |",
              "|---|---|---|---|---|---|"]
        for r in res["rows"]:
            L.append(f"| {r['k']} | {r['magnitude']:g} | {r['induced_rotation']:.5f} | "
                     f"{r['recall']:.2f} | {r['precision']:.2f} | {r['n_selected']:.0f} |")
    else:
        L += [f"Baseline score {res['baseline_score']:.4f} (se {res['baseline_se']:.4f}), "
              f"NLL {res['baseline_nll']:.4f}.", "",
              "| k | magnitude | induced rotation | \\|shift\\| | shift/se | dNLL |",
              "|---|---|---|---|---|---|"]
        for r in res["rows"]:
            L.append(f"| {r['k']} | {r['magnitude']:g} | {r['induced_rotation']:.5f} | "
                     f"{r['abs_shift']:.4f} | {r['shift_over_se']:.1f} | "
                     f"{r['nll_rise']:+.3f} |")
    if part == "select":
        ok = [r for r in res["rows"] if r["recall"] >= 0.5]
        crit = "recall >= 0.5"
    else:
        gate = res["max_nll_rise"]
        ok = [r for r in res["rows"]
              if r["shift_over_se"] >= 2.0 and r["nll_rise"] <= gate]
        crit = f"shift >= 2 se with dNLL <= {gate} (rows above the gate do not count: "
        crit += "they move the score by destroying the model, not by steering it)"
    floor = min((r["induced_rotation"] for r in ok), default=None)
    L += ["", f"Criterion: {crit}.", ""]
    if floor is None:
        L += ["**Nothing in the swept range is detectable.** Whatever the real "
              "instruction does, this stage could not have seen it.", ""]
    else:
        L += [f"**Detection floor: induced rotation {floor:.5f}.** The real "
              f"instruction produces 0.0033 on `pred`.", "",
              ("Well below the real effect, so a null result from this stage is "
               "a real absence and not an instrument limit."
               if floor < 0.0033 else
               "**At or above the real effect — a null result from this stage is "
               "at least partly an instrument limit and must be reported as "
               "such.**"), ""]
    if part == "sensitivity":
        L += ["The planted edits point in random directions, which is the worst "
              "case: a genuinely behavioural direction of the same magnitude "
              "should do better. So this floor bounds the verification stage "
              "from above rather than pinning it.", ""]
    return L


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    z = np.load(args.stats)
    eligible = z["eligible"].astype(bool)
    layout = RoutingLayout(args.layers, args.heads)
    assert layout.D == eligible.size, (
        f"layout D={layout.D} != {eligible.size} from --stats; "
        f"set --layers/--heads to match the checkpoint")
    dir_lev = None
    if args.leverage:
        dir_lev = np.load(args.leverage)["dir_leverage"]
    else:
        print("[warn] no --leverage: rotations will be nan")

    res = (part_select if args.part == "select" else part_sensitivity)(
        args, layout, eligible, dir_lev, rng)
    res.update(part=args.part, stats=args.stats, leverage=args.leverage)
    text = "\n".join(report(res, args.part))
    print(text)
    if args.out:
        out = Path(args.out)
        save_json(res, out.with_suffix(".json"))
        out.with_suffix(".md").write_text(text)
        print(f"[saved] {out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
