"""Step 3 — does the circuit actually *cause* the behaviour?

Step 2 hands over a set of routing coordinates whose weights differ between the
two instructions.  That is a correlational claim.  This script intervenes on
those coordinates and asks whether the behaviour moves, under arms designed so
that a positive result cannot be explained by the boring alternatives:

    sufficiency   On *neutral* prompts, push the circuit toward the pos
                  instruction and see whether the behaviour follows.
                    add           alpha[M] += lam * Delta[M]
                    scale         alpha[M] *= (1 + lam)   <- "enlarge the wiring"
                    signed_scale  alpha[M] += lam*|alpha[M]|*sign(Delta[M])

    patching      Run the *neg* prompt but copy the circuit's routing values
                  from a rerun of the same tokens under the pos instruction.
                  This uses the donor's observed magnitude. A full-mask patch
                  tests the whole eligible set; subset effects may be larger
                  when contributions in the full set cancel.

    necessity     Zero the circuit under the pos instruction.

Controls, all scored on the same prompts with the same metric:

    dense         the same edit over *every* eligible coordinate.  A previous
                  run of this model (experiments/results/interp/steering2_*)
                  found dense alpha-side injection to be a literal no-op, so
                  without this arm a null circuit result is unreadable: you
                  could not tell "this circuit is not causal" from "nothing on
                  the alpha side does anything".
    random        a size-matched random set of eligible coordinates.
    matched       random, but matched to the circuit's (stream, layer) profile
                  — controls for "layer 11 v-stream edits just do more".
    complement    the next-strongest coordinates that the rule did *not* pick.
    per-stream    the circuit restricted to q / k / v / r.  Worth reading
                  carefully: q and k are RMSNorm-ed after mixing, so uniform
                  magnitude changes on those streams are partly normalised
                  away, while v and r (use_v_norm is off here) are not.

By default every control is renormalised to the circuit's ||Delta[M]||_2, so
"the circuit wins" cannot just mean "the circuit has bigger numbers in it".

Additive, scaling and knockout arms also report natural-text NLL.
Instruction-patching arms are evaluated only on the contrast prompts.

Selection used the train-item half; scoring here defaults to the held-out half.

Usage:
    python experiments/interp/verify_circuit.py \
        --config .../config.yaml --ckpt .../checkpoint.pt \
        --behavior honesty --channel pred
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from behaviors import spec_from_dict
from circuit_common import (Edit, RoutingLayout, RoutingRunner, build_prompt_set,
                            capability_nll, capability_windows, load_models,
                            load_tokenizer, save_json, score_behavior)

DEFAULT_TOKENIZER = "/work/hdd/bfqt/shared/dagformer-models/tokenizer"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--behavior", default="honesty")
    ap.add_argument("--channel", default="pred", choices=("pred", "corr", "eff"),
                    help="which channel the circuit was discovered on")
    ap.add_argument("--edit-channel", default=None, choices=("pred", "corr"),
                    help="which channel to intervene on "
                         "(default: the discovery channel; 'eff' maps to 'pred')")
    ap.add_argument("--in-dir", default="experiments/results/interp/circuits")
    ap.add_argument("--out-dir", default="experiments/results/interp/circuits")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="auto", choices=("auto", "bf16", "fp32"))
    ap.add_argument("--batch-size", type=int, default=8)

    ap.add_argument("--items", default="test", choices=("test", "train", "all"),
                    help="which item half to score on; 'test' is the half the "
                         "circuit was NOT selected on")
    ap.add_argument("--delta", default="train", choices=("train", "all"),
                    help="which Delta estimate supplies the steering direction")
    ap.add_argument("--add-lams", type=float, nargs="*",
                    default=[1.0, 4.0, 16.0, 64.0],
                    help="additive arms use alpha += lam * Delta. The grid runs "
                         "this high because Delta can be a tiny fraction of "
                         "alpha; the report prints the lambda at which the two "
                         "are comparable, and an additive sweep that never "
                         "reaches it has not tested anything")
    ap.add_argument("--scale-lams", type=float, nargs="*", default=[1.0, 3.0, 9.0],
                    help="multiplicative arms use alpha *= (1 + lam)")
    ap.add_argument("--max-nll-rise", type=float, default=0.05,
                    help="an arm whose held-out NLL rises by more than this "
                         "many nats is marked damaged: it cannot be the best "
                         "steering arm, and it cannot be the control that "
                         "declares a circuit non-specific")
    ap.add_argument("--edit-from", default="content", choices=("content", "all"),
                    help="apply edits from the start of the content span "
                         "(default) or at every position")
    ap.add_argument("--no-match-norm", action="store_true",
                    help="do not renormalise control deltas to the circuit's norm")
    ap.add_argument("--cross-circuit", default=None,
                    help="another behaviour's *_stats.npz; adds a specificity "
                         "arm that steers with that circuit and scores THIS "
                         "behaviour. If it works as well, neither is specific.")
    ap.add_argument("--skip-controls", action="store_true")
    ap.add_argument("--skip-patch", action="store_true")
    ap.add_argument("--skip-streams", action="store_true")
    ap.add_argument("--donor-variant", type=int, default=0,
                    help="instruction paraphrase used as the patching donor")
    ap.add_argument("--cap-windows", type=int, default=8)
    ap.add_argument("--cap-seq-len", type=int, default=256)
    ap.add_argument("--cap-cache", default=None,
                    help="cached eval-batch .pt for the capability check")
    ap.add_argument("--skip-capability", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Control masks
# ---------------------------------------------------------------------------

def random_mask(eligible: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    pool = np.nonzero(eligible)[0]
    n = min(n, pool.size)
    m = np.zeros(eligible.shape, dtype=bool)
    m[rng.choice(pool, size=n, replace=False)] = True
    return m


def matched_random_mask(layout: RoutingLayout, chosen: np.ndarray,
                        eligible: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random edges with the circuit's (stream, layer) marginal profile.

    Controls for the possibility that the circuit is nothing but "edits at
    layer 11 on the v stream move the model more than edits at layer 2 do".
    """
    m = np.zeros(chosen.shape, dtype=bool)
    key = np.char.add(layout.stream_arr.astype(str),
                      np.char.add("/", layout.layer_arr.astype(str)))
    for k in np.unique(key[chosen]):
        want = int((key[chosen] == k).sum())
        pool = np.nonzero(eligible & (key == k))[0]
        take = min(want, pool.size)
        if take:
            m[rng.choice(pool, size=take, replace=False)] = True
    return m


def complement_mask(delta: np.ndarray, chosen: np.ndarray,
                    eligible: np.ndarray) -> np.ndarray:
    """The next-strongest eligible coordinates the selection rule rejected."""
    pool = np.nonzero(eligible & ~chosen)[0]
    n = min(int(chosen.sum()), pool.size)
    top = pool[np.argsort(-np.abs(delta[pool]))[:n]]
    m = np.zeros(chosen.shape, dtype=bool)
    m[top] = True
    return m


def norm_matched_delta(delta: np.ndarray, mask: np.ndarray,
                       ref_mask: np.ndarray) -> np.ndarray:
    """delta restricted to `mask`, rescaled to ||delta[ref_mask]||_2."""
    d = delta * mask
    n, ref = np.linalg.norm(d), np.linalg.norm(delta * ref_mask)
    return d * (ref / n) if n > 1e-12 else d


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

@dataclass
class Arm:
    name: str
    group: str
    polarity: str                       # which prompts to score on
    baseline: str                       # arm name this is a shift relative to
    edit: Optional[Edit]
    lam: float = 0.0
    n_edges: int = 0
    capability: bool = True
    note: str = ""
    # +1 if a working intervention should move the behaviour toward the pos
    # instruction, -1 if toward neg (knockouts, and patching neg into pos).
    expect: float = 1.0
    result: dict = field(default_factory=dict)


def make_donor(runner: RoutingRunner, prefix: torch.Tensor,
               prefix_len: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Donor routing = the same tokens read under a different instruction.

    Legitimate only because the prompt set is position-aligned: swapping the
    first `prefix_len` ids changes the instruction and nothing else, so donor
    and recipient routing tensors are index-comparable position by position.
    """
    def donor(ids: torch.Tensor) -> torch.Tensor:
        d = ids.clone()
        d[:, :prefix_len] = prefix.to(d.device).view(1, -1)
        return runner.alpha_pred(d)
    return donor


def build_arms(args, layout, chosen, eligible, delta, ps, runner, rng,
               edit_channel: str, pos_mask: torch.Tensor,
               cross: Optional[tuple[str, np.ndarray, np.ndarray]] = None) -> list[Arm]:
    n_sel = int(chosen.sum())
    t_chosen = torch.from_numpy(chosen)
    t_elig = torch.from_numpy(eligible)
    t_delta = torch.from_numpy(delta.astype(np.float32))
    match = not args.no_match_norm

    def add_edit(mask_np: np.ndarray, lam: float, matched: bool) -> Edit:
        d = norm_matched_delta(delta, mask_np, chosen) if (matched and match) \
            else delta * mask_np
        return Edit(op="add", mask=torch.from_numpy(mask_np),
                    lam=lam, delta=torch.from_numpy(d.astype(np.float32)),
                    pos_mask=pos_mask)

    arms: list[Arm] = [
        Arm("ref_neutral", "reference", "neutral", "ref_neutral", None,
            note="no intervention; the starting point for the steering arms"),
        Arm("ref_pos", "reference", "pos", "ref_neutral", None,
            note="pos instruction, no intervention"),
        Arm("ref_neg", "reference", "neg", "ref_neutral", None,
            note="neg instruction, no intervention"),
    ]

    # -- sufficiency: steer neutral prompts toward the pos instruction ------
    for lam in args.add_lams:
        arms.append(Arm(f"circuit_add@{lam:g}", "sufficiency", "neutral",
                        "ref_neutral", add_edit(chosen, lam, False), lam, n_sel))
        if args.skip_controls:
            continue
        arms.append(Arm(f"dense_add@{lam:g}", "control", "neutral", "ref_neutral",
                        add_edit(eligible, lam, False), lam, int(eligible.sum()),
                        note="every eligible coordinate; was a no-op in steering2"))
        for tag, m in (("random", random_mask(eligible, n_sel, rng)),
                       ("matched", matched_random_mask(layout, chosen, eligible, rng)),
                       ("complement", complement_mask(delta, chosen, eligible))):
            arms.append(Arm(f"{tag}_add@{lam:g}", "control", "neutral", "ref_neutral",
                            add_edit(m, lam, True), lam, int(m.sum()),
                            note="norm-matched to the circuit" if match else ""))
        if cross is not None:
            cname, cmask, cdelta = cross
            d = cdelta * cmask
            n = np.linalg.norm(d)
            if match and n > 1e-12:
                d = d * (np.linalg.norm(delta * chosen) / n)
            arms.append(Arm(f"cross_{cname}_add@{lam:g}", "control", "neutral",
                            "ref_neutral",
                            Edit(op="add", mask=torch.from_numpy(cmask), lam=lam,
                                 delta=torch.from_numpy(d.astype(np.float32)),
                                 pos_mask=pos_mask),
                            lam, int(cmask.sum()),
                            note=f"{cname}'s circuit steering this behaviour"))

    # -- the user's literal proposal: enlarge the connection magnitudes -----
    for lam in args.scale_lams:
        arms.append(Arm(f"circuit_scale@{lam:g}", "amplify", "neutral", "ref_neutral",
                        Edit(op="scale", mask=t_chosen, lam=lam, pos_mask=pos_mask),
                        lam, n_sel, note=f"alpha *= {1 + lam:g} on the circuit"))
        arms.append(Arm(f"circuit_signed_scale@{lam:g}", "amplify", "neutral",
                        "ref_neutral",
                        Edit(op="signed_scale", mask=t_chosen, lam=lam,
                             delta=t_delta, pos_mask=pos_mask),
                        lam, n_sel, note="amplify along sign(Delta)"))
        if not args.skip_controls:
            arms.append(Arm(f"dense_scale@{lam:g}", "control", "neutral",
                            "ref_neutral",
                            Edit(op="scale", mask=t_elig, lam=lam, pos_mask=pos_mask),
                            lam, int(eligible.sum())))
            # Multiplicative arms cannot be norm-matched the way additive ones
            # are -- the edit size is set by whatever alpha already holds -- so
            # the size-matched question "would scaling *any* n_sel edges do
            # this?" needs its own arms rather than reusing the additive ones.
            for tag, m in (("random", random_mask(eligible, n_sel, rng)),
                           ("matched", matched_random_mask(layout, chosen,
                                                           eligible, rng)),
                           ("complement", complement_mask(delta, chosen, eligible))):
                arms.append(Arm(f"{tag}_scale@{lam:g}", "control", "neutral",
                                "ref_neutral",
                                Edit(op="scale", mask=torch.from_numpy(m), lam=lam,
                                     pos_mask=pos_mask),
                                lam, int(m.sum()),
                                note="same edge count, same multiplier"))
        # The pos instruction is the thing we are trying to imitate, so the
        # same amplification under it is the "does it saturate" reading.
        arms.append(Arm(f"circuit_scale_on_pos@{lam:g}", "amplify", "pos", "ref_pos",
                        Edit(op="scale", mask=t_chosen, lam=lam, pos_mask=pos_mask),
                        lam, n_sel))

    # -- per-stream decomposition ------------------------------------------
    if not args.skip_streams:
        lam = max(args.add_lams) if args.add_lams else 1.0
        for s in ("q", "k", "v", "r"):
            m = chosen & layout.mask(streams=[s])
            if not m.any():
                continue
            arms.append(Arm(f"circuit_add_{s}@{lam:g}", "stream", "neutral",
                            "ref_neutral", add_edit(m, lam, False), lam, int(m.sum()),
                            note="q/k are RMSNorm-ed after mixing; v/r are not"))

    # -- necessity ----------------------------------------------------------
    arms.append(Arm("zero_circuit_on_pos", "necessity", "pos", "ref_pos",
                    Edit(op="zero", mask=t_chosen, pos_mask=pos_mask), 0.0, n_sel,
                    expect=-1.0,
                    note="knock the circuit out while the pos instruction is given"))
    if not args.skip_controls:
        m = random_mask(eligible, n_sel, rng)
        arms.append(Arm("zero_random_on_pos", "control", "pos", "ref_pos",
                        Edit(op="zero", mask=torch.from_numpy(m), pos_mask=pos_mask),
                        0.0, int(m.sum()), expect=-1.0))

    # -- activation patching -------------------------------------------------
    if not args.skip_patch:
        if edit_channel != "pred":
            print("[verify] skipping patch arms: donor routing is only "
                  "recomputable on the 'pred' channel")
        else:
            pl = ps.prefix_len
            dv = {pol: min(args.donor_variant, ps.variants[pol] - 1)
                  for pol in ("pos", "neg")}
            pref = {pol: torch.tensor(ps.by(pol, dv[pol])[0].ids[:pl],
                                      dtype=torch.long)
                    for pol in ("pos", "neg")}
            for src, dst, base, exp in (("pos", "neg", "ref_neg", 1.0),
                                        ("neg", "pos", "ref_pos", -1.0)):
                donor = make_donor(runner, pref[src], pl)
                arms.append(Arm(f"patch_{src}_into_{dst}", "patch", dst, base,
                                Edit(op="patch", mask=t_chosen, target=donor,
                                     pos_mask=pos_mask),
                                0.0, n_sel, capability=False, expect=exp,
                                note=f"circuit coords copied from the {src} run"))
                if args.skip_controls:
                    continue
                m = random_mask(eligible, n_sel, rng)
                arms.append(Arm(f"patch_{src}_into_{dst}_random", "control", dst, base,
                                Edit(op="patch", mask=torch.from_numpy(m),
                                     target=donor, pos_mask=pos_mask),
                                0.0, int(m.sum()), capability=False, expect=exp))
                arms.append(Arm(f"patch_{src}_into_{dst}_all", "patch", dst, base,
                                Edit(op="patch", mask=t_elig, target=donor,
                                     pos_mask=pos_mask),
                                0.0, int(eligible.sum()), capability=False, expect=exp,
                                note="every eligible coordinate patched"))
    return arms


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def paired_shift(arm_scores: list, base_scores: list) -> tuple[float, Optional[float]]:
    """Mean and t statistic of the per-prompt shift against the baseline arm."""
    a, b = np.asarray(arm_scores, float), np.asarray(base_scores, float)
    if a.shape != b.shape or a.size < 2:
        # Different polarities have different paraphrase counts, so the
        # reference arms are only comparable unpaired.
        return float(a.mean() - b.mean()), None
    d = a - b
    sd = d.std(ddof=1)
    t = float(d.mean() / (sd / np.sqrt(d.size))) if sd > 0 else 0.0
    return float(d.mean()), t


def fmt_table(arms: list[Arm], headroom: float, nll_ref: Optional[float]) -> list[str]:
    head = ("| arm | group | edges | lam | score | shift vs base | t | "
            "% headroom | NLL |")
    rows = [head, "|" + "---|" * 9]
    for a in arms:
        r = a.result
        if not r:
            continue
        # Signed by intent: positive means the intervention moved the
        # behaviour the way it was supposed to, so knockouts read positive
        # when knocking the circuit out does suppress the behaviour.
        rec = (f"{100 * a.expect * r['shift'] / headroom:+.1f}%"
               if headroom and abs(headroom) > 1e-9 else "n/a")
        nll = (f"{r['nll']:.4f} ({r['nll'] - nll_ref:+.4f})"
               if r.get("nll") is not None and nll_ref is not None else "—")
        if r.get("damaged"):
            nll += " **damaged**"
        t_text = f"{r['t']:+.2f}" if r.get("t") is not None else "n/a"
        rows.append(f"| `{a.name}` | {a.group} | {a.n_edges} | {a.lam:g} | "
                    f"{r['score']:+.4f} ± {r['sem']:.4f} | {r['shift']:+.4f} | "
                    f"{t_text} | {rec} | {nll} |")
    return rows


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = {"auto": None, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    if args.dtype == "auto" and device.type == "cpu":
        dtype = torch.float32
    rng = np.random.default_rng(args.seed)

    in_dir = Path(args.in_dir)
    stem = f"circuit_{args.behavior}_{args.channel}"
    circuit = json.loads((in_dir / f"{stem}.json").read_text())
    stats = np.load(in_dir / f"{stem}_stats.npz")
    contrast = torch.load(in_dir / f"contrast_{args.behavior}.pt",
                          map_location="cpu", weights_only=False)
    print(f"[verify] circuit {in_dir / f'{stem}.json'} "
          f"({circuit['graph']['n_edges']} edges, rule {circuit['selection']})")

    chosen = stats["chosen"].astype(bool)
    eligible = stats["eligible"].astype(bool)
    delta = stats["delta_train" if args.delta == "train" else "delta_all"]
    if not chosen.any():
        raise SystemExit("the circuit is empty — nothing to verify. Loosen the "
                         "selection rule in discover_circuit.py (--select topk) "
                         "or accept the null result.")

    edit_channel = args.edit_channel or ("pred" if args.channel == "eff" else args.channel)
    if args.channel == "eff" and args.edit_channel is None:
        print("[verify] circuit was found on 'eff' (= pred + corr); intervening "
              "on 'pred', which shifts eff identically for additive arms")

    L, H = circuit["layout"]["L"], circuit["layout"]["H"]
    layout = RoutingLayout(L, H)
    cfg, fourway, predictor = load_models(args.config, args.ckpt, device,
                                          need_base=True, dtype=dtype)
    tokenizer = load_tokenizer(cfg, args.tokenizer if Path(args.tokenizer).is_dir() else None)
    runner = RoutingRunner(layout, predictor, fourway, device)
    if edit_channel == "corr" and not runner.has_corr:
        raise SystemExit("no correction MLPs in this checkpoint; "
                         "use --edit-channel pred")

    spec = spec_from_dict(contrast["behavior"])
    ps = build_prompt_set(spec, tokenizer, max_items=contrast["n_items"])
    assert list(ps.span) == list(contrast["span"]), \
        f"prompt span {ps.span} != recorded {contrast['span']} (tokenizer mismatch?)"
    lo, hi = ps.span

    keep = {"test": set(stats["test_items"].tolist()),
            "train": set(stats["train_items"].tolist()),
            "all": set(range(ps.n_items))}[args.items]
    prompts = {pol: [p for p in ps.by(pol) if p.item in keep]
               for pol in ("pos", "neg", "neutral")}
    print(f"[verify] scoring on the {args.items} half: {len(keep)} items, "
          f"{ {k: len(v) for k, v in prompts.items()} } prompts, "
          f"edit channel = {edit_channel}")

    # Edits run from the content span through the scored continuation; the
    # routing at the continuation positions is what produces those tokens.
    max_cont = max((len(tokenizer(c, add_special_tokens=False)["input_ids"])
                    for it in spec.items for c in it.candidates.values()), default=8)
    T_max = hi + max_cont + 8
    pos_mask = torch.zeros(T_max, dtype=torch.bool)
    pos_mask[(lo if args.edit_from == "content" else 0):] = True

    cross = None
    if args.cross_circuit:
        cp = Path(args.cross_circuit)
        cs = np.load(cp)
        cmask = cs["chosen"].astype(bool)
        if cmask.shape != chosen.shape:
            raise SystemExit(f"--cross-circuit has D={cmask.shape[0]}, "
                             f"this model has D={chosen.shape[0]}")
        cname = cp.name.replace("circuit_", "").replace("_stats.npz", "")
        cross = (cname, cmask, cs["delta_train"])
        overlap = int((cmask & chosen).sum())
        print(f"[verify] specificity arm from {cname}: {int(cmask.sum())} edges, "
              f"{overlap} shared with this circuit")

    # How much is there to enlarge?  The sequential path is bias-initialised
    # to 1.0 while hyperconnections start near zero, so a multiplicative arm
    # can only do something in proportion to the weights it scales.  Reporting
    # this up front stops a flat `circuit_scale` result from being read as
    # "amplifying the circuit does nothing" when the truth is "there was
    # nothing there to amplify" — and the |Delta|/|alpha| ratio says what
    # lambda the additive arms need to be a comparable perturbation.
    a0 = runner.alpha_pred(
        torch.tensor([p.ids for p in prompts["neutral"][:args.batch_size]],
                     dtype=torch.long))[:, lo:hi].mean((0, 1)).numpy()
    ac, ae = np.abs(a0[chosen]), np.abs(a0[eligible])
    alpha_stats = {
        "magnitude_reference_channel": "pred",
        "mean_abs_alpha_circuit": float(ac.mean()),
        "max_abs_alpha_circuit": float(ac.max()),
        "mean_abs_alpha_eligible": float(ae.mean()),
        "mean_abs_delta_circuit": float(np.abs(delta[chosen]).mean()),
        "delta_over_alpha_circuit":
            float(np.abs(delta[chosen]).mean() / max(ac.mean(), 1e-12)),
        # Over every eligible coordinate this is the size of the *whole*
        # instruction effect on the routing weights, i.e. how much the full
        # patch arm can possibly change. A value of a few percent says the
        # predictor reads the instruction without much changing what it emits.
        "mean_abs_delta_eligible": float(np.abs(delta[eligible]).mean()),
        "delta_over_alpha_eligible":
            float(np.abs(delta[eligible]).mean() / max(ae.mean(), 1e-12)),
        # lambda at which the additive arms perturb alpha by its own size.
        "add_lam_for_parity":
            float(ac.mean() / max(np.abs(delta[chosen]).mean(), 1e-12)),
        "n_circuit_hyper": int((chosen & layout.hyper_arr).sum()),
        "n_circuit_sequential": int((chosen & ~layout.hyper_arr).sum()),
    }
    print(f"[verify] on the circuit, mean|alpha|={ac.mean():.4g} "
          f"(all eligible {ae.mean():.4g}), mean|Delta|="
          f"{alpha_stats['mean_abs_delta_circuit']:.4g} -> "
          f"Delta/alpha = {alpha_stats['delta_over_alpha_circuit']:.3g} "
          f"(parity at lam={alpha_stats['add_lam_for_parity']:.3g}; "
          f"whole-channel Delta/alpha = "
          f"{alpha_stats['delta_over_alpha_eligible']:.3g})")
    if max(args.add_lams, default=0.0) < 0.5 * alpha_stats["add_lam_for_parity"]:
        print(f"[verify] NOTE: the largest additive lambda "
              f"({max(args.add_lams, default=0.0):g}) perturbs alpha by only "
              f"{100 * max(args.add_lams, default=0.0) * alpha_stats['delta_over_alpha_circuit']:.1f}"
              f"% of its own magnitude; a flat additive sweep here means the "
              f"perturbation was small, not that the circuit is inert")

    arms = build_arms(args, layout, chosen, eligible, delta, ps, runner, rng,
                      edit_channel, pos_mask, cross)

    windows = None
    if not args.skip_capability:
        windows = capability_windows(tokenizer, args.cap_seq_len, args.cap_windows,
                                     args.cap_cache)
        print(f"[verify] capability check on {tuple(windows.shape)} held-out tokens")

    # The capability check runs on plain text with no content span, so the
    # positional restriction is dropped there: this is the worst case, which
    # is the honest thing to report for a deployed steering vector.
    cap_mask = torch.ones(max(T_max, args.cap_seq_len + 1), dtype=torch.bool)

    results: dict[str, dict] = {}
    t0 = time.time()
    for arm in arms:
        runner.clear_edits()
        if arm.edit is not None:
            runner.set_edit(edit_channel, arm.edit)
        sc = score_behavior(runner, spec, prompts[arm.polarity], tokenizer,
                            ps.filler_id, args.batch_size)
        arm.result = {"score": sc["score"], "sem": sc["sem"],
                      "per_item": sc["per_item"], "polarity": arm.polarity,
                      "metric": sc["metric"]}
        if windows is not None and arm.capability:
            runner.clear_edits()
            if arm.edit is not None:
                runner.set_edit(edit_channel, replace(arm.edit, pos_mask=cap_mask))
            arm.result["nll"] = capability_nll(runner, windows,
                                               max(1, args.batch_size // 4))
        results[arm.name] = arm.result
        print(f"[verify] {arm.name:34s} n={arm.n_edges:5d} "
              f"score={sc['score']:+.4f} "
              f"nll={arm.result.get('nll', float('nan')):.4f} "
              f"({time.time() - t0:.0f}s)")
    runner.clear_edits()

    # -- shifts relative to each arm's own baseline -------------------------
    # Two passes: every shift is computed against the *unmodified* baseline
    # results before any per-item list is dropped, since results[name] and
    # arm.result are the same dict and popping in one loop would blank out the
    # reference arms before the arms that point at them are compared.
    for arm in arms:
        base = results.get(arm.baseline, {})
        arm.result["shift"], arm.result["t"] = paired_shift(
            arm.result["per_item"], base.get("per_item", arm.result["per_item"]))
    for arm in arms:
        arm.result.pop("per_item", None)

    ref_pos = results["ref_pos"]["score"]
    ref_neg = results["ref_neg"]["score"]
    ref_neu = results["ref_neutral"]["score"]
    headroom = ref_pos - ref_neg
    nll_ref = results["ref_neutral"].get("nll")
    print(f"[verify] headroom pos-neg = {headroom:+.4f} "
          f"(neutral sits at {ref_neu:+.4f})")

    # Damage gate: a behaviour shift bought by wrecking the language model is
    # not steering, so such an arm can neither win nor disqualify. Without this
    # the loudest control is usually `dense_scale` at the top lambda, which
    # moves the metric a long way in whatever direction breaking the model
    # happens to push it.
    for a in arms:
        rise = (a.result["nll"] - nll_ref
                if a.result.get("nll") is not None and nll_ref is not None
                else 0.0)
        a.result["nll_rise"] = rise
        a.result["damaged"] = bool(rise > args.max_nll_rise)

    # Toward-target shift: `expect` says which way a working intervention
    # should move the metric, `sign(headroom)` says which way "toward pos" is
    # for this behaviour. Ranking on this rather than |shift| stops an arm that
    # moves hugely the *wrong* way from being read as the strongest effect.
    hsign = np.sign(headroom) if abs(headroom) > 1e-9 else 1.0

    def toward(a: Arm) -> float:
        return float(a.expect * hsign * a.result["shift"])

    # The headline. Arms are only comparable when they share a baseline, so
    # each family is judged against controls measured on the same prompts.
    def pool(groups: set, baseline: str) -> list[Arm]:
        return [a for a in arms
                if a.group in groups and a.baseline == baseline and a.result]

    def best(groups: set, baseline: str) -> Optional[Arm]:
        cand = [a for a in pool(groups, baseline) if not a.result["damaged"]]
        return max(cand, key=toward) if cand else None

    def judge(circuit_arm: Optional[Arm], control_arm: Optional[Arm],
              label: str, ran: Sequence[Arm] = ()) -> str:
        if circuit_arm is None:
            if ran:
                # Not "nothing happened" — everything happened, and all of it
                # was damage. Worth saying out loud: a channel can have plenty
                # of causal leverage and still offer no usable steering.
                worst = max(ran, key=lambda a: abs(a.result["shift"]))
                return (f"{label}: ALL ARMS DAMAGED — every one of the "
                        f"{len(ran)} arms raised held-out NLL by more than "
                        f"{args.max_nll_rise:g} nats (largest shift "
                        f"{worst.result['shift']:+.4f} by `{worst.name}` at "
                        f"{worst.result['nll_rise']:+.2f} nats); none of these "
                        f"tested arms meets the NLL criterion")
            return f"{label}: not run"
        s, tw = circuit_arm.result["shift"], toward(circuit_arm)
        if circuit_arm.result.get("damaged"):
            return (f"{label}: DAMAGED — `{circuit_arm.name}` shifts {s:+.4f} "
                    f"but raises held-out NLL by "
                    f"{circuit_arm.result['nll_rise']:+.3f} nats; that is model "
                    f"damage, not steering")
        frac = f"{100 * abs(s) / abs(headroom):.0f}% of headroom" \
            if abs(headroom) > 1e-9 else "headroom is ~0"
        if tw <= 0:
            return (f"{label}: NO EFFECT IN THE RIGHT DIRECTION — the best "
                    f"undamaged arm `{circuit_arm.name}` shifts {s:+.4f}, "
                    f"which is the opposite way from what the intervention "
                    f"should do")
        if tw < 2 * abs(circuit_arm.result["sem"]):
            return (f"{label}: SMALL RELATIVE TO RAW SCORE SPREAD — the best "
                    f"arm `{circuit_arm.name}` shifts {s:+.4f}, below twice "
                    "the arm's cross-prompt SEM; this is not a paired null test")
        if control_arm is None:
            return (f"{label}: MOVED but UNCONTROLLED — `{circuit_arm.name}` "
                    f"shifts {s:+.4f} ({frac}); no control arm was run")
        k = control_arm.result["shift"]
        if tw <= toward(control_arm):
            return (f"{label}: MATCHED BY A TESTED CONTROL (descriptive) — `{control_arm.name}` "
                    f"matches the circuit ({k:+.4f} vs {s:+.4f})")
        return (f"{label}: LARGER THAN TESTED CONTROL (descriptive) — `{circuit_arm.name}` shifts {s:+.4f} "
                f"({frac}) vs {k:+.4f} for the best control "
                f"`{control_arm.name}`")

    parts = [judge(best({"sufficiency", "amplify"}, "ref_neutral"),
                   best({"control"}, "ref_neutral"), "steering",
                   pool({"sufficiency", "amplify"}, "ref_neutral"))]
    if any(a.group == "patch" for a in arms):
        for base, lab in (("ref_neg", "patch pos->neg"), ("ref_pos", "patch neg->pos")):
            pa = [a for a in arms if a.group == "patch" and a.baseline == base
                  and not a.name.endswith("_all") and a.result]
            ca = [a for a in arms if a.group == "control" and a.baseline == base
                  and a.name.startswith("patch") and a.result]
            if pa:
                parts.append(judge(max(pa, key=toward),
                                   max(ca, key=toward) if ca else None, lab))
        # Full and subset interventions need not be monotone: coordinate
        # effects can cancel. Report the measured full patch without treating
        # it as a mathematical bound on other masks or editing directions.
        ceil = [a for a in arms if a.name.endswith("_all") and a.result]
        if ceil:
            c = max(ceil, key=toward)
            pct = 100 * abs(c.result["shift"]) / abs(headroom) \
                if abs(headroom) > 1e-9 else float("nan")
            parts.append(
                f"full patch: `{c.name}` patches every eligible coordinate "
                f"and moves {c.result['shift']:+.4f} = {pct:.1f}% of headroom; "
                "subset interventions can differ because effects may cancel")
    nec = next((a for a in arms if a.name == "zero_circuit_on_pos" and a.result), None)
    ncl = next((a for a in arms if a.name == "zero_random_on_pos" and a.result), None)
    if nec is not None:
        parts.append(judge(nec, ncl, "necessity (knockout)"))
    verdict = "; ".join(parts)
    for p in parts:
        print(f"[verify] VERDICT: {p}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "behavior": args.behavior, "channel": args.channel,
        "edit_channel": edit_channel, "items": args.items,
        "delta_source": args.delta, "match_norm": not args.no_match_norm,
        "edit_from": args.edit_from,
        "n_scored_items": len(keep),
        "circuit": {"n_edges": int(chosen.sum()),
                    "selection": circuit["selection"],
                    "hyper_only": circuit.get("hyper_only"),
                    "replication": circuit.get("replication")},
        "alpha_stats": alpha_stats,
        "references": {"pos": ref_pos, "neg": ref_neg, "neutral": ref_neu,
                       "headroom": headroom,
                       "instruction_gap_from_step1":
                           contrast.get("behavior_scores", {}).get("gap", {})},
        "capability_nll_reference": nll_ref,
        "verdict": verdict,
        "verdict_parts": parts,
        "arms": [{"name": a.name, "group": a.group, "polarity": a.polarity,
                  "baseline": a.baseline, "lam": a.lam, "n_edges": a.n_edges,
                  "expect": a.expect, "note": a.note, **a.result} for a in arms],
        "config": args.config, "ckpt": args.ckpt,
        "source_circuit": str(in_dir / f"{stem}.json"),
    }
    save_json(payload, out_dir / f"verify_{args.behavior}_{args.channel}.json")

    gap = contrast.get("behavior_scores", {}).get("gap", {})
    md = [
        f"# Causal verification — `{args.behavior}` / channel `{args.channel}`",
        "",
        f"Circuit: {int(chosen.sum())} edges, rule `{json.dumps(circuit['selection'])}`  ",
        f"Intervening on the `{edit_channel}` channel, scored on the "
        f"**{args.items}** item half ({len(keep)} items)  ",
        f"Metric: {results['ref_neutral']['metric']}  ",
        f"Controls {'norm-matched' if not args.no_match_norm else 'raw'}; "
        f"edits applied from {'the content span' if args.edit_from == 'content' else 'position 0'}",
        "",
        "## Verdict",
        "",
    ] + [f"- **{p}**" for p in parts] + [
        "",
        f"- reference: pos `{ref_pos:+.4f}`, neutral `{ref_neu:+.4f}`, "
        f"neg `{ref_neg:+.4f}` → headroom `{headroom:+.4f}`",
        f"- step-1 instruction sensitivity: `{gap.get('verdict', 'not measured')}` "
        f"(paired t = {gap.get('paired_t', float('nan')):.2f})",
        f"- circuit composition: {alpha_stats['n_circuit_hyper']} hyperconnections, "
        f"{alpha_stats['n_circuit_sequential']} sequential",
        f"- on the circuit: mean `|alpha|` = "
        f"`{alpha_stats['mean_abs_alpha_circuit']:.4g}`, mean `|Delta|` = "
        f"`{alpha_stats['mean_abs_delta_circuit']:.4g}` → `Delta/alpha` = "
        f"`{alpha_stats['delta_over_alpha_circuit']:.3g}`. If `|alpha|` is tiny "
        f"the multiplicative arms have almost nothing to enlarge, and a flat "
        f"`circuit_scale` row says that rather than saying the circuit is inert.",
        f"- the additive arms reach parity with `|alpha|` at "
        f"`lam = {alpha_stats['add_lam_for_parity']:.3g}`; this sweep went to "
        f"`{max(args.add_lams, default=0.0):g}`",
        f"- over the whole channel `Delta/alpha` = "
        f"`{alpha_stats['delta_over_alpha_eligible']:.3g}` — that is the entire "
        f"effect the instruction has on the routing weights, and so the size of "
        f"the perturbation the full patch arm applies",
        f"- arms are marked **damaged** when held-out NLL rises more than "
        f"`{args.max_nll_rise:g}` nats; they are excluded from the verdict on "
        f"both sides, since an arm that breaks the model neither steers it nor "
        f"proves a circuit unspecific",
        "",
        "`% headroom` is the shift as a fraction of the pos−neg gap, signed by "
        "intent: 100% means the edit reproduced the whole effect of changing "
        "the instruction, and a knockout reads positive when it does suppress "
        "the behaviour. `t` is a paired test over prompts against the arm's "
        "own baseline.",
        "The automatic size label compares the shift with the arm's raw-score "
        "SEM, not the SEM of paired differences. It is descriptive rather than "
        "a significance test. Prompt paraphrases also share content items; "
        "item-clustered uncertainty is needed for inferential claims.",
        "",
        "## Arms",
        "",
    ]
    md += fmt_table(arms, headroom, nll_ref)
    md += [
        "",
        "## How to read this",
        "",
        "- If `dense_add` moves the behaviour as much as `circuit_add`, the "
        "circuit is not localised — the effect is whatever a bulk shift of the "
        "routing weights does.",
        "- If `dense_add` and every other tested arm are flat at large λ, these "
        "directions do not establish useful steering for this behavior. Other "
        "directions, masks and tasks remain untested.",
        "- `patch_pos_into_neg` reaching `ref_pos` while "
        "`patch_pos_into_neg_random` does not is the cleanest possible positive "
        "result using an observed donor magnitude. The full patch is a "
        "comparison intervention, not a bound on subset effects.",
        "- q/k stream arms being flat while v/r move is expected, not a bug: "
        "q_norm/k_norm renormalise after mixing, so magnitude changes on those "
        "streams are partly undone.",
        "- Any arm whose NLL rises sharply bought its behaviour shift by "
        "damaging the model; it is not steering.",
    ]
    (out_dir / f"verify_{args.behavior}_{args.channel}.md").write_text("\n".join(md) + "\n")
    print(f"[saved] {out_dir / f'verify_{args.behavior}_{args.channel}.md'}")
    runner.close()
    print("[verify] DONE")


if __name__ == "__main__":
    main()
