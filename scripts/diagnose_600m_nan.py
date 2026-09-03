"""Diagnose the 600M FourWay NaN divergence (loss 11.81 @ step 0 -> nan @ step 10).

WHAT THE LOGS ALREADY TELL US
-----------------------------
metrics.csv of the failing run
(/work/hdd/bfqt/dagformer_checkpoints/fourway_600m_chinchilla_mmap/metrics.csv):

    step,train/nll,...,grad/base_norm,grad/predictor_norm,...
    0,11.8125,...,nan,nan,...          <- LOSS IS FINITE, GRADS ARE ALREADY NaN
    10,nan,...,nan,nan,...

At step 0 `get_lr(0) == 0.0` (warmup), so *no weight has been updated yet*.
A finite loss with a NaN gradient on the very first backward pass rules out
every "learning rate too high / warmup too short / slow blowup" explanation.
The failure is a single non-finite value produced inside one forward/backward,
which is then broadcast to *every* parameter by two amplifiers:

  1. `torch.nn.utils.clip_grad_norm_`: if ANY grad element is +/-inf the total
     norm is inf, clip_coef = max_norm / (inf + 1e-6) = 0.0, and every grad is
     multiplied by 0 -> inf * 0 = NaN. One bad element poisons all params.
     (scripts/pretrain_dagformer.py:1576)
  2. AdamW: `exp_avg`/`exp_avg_sq` absorb the NaN even though lr == 0, and
     `addcdiv_(..., value=-0.0)` still yields 0 * NaN = NaN in the params.
     From step 0 on, the run is permanently NaN. Nothing recovers it.

Run `--proof-clip` to see amplifier (1) demonstrated in two lines of torch.

WHAT THIS SCRIPT DOES
---------------------
Rebuilds the exact training-step-0 computation (same model construction, same
bf16 cast, same FourWay predictor, same mmap batches in the same order) and
instruments it so the FIRST non-finite tensor is identified:

  * per-module forward output absmax + non-finite count
  * per-module backward grad_output absmax + non-finite count
  * per-parameter grad inspection BEFORE clipping (inf vs nan vs finite)
  * post-clip check, proving the inf -> NaN amplification

A/B knobs to isolate the two variables that actually changed between the
HEALTHY June 600M run and the two NaN runs:

  --vnorm on|off   `use_v_norm` is true only in the 600M/1B configs. Every
                   healthy mmap FourWay run (75M/150M/300M) has it false.
  --data mmap|random   the June run streamed HF Dolma; the NaN runs read the
                   pretokenized mmap shards. Same config otherwise.

EXAMPLES
--------
  # plumbing smoke test on CPU (small model, seconds)
  python3 scripts/diagnose_600m_nan.py --scale 8 --seq-len 128 --micro-batch 2 \
      --micro-steps 2 --device cpu --data random

  # the real test, 1 GPU, real 600M dims, real first mmap batches
  python3 scripts/diagnose_600m_nan.py --config configs/fourway_600m_chinchilla.yaml \
      --device cuda --data mmap --micro-steps 16

  # A/B: is v_norm required to trigger it?
  python3 scripts/diagnose_600m_nan.py --device cuda --data mmap --vnorm off

  # clip_grad_norm_ inf->NaN amplification proof (no model, instant, CPU)
  python3 scripts/diagnose_600m_nan.py --proof-clip
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.pretrain_dagformer import (  # noqa: E402
    DAGFormerPretrainConfig,
    create_model,
    replace_olmo_rmsnorm,
)
from src.model.olmo_graph import FourWayDAGFormer  # noqa: E402
from src.model.predictor import FourWayPredictor  # noqa: E402

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp16": torch.float16}


# ─── clip_grad_norm_ amplification proof ─────────────────────────────────────

def proof_clip() -> None:
    """Show that ONE inf gradient turns EVERY parameter's grad into NaN."""
    print("=" * 72)
    print("PROOF: clip_grad_norm_ turns a single inf grad into NaN everywhere")
    print("=" * 72)
    good = nn.Parameter(torch.ones(4))
    bad = nn.Parameter(torch.ones(4))
    good.grad = torch.full((4,), 0.5)
    bad.grad = torch.full((4,), float("inf"))
    total = torch.nn.utils.clip_grad_norm_([good, bad], max_norm=1.0)
    print(f"  total_norm returned      : {total}")
    print(f"  'good' grad after clip   : {good.grad}   <- was 0.5, finite")
    print(f"  'bad'  grad after clip   : {bad.grad}")
    print()
    print("  Then AdamW, even at lr=0.0:")
    p = nn.Parameter(torch.ones(4))
    p.grad = torch.full((4,), float("nan"))
    opt = torch.optim.AdamW([p], lr=0.0, weight_decay=0.1)
    opt.step()
    print(f"  param after AdamW(lr=0)  : {p.data}   <- 0 * NaN = NaN, unrecoverable")
    print()
    print("  => any transient inf anywhere in the backward permanently NaNs the run,")
    print("     which is exactly the observed step-0 signature.")


# ─── data ────────────────────────────────────────────────────────────────────

def mmap_batches(
    index_path: str,
    seq_len: int,
    batch_size: int,
    n_batches: int,
    rank: int,
    world_size: int,
    num_workers: int,
    worker_id: int,
    seed: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Reproduce the exact windows one (rank, worker) consumes at step 0.

    Mirrors MmapPackedDataset._worker_stride()/__iter__ without needing a
    DataLoader, so a single process can replay any rank's stream.
    """
    from src.data.mmap_dataset import MmapPackedDataset, _epoch_positions

    ds = MmapPackedDataset(
        index_path=index_path, seq_len=seq_len, rank=rank,
        world_size=world_size, seed=seed, block_size=1024,
    )
    total_workers = world_size * max(num_workers, 1)
    global_worker_id = rank * max(num_workers, 1) + worker_id
    positions = _epoch_positions(
        n_samples=ds.n_samples, seed=seed, block_size=ds.block_size,
        global_worker_id=global_worker_id, total_workers=total_workers,
        skip_positions=0,
    )
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    ids: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for sample_idx in positions:
        w = ds._read_window(sample_idx)
        ids.append(torch.from_numpy(w[:seq_len].astype("int64")))
        labels.append(torch.from_numpy(w[1:seq_len + 1].astype("int64")))
        if len(ids) == batch_size:
            out.append((torch.stack(ids), torch.stack(labels)))
            ids, labels = [], []
            if len(out) >= n_batches:
                break
    return out


def random_batches(
    vocab_size: int, seq_len: int, batch_size: int, n_batches: int, seed: int
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(n_batches):
        w = torch.randint(0, vocab_size, (batch_size, seq_len + 1), generator=g)
        out.append((w[:, :seq_len].contiguous(), w[:, 1:].contiguous()))
    return out


# ─── instrumentation ─────────────────────────────────────────────────────────

class NonFiniteTracer:
    """Records absmax / non-finite counts for every module's fwd out and bwd grad."""

    def __init__(self) -> None:
        self.fwd: List[Tuple[str, float, int]] = []
        self.bwd: List[Tuple[str, float, int]] = []
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    @staticmethod
    def _stat(t) -> Optional[Tuple[float, int]]:
        if not torch.is_tensor(t) or not t.is_floating_point() or t.numel() == 0:
            return None
        f = t.detach().float()
        finite = torch.isfinite(f)
        n_bad = int((~finite).sum().item())
        amax = float(f[finite].abs().max().item()) if bool(finite.any()) else float("nan")
        return amax, n_bad

    def attach(self, root: nn.Module, prefix: str) -> None:
        for name, mod in root.named_modules():
            if len(list(mod.children())) > 0:  # leaves only, keeps output readable
                continue
            tag = f"{prefix}.{name}" if name else prefix

            def fwd_hook(m, inp, out, tag=tag):
                outs = out if isinstance(out, (tuple, list)) else (out,)
                for o in outs:
                    s = self._stat(o)
                    if s is not None:
                        self.fwd.append((tag, s[0], s[1]))

            def bwd_hook(m, gin, gout, tag=tag):
                for g in gout:
                    s = self._stat(g)
                    if s is not None:
                        self.bwd.append((tag, s[0], s[1]))

            self.handles.append(mod.register_forward_hook(fwd_hook))
            self.handles.append(mod.register_full_backward_hook(bwd_hook))

    def clear(self) -> None:
        self.fwd.clear()
        self.bwd.clear()

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def report(self, top: int = 8) -> None:
        bad_f = [r for r in self.fwd if r[2] > 0]
        bad_b = [r for r in self.bwd if r[2] > 0]
        if bad_f:
            print(f"    FORWARD non-finite in {len(bad_f)} module output(s); first:")
            for tag, amax, n in bad_f[:top]:
                print(f"      {tag:<58s} n_bad={n}")
        if bad_b:
            # hooks fire in backward order, so bad_b[-1] is the earliest-in-network
            print(f"    BACKWARD non-finite in {len(bad_b)} grad_output(s).")
            print("      first to fire (closest to loss):")
            for tag, amax, n in bad_b[:top]:
                print(f"        {tag:<56s} n_bad={n}")
            print("      last to fire (deepest into the net):")
            for tag, amax, n in bad_b[-top:]:
                print(f"        {tag:<56s} n_bad={n}")
        if not bad_f and not bad_b:
            fmax = max((r[1] for r in self.fwd), default=0.0)
            bmax = max((r[1] for r in self.bwd), default=0.0)
            print(f"    all finite. max |fwd out|={fmax:.4g}  max |bwd grad|={bmax:.4g}")
        # magnitude ranking is useful even when nothing overflowed yet
        hot_f = sorted(self.fwd, key=lambda r: -r[1])[:top]
        hot_b = sorted(self.bwd, key=lambda r: -r[1])[:top]
        print("    largest forward activations:")
        for tag, amax, _ in hot_f:
            print(f"      {tag:<58s} {amax:.4g}")
        print("    largest backward grads:")
        for tag, amax, _ in hot_b:
            print(f"      {tag:<58s} {amax:.4g}")


def grad_report(named_params, label: str) -> Tuple[int, int, float]:
    """Per-parameter grad audit. Returns (n_nan_tensors, n_inf_tensors, max_abs)."""
    n_nan = n_inf = 0
    amax = 0.0
    offenders: List[str] = []
    for name, p in named_params:
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        has_nan = bool(torch.isnan(g).any().item())
        has_inf = bool(torch.isinf(g).any().item())
        if has_nan:
            n_nan += 1
        if has_inf:
            n_inf += 1
        if has_nan or has_inf:
            if len(offenders) < 10:
                offenders.append(f"{name} (nan={has_nan}, inf={has_inf})")
        else:
            amax = max(amax, float(g.abs().max().item()))
    print(f"  [{label}] tensors with NaN={n_nan}  with inf={n_inf}  "
          f"max|grad| over clean tensors={amax:.4g}")
    for o in offenders:
        print(f"      offender: {o}")
    return n_nan, n_inf, amax


# ─── main ────────────────────────────────────────────────────────────────────

def build(args, cfg: DAGFormerPretrainConfig, device, dtype):
    """Construct base model + FourWay wrapper + predictor exactly as the trainer does."""
    base = create_model(cfg)
    if cfg.replace_rmsnorm:
        n = replace_olmo_rmsnorm(base)
        print(f"Replaced {n} Olmo2RMSNorm -> torch.nn.RMSNorm")
    base = base.to(device, dtype=dtype)

    use_v_norm = cfg.use_v_norm
    if args.vnorm == "on":
        use_v_norm = True
    elif args.vnorm == "off":
        use_v_norm = False

    fourway = FourWayDAGFormer(
        model=base,
        num_layers=cfg.num_hidden_layers,
        num_heads=cfg.num_attention_heads,
        use_local_correction=(cfg.routing_mode == "fourway_corrected"),
        correction_hidden=cfg.correction_hidden,
        use_triton_kernel=False,
        use_v_norm=use_v_norm,
        correction_pool=cfg.correction_pool,
    ).to(device)
    predictor = FourWayPredictor(
        vocab_size=cfg.vocab_size,
        encoder_dim=cfg.predictor_encoder_dim,
        encoder_layers=cfg.predictor_encoder_layers,
        encoder_heads=cfg.predictor_encoder_heads,
        max_seq_len=cfg.predictor_max_seq_len,
        num_layers=cfg.num_hidden_layers,
        num_heads=cfg.num_attention_heads,
        hidden_dim=cfg.fourway_hidden,
        causal=cfg.predictor_causal,
        dropout=cfg.predictor_dropout,
    ).to(device)
    print(f"use_v_norm={use_v_norm}  (v_norms dtype="
          f"{next(fourway.v_norms.parameters()).dtype if use_v_norm else 'n/a'}, "
          f"base dtype={dtype})")
    return base, fourway, predictor


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/fourway_600m_chinchilla.yaml")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--data", default="mmap", choices=["mmap", "random"])
    ap.add_argument("--micro-steps", type=int, default=16,
                    help="micro-batches to run (16 = one full step-0 accumulation)")
    ap.add_argument("--micro-batch", type=int, default=0, help="0 = from config")
    ap.add_argument("--seq-len", type=int, default=0, help="0 = from config")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--vnorm", default="config", choices=["config", "on", "off"])
    ap.add_argument("--scale", type=int, default=1,
                    help="divide hidden/intermediate/layers/vocab by this for a CPU smoke test")
    ap.add_argument("--no-hooks", action="store_true",
                    help="skip per-module tracing (much faster, param grads still audited)")
    ap.add_argument("--proof-clip", action="store_true",
                    help="only run the clip_grad_norm_ inf->NaN demonstration and exit")
    args = ap.parse_args()

    if args.proof_clip:
        proof_clip()
        return

    cfg = DAGFormerPretrainConfig.from_yaml(args.config)
    if args.scale > 1:
        s = args.scale
        cfg.hidden_size = max(cfg.num_attention_heads * 8, cfg.hidden_size // s)
        cfg.intermediate_size = max(64, cfg.intermediate_size // s)
        cfg.num_hidden_layers = max(3, cfg.num_hidden_layers // s)
        cfg.vocab_size = max(4096, cfg.vocab_size // s)
    micro_batch = args.micro_batch or cfg.micro_batch_size
    seq_len = args.seq_len or cfg.seq_len

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cuda" if args.device == "cuda" else "cpu")
    )
    dtype = DTYPES[args.dtype]

    torch.manual_seed(cfg.seed + args.rank)

    print("=" * 72)
    print("600M FourWay NaN diagnostic")
    print("=" * 72)
    print(f"config      : {args.config}")
    print(f"dims        : hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
          f"heads={cfg.num_attention_heads} head_dim={cfg.hidden_size // cfg.num_attention_heads}")
    print(f"device/dtype: {device} / {dtype}")
    print(f"batch       : micro_batch={micro_batch} seq_len={seq_len} "
          f"micro_steps={args.micro_steps}")
    print(f"data        : {args.data}")
    print()

    base, fourway, predictor = build(args, cfg, device, dtype)

    if args.data == "mmap":
        assert cfg.mmap_index_path, "config has no mmap_index_path"
        batches = mmap_batches(
            cfg.mmap_index_path, seq_len, micro_batch, args.micro_steps,
            args.rank, args.world_size, args.num_workers, args.worker_id, cfg.seed,
        )
        print(f"loaded {len(batches)} real mmap micro-batches "
              f"(rank={args.rank}, worker={args.worker_id})")
    else:
        batches = random_batches(cfg.vocab_size, seq_len, micro_batch,
                                 args.micro_steps, cfg.seed)
        print(f"generated {len(batches)} random micro-batches")

    tracer = NonFiniteTracer()
    if not args.no_hooks:
        tracer.attach(fourway, "fourway")
        tracer.attach(predictor, "predictor")

    all_params = [p for p in base.parameters() if p.requires_grad]
    all_params += list(predictor.parameters())
    all_params += list(fourway.get_routing_parameters())
    named = ([(f"base.{n}", p) for n, p in base.named_parameters()]
             + [(f"pred.{n}", p) for n, p in predictor.named_parameters()]
             + [(f"routing.{n}", p) for n, p in fourway.named_parameters()
                if not n.startswith("olmo.")])

    fourway.train()
    predictor.train()
    for p in all_params:
        p.grad = None

    accum = len(batches)
    first_bad = None
    for i, (ids, labels) in enumerate(batches):
        ids = ids.to(device)
        labels = labels.to(device)
        tracer.clear()

        rw = predictor(ids)
        logits = fourway(ids, rw)
        nll = F.cross_entropy(
            logits.contiguous().view(-1, cfg.vocab_size),
            labels.contiguous().view(-1),
        )
        (nll / accum).backward()

        loss_ok = bool(torch.isfinite(nll).item())
        print(f"\n[micro {i:02d}] nll={nll.item():.6f} finite={loss_ok}")
        if not args.no_hooks:
            tracer.report()
        n_nan, n_inf, amax = grad_report(named, f"grads after micro {i:02d} (PRE-CLIP)")
        if first_bad is None and (n_nan or n_inf or not loss_ok):
            first_bad = i
            print(f"  >>> FIRST non-finite appeared at micro-batch {i} <<<")
            if args.data == "mmap":
                print(f"      token id range in this batch: "
                      f"[{int(ids.min())}, {int(ids.max())}]")

    print("\n" + "=" * 72)
    print("ACCUMULATED-GRADIENT AUDIT (this is what clip_grad_norm_ sees)")
    print("=" * 72)
    n_nan, n_inf, amax = grad_report(named, "pre-clip")
    total_norm = torch.nn.utils.clip_grad_norm_(all_params, cfg.max_grad_norm)
    print(f"  clip_grad_norm_ returned total_norm = {total_norm}")
    post_nan, post_inf, post_amax = grad_report(named, "post-clip")
    if (n_inf or n_nan) and post_nan > n_nan:
        print("  >>> CONFIRMED: clipping propagated the non-finite value to "
              f"{post_nan} parameter tensors (was {n_nan}). "
              "This is the mechanism that NaNs the whole model in one step.")
    print()
    if first_bad is None:
        print("RESULT: no non-finite value reproduced with these settings.")
        print("        Re-run with --vnorm on / --data mmap / a different --worker-id")
        print("        or --rank to sweep the other ranks' step-0 streams.")
    else:
        print(f"RESULT: reproduced. First non-finite at micro-batch {first_bad}.")
        print("        Re-run with --vnorm off to test whether the V post-mix")
        print("        RMSNorm is required to trigger it (it is enabled ONLY in the")
        print("        600M/1B configs; every healthy mmap FourWay run has it off).")


if __name__ == "__main__":
    main()
