"""Compare frozen dense/DAG backbone attention-head supports on copying.

Binary masks act on attention-head outputs before o_proj. All MLPs and routing
networks remain present. Counts are NOT sizes of complete causal circuits.
Only mask scores are optimized, separately at each exact cardinality.
"""
from __future__ import annotations

import argparse
import json
import time
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from interp_common import load_elh, load_eval_ids, synthetic_induction_ids


def scored_positions(length, count, natural=False):
    start, end = (0, length - 1) if natural else (length // 2 - 1, length - 2)
    return torch.linspace(start, end, count).long().unique()


def stats(values):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260930)
    boot = a[rng.integers(len(a), size=(2000, len(a)))].mean(1)
    return {"mean": float(a.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist()}


def row_metrics(logits, teacher, targets):
    lp = logits.float().log_softmax(-1)
    prob = teacher.exp()
    return {
        "teacher_kl": (prob * (teacher - lp)).sum(-1).mean(-1).detach().cpu().tolist(),
        "target_nll": (-lp.gather(-1, targets[..., None]).squeeze(-1).mean(-1)).detach().cpu().tolist(),
        "accuracy": (lp.argmax(-1) == targets).float().mean(-1).detach().cpu().tolist(),
        "teacher_agreement": (lp.argmax(-1) == teacher.argmax(-1)).float().mean(-1).detach().cpu().tolist(),
        "teacher_entropy": (-(prob * teacher).sum(-1).mean(-1)).detach().cpu().tolist(),
    }


class Runner:
    def __init__(self, kind, model_dir):
        self.kind, self.device = kind, torch.device("cuda")
        elh = load_elh()
        self.cfg = elh.load_config(str(model_dir / "config.yaml"))
        self.predictor = None
        if kind == "dagformer":
            self.model, self.predictor = elh.load_fourway(str(model_dir / "checkpoint.pt"), self.cfg, self.device)
            self.model.use_triton_kernel = False
            self.base = self.model.olmo
        else:
            self.model = elh.load_dense(str(model_dir / "checkpoint.pt"), self.cfg, self.device)
            self.base = self.model
        self.L, self.H = self.cfg["num_hidden_layers"], self.cfg["num_attention_heads"]
        self.mask, self.positions, self.fixed_routes = None, None, False
        self.route_table = None
        self.handles = []
        for layer, block in enumerate(self.base.model.layers):
            def head_hook(module, inputs, layer=layer):
                x = inputs[0]
                if self.mask is None:
                    return None
                y = x.reshape(*x.shape[:-1], self.H, -1)
                y = y * self.mask[layer].to(y.dtype).view(1, 1, self.H, 1)
                return (y.reshape_as(x), *inputs[1:])
            self.handles.append(block.self_attn.o_proj.register_forward_pre_hook(head_hook))

        def output_hook(module, inputs):
            if self.positions is None:
                return None
            return (inputs[0][:, self.positions], *inputs[1:])
        self.handles.append(self.base.lm_head.register_forward_pre_hook(output_hook))

    def forward(self, rows, positions, mask):
        rows = rows.to(self.device)
        self.mask, self.positions = mask, positions.to(self.device)
        if self.predictor is None:
            return self.model(input_ids=rows, use_cache=False).logits
        if self.fixed_routes:
            routing = {s: [t[:, :rows.shape[1]].expand(rows.shape[0], -1, *t.shape[2:])
                           for t in ts] for s, ts in self.route_table.items()}
        else:
            with torch.no_grad():
                routing = self.predictor(rows)
        return self.model(rows, routing)

    @torch.no_grad()
    def calibrate_routes(self, rows, batch_size):
        totals = None
        for start in range(0, len(rows), batch_size):
            routing = self.predictor(rows[start:start + batch_size].to(self.device))
            if totals is None:
                totals = {s: [t.sum(0, keepdim=True) for t in ts] for s, ts in routing.items()}
            else:
                for s, ts in routing.items():
                    for dest, t in zip(totals[s], ts):
                        dest.add_(t.sum(0, keepdim=True))
        self.route_table = {s: [t / len(rows) for t in ts] for s, ts in totals.items()}

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def cache_teacher(runner, dataset, batch_size):
    rows, positions = dataset["ids"], dataset["positions"]
    cache = []
    for start in range(0, len(rows), batch_size):
        logits = runner.forward(rows[start:start + batch_size], positions, None)
        cache.append(logits.float().log_softmax(-1).cpu())
    dataset["teacher"] = torch.cat(cache)


@torch.no_grad()
def evaluate(runner, dataset, mask, batch_size):
    values = {}
    rows, positions = dataset["ids"], dataset["positions"]
    for start in range(0, len(rows), batch_size):
        logits = runner.forward(rows[start:start + batch_size], positions, mask)
        teacher = dataset["teacher"][start:start + batch_size].to(runner.device)
        target = dataset["targets"][start:start + batch_size].to(runner.device)
        for name, vals in row_metrics(logits, teacher, target).items():
            values.setdefault(name, []).extend(vals)
    return {"values": values, "summary": {k: stats(v) for k, v in values.items()}}


def hard_mask(scores, k):
    hard = torch.zeros_like(scores).flatten()
    hard[scores.detach().flatten().topk(k).indices] = 1
    return hard.reshape_as(scores)


def optimize(runner, train, valid, k, args):
    mask_seed = args.seed if args.mask_seed is None else args.mask_seed
    gen = torch.Generator().manual_seed(mask_seed + k)
    scores = torch.nn.Parameter((.01 * torch.randn(runner.L, runner.H, generator=gen)).to(runner.device))
    opt = torch.optim.Adam([scores], lr=args.lr)
    history, best = [], None
    order = torch.randperm(len(train["ids"]), generator=gen)
    cursor = 0
    for step in range(args.steps + 1):
        if step % args.validate_every == 0 or step == args.steps:
            mask = hard_mask(scores, k)
            measured = evaluate(runner, valid, mask, args.batch_size)
            loss = measured["summary"]["teacher_kl"]["mean"]
            entry = {"step": step, "validation_kl": loss,
                     "validation_accuracy": measured["summary"]["accuracy"]["mean"]}
            history.append(entry)
            if best is None or loss < best["validation_kl"]:
                best = {**entry, "mask": mask.detach().clone()}
            print("validation", runner.kind, k, entry, flush=True)
        if step == args.steps:
            break
        if cursor + args.batch_size > len(order):
            order = torch.randperm(len(train["ids"]), generator=gen)
            cursor = 0
        indices = order[cursor:cursor + args.batch_size]
        cursor += args.batch_size
        opt.zero_grad(set_to_none=True)
        soft = scores.sigmoid()
        mask = hard_mask(scores, k) - soft.detach() + soft
        logits = runner.forward(train["ids"][indices], train["positions"], mask).float()
        teacher = train["teacher"][indices].to(runner.device)
        lp = logits.log_softmax(-1)
        loss = (teacher.exp() * (teacher - lp)).sum(-1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at {k=} {step=}")
        loss.backward()
        if scores.grad is None or not torch.isfinite(scores.grad).all():
            raise RuntimeError("Mask gradients are missing or nonfinite")
        torch.nn.utils.clip_grad_norm_([scores], 1.)
        opt.step()
        with torch.no_grad():
            scores.clamp_(-6, 6)
    assert int(best["mask"].sum()) == k
    return best, history


def random_mask_like(mask, seed):
    rng = np.random.default_rng(seed)
    random = torch.zeros_like(mask)
    for i, row in enumerate(mask):
        chosen = rng.choice(mask.shape[1], int(row.sum()), replace=False)
        random[i, chosen.tolist()] = 1
    return random


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=["baseline", "dagformer"], required=True)
    ap.add_argument("--root", type=Path, default=Path("checkpoints/pr_sync_20260917"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--budgets", nargs="+", type=int, default=[64, 96, 128, 160])
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--validate-every", type=int, default=30)
    ap.add_argument("--lr", type=float, default=.05)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-valid", type=int, default=16)
    ap.add_argument("--n-test", type=int, default=32)
    ap.add_argument("--n-natural", type=int, default=24)
    ap.add_argument("--controls", type=int, default=3)
    ap.add_argument("--score-positions", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--mask-seed", type=int, default=None,
                    help="Change mask initialization and minibatch ordering while retaining all data")
    ap.add_argument("--verify-projection-only", action="store_true",
                    help="Verify sampled lm_head logits against the unmodified full projection, then exit")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.time()
    corpus, _ = load_eval_ids(str(args.root / "eval_corpora/wikitext_train.pt"))
    natural, labels = load_eval_ids(str(args.root / "eval_corpora/wikitext_test.pt"))
    datasets = {}
    specs = [("train", args.n_train, 512, 128, 0),
             ("validation", args.n_valid, 512, 128, 1),
             ("test_p128_l512", args.n_test, 512, 128, 2),
             ("test_p128_l1024", args.n_test, 1024, 128, 3),
             ("test_p256_l1024", args.n_test, 1024, 256, 4)]
    for name, n, length, period, offset in specs:
        ids, _ = synthetic_induction_ids(corpus, n, length, period, args.seed + offset)
        positions = scored_positions(length, args.score_positions)
        datasets[name] = {"ids": ids, "positions": positions, "targets": ids[:, positions + 1]}
    positions = scored_positions(512, 64, natural=True)
    datasets["natural_wikitext"] = {"ids": natural[:args.n_natural, :512], "positions": positions,
                                    "targets": labels[:args.n_natural, positions]}
    runner = Runner(args.kind, args.root / ("300m-" + args.kind))
    result = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "git_commit_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "complete": False, "architecture": runner.cfg,
        "protocol": {
            "intervention": "Shared binary gate on each backbone attention-head output before o_proj; all MLPs, predictor and correction computations remain active",
            "scope": "Backbone attention-head support, not complete circuit size or native-routing-edge interpretability; no physical head deletion",
            "optimization": "Only mask scores learned; exact top-k forward with sigmoid straight-through gradient; independent optimization per cardinality; checkpoint selected by validation teacher KL",
            "train_eval": "Teacher-forced copying; loss and metrics at fixed uniformly spaced positions in second half, excluding impossible first-block prediction; new random blocks per split",
            "controls": f"{args.controls} binary random masks per learned mask with exactly matching retained-head counts in each layer; learned keep and complementary remove masks both tested",
            "randomness": "One mask-optimization seed per budget; intervals bootstrap independent sequence/window units, not optimization seeds",
            "natural": f"{args.n_natural} WikiText test windows, first512 tokens, 64 output positions/window; capability cost is preliminary and sampled, not full-corpus PPL",
            "fixed_routes": "DAG-only diagnostic uses global-predictor position means from 32 separate discovery-copy sequences; local correction remains live; masks unchanged",
        },
        "dataset_metadata": {name: {"n": len(d["ids"]), "length": d["ids"].shape[1],
                                      "scored_logit_positions": d["positions"].tolist()}
                             for name, d in datasets.items()},
        "checks": {}, "optimization": {}, "arms": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        result["elapsed_seconds"] = time.time() - started
        args.out.write_text(json.dumps(result, indent=2) + "\n")

    tests = {n: d for n, d in datasets.items() if n.startswith("test_") or n.startswith("natural")}
    all_mask = torch.ones(runner.L, runner.H, device=runner.device)
    zero_mask = torch.zeros_like(all_mask)
    with torch.no_grad():
        ids = datasets["validation"]["ids"][:1]
        pos = datasets["validation"]["positions"]
        a = runner.forward(ids, pos, None)
        b = runner.forward(ids, pos, all_mask)
        error = float((a.float() - b.float()).abs().max())
        result["checks"]["all_one_max_logit_error"] = error
        if error != 0:
            raise RuntimeError(f"All-one head hook changed logits: {error}")
        if args.verify_projection_only:
            runner.handles[-1].remove()
            full = runner.forward(ids, pos, None)[:, pos.to(runner.device)]
            result["checks"]["sampled_projection_max_logit_error"] = float((full.float() - a.float()).abs().max())
            result["checks"]["sampled_projection_argmax_agreement"] = float((full.argmax(-1) == a.argmax(-1)).float().mean())
            torch.testing.assert_close(full, a, rtol=0, atol=0)
            result["complete"] = True
            save()
            runner.close()
            print("projection checks", args.kind, result["checks"], flush=True)
            return
    for name, d in datasets.items():
        cache_teacher(runner, d, args.batch_size)
        print("teacher cached", args.kind, name, flush=True)
    del a, b

    def evaluate_arm(name, mask):
        arm = {"retained_heads": int(mask.sum()), "mask": mask.long().cpu().tolist(), "datasets": {}}
        for label, data in tests.items():
            measured = evaluate(runner, data, mask, args.batch_size)
            if "full" in result["arms"]:
                ref = result["arms"]["full"]["datasets"][label]["values"]
                measured["paired_vs_full"] = {k: stats(np.asarray(v) - np.asarray(ref[k]))
                                              for k, v in measured["values"].items()}
            arm["datasets"][label] = measured
        result["arms"][name] = arm
        save()
        print("arm", args.kind, name,
              {n: {k: r["mean"] for k, r in d["summary"].items() if k in ("accuracy", "teacher_kl", "target_nll")}
               for n, d in arm["datasets"].items()}, flush=True)

    evaluate_arm("full", all_mask)
    evaluate_arm("zero", zero_mask)
    learned = {}
    for k in args.budgets:
        if not 0 < k < runner.L * runner.H:
            raise ValueError("Budgets must be between zero and the full backbone-head count")
        best, history = optimize(runner, datasets["train"], datasets["validation"], k, args)
        mask = best.pop("mask")
        learned[k] = mask
        result["optimization"][str(k)] = {"selected": best, "history": history}
        evaluate_arm(f"learned_{k}", mask)
        evaluate_arm(f"remove_learned_{k}", 1 - mask)
        for index in range(args.controls):
            random = random_mask_like(mask, args.seed + k * 100 + index)
            evaluate_arm(f"random_{k}_{index}", random)
            evaluate_arm(f"remove_random_{k}_{index}", 1 - random)

    if runner.predictor is not None:
        rows, _ = synthetic_induction_ids(corpus, 32, 1024, 128, args.seed + 100)
        runner.calibrate_routes(rows, args.batch_size)
        runner.fixed_routes = True
        evaluate_arm("fixed_routes_full", all_mask)
        evaluate_arm("fixed_routes_zero", zero_mask)
        for k, mask in learned.items():
            evaluate_arm(f"fixed_routes_learned_{k}", mask)
    result["complete"] = True
    save()
    runner.close()
    print("complete", args.kind, result["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
