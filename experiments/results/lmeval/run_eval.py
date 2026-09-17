#!/usr/bin/env python
"""Run lm-evaluation-harness on the shared DAGFormer / baseline checkpoints.

    # headline reasoning suite for one pair of models
    python run_eval.py --model 300m --suite reasoning

    # everything, into the default results directory
    python run_eval.py --model all --suite reasoning

    # quick end-to-end check of the generative (gsm8k) path
    python run_eval.py --model 75m-dagformer --suite gen --gen-limit 5 --save-samples

One JSON per model lands in ``--out-dir`` (default ``./reasoning/``); feed them
to ``compare.py`` for the baseline-vs-DAGFormer table.

Two evaluation passes run per model, because generative and log-likelihood
tasks have completely different cost profiles:

    log-likelihood pass   one forward per request, batched  -> full test sets
    generative pass       one forward per *token* on the routed model
                          (no KV cache) -> capped by --gen-limit

Both passes share the same model instance, tokenizer, few-shot seed and
context length, so the merged JSON is a single coherent result.
"""
from __future__ import annotations

import argparse
import json
import logging
import importlib.metadata
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RUN_GIT_COMMIT = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from harness import CheckpointLM  # noqa: E402
from models import (  # noqa: E402
    DEFAULT_TOKENIZER,
    MODELS_ROOT,
    checkpoint_step,
    load_model,
    load_tokenizer,
    resolve_models,
)
from suites import SUITES, resolve_tasks  # noqa: E402
from interventions import MODES, install_intervention, intervention_label  # noqa: E402

TASKS_DIR = HERE / "tasks"
DEFAULT_OUT_DIR = HERE / "reasoning"


class TruncationCounter(logging.Handler):
    """Counts lm-eval's left-truncation warnings.

    These models have a 1024-token context, and a 5-shot gsm8k prompt plus room
    for 256 generated tokens can exceed it — in which case lm-eval silently
    drops the front of the prompt.  It applies equally to every model, but it
    is the kind of thing that should show up in the result file rather than
    scroll past in a log.
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.truncated = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "Left truncation applied" in record.getMessage():
            self.truncated += 1


def split_by_output_type(task_manager, tasks: list[str]) -> tuple[list[str], list[str]]:
    """Partition task names into (generative, log-likelihood).

    Read off the indexed YAML config rather than a hardcoded list, so a task
    passed via ``--tasks`` is classified correctly without touching this file.
    """
    generative, loglikelihood = [], []
    index = task_manager.task_index
    for task in tasks:
        entry = index.get(task)
        if entry is None:
            raise SystemExit(
                f"unknown task {task!r} — not in lm-eval's task index nor in {TASKS_DIR}"
            )
        cfg = entry.cfg or {}
        (generative if cfg.get("output_type") == "generate_until" else loglikelihood).append(task)
    return generative, loglikelihood


def merge_results(into: dict, new: dict) -> dict:
    for key in ("results", "n-samples", "versions", "n-shot", "higher_is_better", "group_subtasks"):
        if key in new:
            into.setdefault(key, {}).update(new[key])
    return into


def write_samples(results: dict, out_dir: Path, model_name: str) -> list[str]:
    """Dump per-document model outputs (gsm8k generations are worth eyeballing)."""
    written = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for task, samples in (results.get("samples") or {}).items():
        path = out_dir / f"{model_name}__{task}.jsonl"
        with path.open("w") as f:
            for sample in samples:
                f.write(json.dumps(sample, default=str) + "\n")
        written.append(str(path))
    return written


def evaluate_model(spec, args, task_manager, gen_tasks: list[str], ll_tasks: list[str]) -> dict:
    import lm_eval

    device = torch.device(args.device)
    max_length = args.max_length or spec.train_seq_len

    t_load = time.time()
    model = load_model(spec, device)
    table_path = (args.routing_table.format(size=spec.size, name=spec.name)
                  if args.routing_table else None)
    intervention = install_intervention(model, args.routing_intervention, table_path,
                                        circuit_path=args.routing_circuit, gamma=args.routing_gamma)
    edit_label = intervention_label(args.routing_intervention, args.routing_gamma)
    evaluation_name = spec.name + ("__" + edit_label if intervention else "")
    tokenizer = load_tokenizer(args.tokenizer)
    lm = CheckpointLM(
        model,
        tokenizer,
        batch_size=args.batch_size,
        max_length=max_length,
        max_gen_toks=args.max_gen_toks,
        use_kv_cache=None if not args.no_kv_cache else False,
        softmax_dtype=args.softmax_dtype,
    )
    load_seconds = time.time() - t_load
    print(
        f"[{spec.name}] kind={spec.kind} params={sum(p.numel() for p in model.parameters()):,} "
        f"max_length={max_length} kv_cache={lm.uses_kv_cache} loaded in {load_seconds:.1f}s",
        flush=True,
    )

    counter = TruncationCounter()
    logging.getLogger("lm_eval").addHandler(counter)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # A 5-shot gsm8k prompt averages 872 OLMo tokens and these models have a
    # 1024-token context, so the standard config leaves no room to generate and
    # lm-eval silently left-truncates the prompt. 3-shot (mean 563, p90 713)
    # fits under the 1024 - max_gen_toks budget for ~90% of documents.
    gen_fewshot = args.num_fewshot if args.num_fewshot is not None else args.gen_num_fewshot

    merged: dict = {}
    timings: dict[str, float] = {}
    sample_files: list[str] = []
    passes = [
        ("loglikelihood", ll_tasks, args.limit, args.batch_size, args.num_fewshot),
        ("generative", gen_tasks, args.gen_limit, args.gen_batch_size, gen_fewshot),
    ]
    for label, tasks, limit, batch_size, num_fewshot in passes:
        if not tasks:
            continue
        lm.batch_size_per_gpu = int(batch_size)
        # Likelihood samples retain paired document scores for uncertainty
        # estimates, even when there is no readable generation to inspect.
        save_samples = args.save_samples and (
            label == "generative" or args.save_likelihood_samples)
        print(
            f"[{spec.name}] {label} pass: tasks={tasks} limit={limit} "
            f"batch_size={batch_size} num_fewshot={num_fewshot}",
            flush=True,
        )
        t0 = time.time()
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            task_manager=task_manager,
            bootstrap_iters=args.bootstrap_iters,
            log_samples=save_samples,
            fewshot_random_seed=args.fewshot_seed,
        )
        timings[label] = time.time() - t0
        print(f"[{spec.name}] {label} pass finished in {timings[label]:.1f}s", flush=True)
        if save_samples:
            sample_files += write_samples(results, args.out_dir / "samples", evaluation_name)
            results.pop("samples", None)
        merge_results(merged, results)

    logging.getLogger("lm_eval").removeHandler(counter)

    payload = {
        "model": {
            "name": evaluation_name,
            "family": spec.family,
            "size": spec.size,
            "kind": spec.kind,
            "path": str(spec.path),
            "checkpoint": str(spec.ckpt_path),
            "step": checkpoint_step(spec.ckpt_path),
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "train_seq_len": spec.train_seq_len,
            "routing_intervention": intervention,
        },
        "eval": {
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "suite": args.suite if not args.tasks else "custom",
            "loglikelihood_tasks": ll_tasks,
            "generative_tasks": gen_tasks,
            "limit": args.limit,
            "gen_limit": args.gen_limit,
            "batch_size": args.batch_size,
            "gen_batch_size": args.gen_batch_size,
            "max_length": max_length,
            "max_gen_toks": args.max_gen_toks,
            "num_fewshot": args.num_fewshot,
            "gen_num_fewshot": gen_fewshot,
            "fewshot_seed": args.fewshot_seed,
            "softmax_dtype": args.softmax_dtype,
            "kv_cache": lm.uses_kv_cache,
            "tokenizer": str(args.tokenizer),
            "lm_eval_version": lm_eval.__version__,
            "torch_version": torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
            "git_commit": RUN_GIT_COMMIT,
            "git_commit_recorded_at": "process_start",
            "prompts_left_truncated_batches": counter.truncated,
            "generations_hitting_context_limit": lm.truncated_generations,
            "seconds": timings,
            "peak_gpu_gb": (
                round(torch.cuda.max_memory_allocated(device) / 1024**3, 2)
                if device.type == "cuda"
                else None
            ),
            "sample_files": sample_files,
        },
        **merged,
    }

    del lm, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model",
        nargs="+",
        default=["all"],
        help="checkpoint dir name (300m-dagformer), a size (300m -> both families), "
        "a family (dagformer), or 'all'",
    )
    p.add_argument("--models-root", type=Path, default=MODELS_ROOT)
    p.add_argument("--suite", default="reasoning", choices=sorted(SUITES))
    p.add_argument("--tasks", default=None, help="comma-separated task names; overrides --suite")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    p.add_argument("--device", default="cuda")
    p.add_argument("--routing-intervention", choices=MODES, default="none")
    p.add_argument("--routing-table", default=None,
                   help="calibrated .pt table path; may contain {size} or {name}")
    p.add_argument("--routing-gamma", type=float, default=1.25)
    p.add_argument("--routing-circuit", default="experiments/results/interp/liar_cloze/deception_conns.json")
    p.add_argument("--batch-size", type=int, default=16, help="log-likelihood pass batch size")
    p.add_argument(
        "--gen-batch-size",
        type=int,
        default=1,
        help="generative pass batch size; the routed model decodes rows serially anyway, "
        "and batch 1 keeps the progress bar honest",
    )
    p.add_argument("--limit", type=int, default=None, help="docs per log-likelihood task")
    p.add_argument(
        "--gen-limit",
        type=int,
        default=200,
        help="docs per generative task. Cache-free decoding costs one full-prefix forward "
        "per token, so the full 1319-doc gsm8k test set is hours on the routed models. "
        "The limit takes the first N docs, so every model sees the same subset",
    )
    p.add_argument("--max-length", type=int, default=None, help="default: training seq_len (1024)")
    p.add_argument("--max-gen-toks", type=int, default=256)
    p.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        help="override every task; default is each task's own setting (gsm8k 5-shot, MC 0-shot)",
    )
    p.add_argument(
        "--gen-num-fewshot",
        type=int,
        default=3,
        help="few-shot count for the generative pass only. Default 3 rather than gsm8k's "
        "standard 5 because a 5-shot prompt does not fit these models' 1024-token context "
        "alongside --max-gen-toks; --num-fewshot overrides both passes",
    )
    p.add_argument("--fewshot-seed", type=int, default=1234)
    p.add_argument("--bootstrap-iters", type=int, default=100000)
    p.add_argument("--softmax-dtype", default="float32")
    p.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="force dense baselines through the same cache-free decode as the routed models",
    )
    p.add_argument(
        "--save-samples",
        action="store_true",
        help="write per-doc generations to samples/",
    )
    p.add_argument("--save-likelihood-samples", action="store_true",
                   help="also save likelihood task document scores (requires --save-samples)")
    p.add_argument("--overwrite", action="store_true", help="re-run models that already have a JSON")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    from lm_eval.tasks import TaskManager

    task_manager = TaskManager(include_path=str(TASKS_DIR))
    tasks = resolve_tasks(args.suite, args.tasks)
    gen_tasks, ll_tasks = split_by_output_type(task_manager, tasks)
    specs = resolve_models(args.model, args.models_root)

    suite_label = args.suite if not args.tasks else "custom"
    if args.routing_intervention != "none":
        suite_label += "__" + intervention_label(args.routing_intervention, args.routing_gamma)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"models     : {[s.name for s in specs]}")
    print(f"generative : {gen_tasks}")
    print(f"likelihood : {ll_tasks}")
    print(f"out-dir    : {args.out_dir}")
    if args.dry_run:
        return 0

    failures = []
    for spec in specs:
        out_path = args.out_dir / f"{spec.name}__{suite_label}.json"
        if out_path.exists() and not args.overwrite:
            print(f"[{spec.name}] skip — {out_path} exists (use --overwrite)")
            continue
        try:
            payload = evaluate_model(spec, args, task_manager, gen_tasks, ll_tasks)
        except Exception as exc:  # keep going: one bad checkpoint shouldn't sink the sweep
            logging.exception("[%s] FAILED: %s", spec.name, exc)
            failures.append(spec.name)
            continue
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"[{spec.name}] wrote {out_path}")
        for task, metrics in payload.get("results", {}).items():
            scores = {k: v for k, v in metrics.items() if k != "alias" and "_stderr" not in k}
            print(f"    {task}: {json.dumps(scores, default=str)}")

    if failures:
        print(f"FAILED: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
