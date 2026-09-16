#!/usr/bin/env python
"""Download (and sanity-check) every dataset a suite needs.

Compute nodes are not guaranteed outbound network, and a job that dies 40
minutes in because ``openbookqa`` was not cached is a wasted allocation.  Run
this once from a login node before submitting:

    python prefetch_data.py --suite all

It resolves each task's ``dataset_path``/``dataset_name`` straight from the
lm-eval task index (including the custom tasks in ``tasks/``), pulls it into
``HF_HOME``, and prints a per-task OK/FAIL line.  A FAIL here is the cheap
place to find out that a task needs ``trust_remote_code`` or that its hub repo
moved.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from suites import SUITES, resolve_tasks  # noqa: E402

TASKS_DIR = HERE / "tasks"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="all", choices=sorted(SUITES))
    p.add_argument("--tasks", default=None, help="comma-separated task names; overrides --suite")
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="allow datasets that ship a loading script (executes code from the hub)",
    )
    args = p.parse_args(argv)

    import datasets
    from lm_eval.tasks import TaskManager

    task_manager = TaskManager(include_path=str(TASKS_DIR))
    index = task_manager.task_index
    tasks = resolve_tasks(args.suite, args.tasks)

    failures = []
    seen: set[tuple] = set()
    for task in tasks:
        entry = index.get(task)
        if entry is None or not entry.cfg:
            print(f"FAIL {task}: not in the task index")
            failures.append(task)
            continue
        cfg = entry.cfg
        key = (cfg.get("dataset_path"), cfg.get("dataset_name"))
        if key in seen:
            print(f"OK   {task}: {key[0]} (already fetched)")
            continue
        try:
            datasets.load_dataset(
                cfg["dataset_path"],
                cfg.get("dataset_name"),
                trust_remote_code=args.trust_remote_code or None,
                **(cfg.get("dataset_kwargs") or {}),
            )
            seen.add(key)
            print(f"OK   {task}: {key[0]}" + (f"/{key[1]}" if key[1] else ""))
        except Exception as exc:  # noqa: BLE001 — we want the reason, per task
            print(f"FAIL {task}: {key[0]}: {type(exc).__name__}: {str(exc).splitlines()[0]}")
            failures.append(task)

    if failures:
        print(f"\n{len(failures)} task(s) could not be fetched: {failures}", file=sys.stderr)
        return 1
    print(f"\nall {len(tasks)} task datasets cached")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
