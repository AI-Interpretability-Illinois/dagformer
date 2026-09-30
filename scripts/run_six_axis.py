"""Run or resume one Slurm array item, or a local sequential subset."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def run_one(run, manifest, minutes):
    meta = Path(run["metadata_dir"])
    meta.mkdir(parents=True, exist_ok=True)
    lock = (meta / "run.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Already running: {run['name']}", flush=True)
        return 0
    if (meta / "DONE.json").exists():
        return 0
    started = time.time()
    deadline = started + minutes * 60 if minutes else float("inf")
    stop_file = meta / "STOP_REQUEST"
    stop_file.unlink(missing_ok=True)
    stop = {"requested": False}

    def request_stop(*_):
        stop["requested"] = True
        stop_file.touch()

    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    checkpoint = Path(run["checkpoint_dir"]) / f"checkpoint_step{run['optimizer_updates']}.pt"
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), WANDB_MODE="disabled", HF_HUB_OFFLINE="1",
               TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    status = {"run": run["name"], "status": "running", "started_unix": started,
              "slurm_job": env.get("SLURM_JOB_ID"), "host": os.uname().nodename,
              "revision": run["revision"]}
    write_json(meta / "status.json", status)
    rc = 0
    if not checkpoint.exists():
        trainer = "pretrain_baseline.py" if run["family"] == "dense" else "pretrain_dagformer.py"
        cmd = [sys.executable]
        if run["world_size"] > 1:
            cmd += ["-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={run['world_size']}"]
        cmd += [str(ROOT / "scripts" / trainer), "--config", run["config"]]
        print(f"Starting {run['name']}", flush=True)
        with (meta / "train.log").open("a", buffering=1) as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                if time.time() > deadline and not stop["requested"]:
                    request_stop()
                time.sleep(2)
            rc = proc.returncode
    elapsed = time.time() - started
    with (meta / "invocations.jsonl").open("a") as f:
        f.write(json.dumps({**status, "training_seconds": elapsed, "returncode": rc,
                            "allocated_gpu_hours": elapsed * run["world_size"] / 3600}) + "\n")
    if rc:
        write_json(meta / "status.json", {**status, "status": "failed", "returncode": rc})
        return rc
    if not checkpoint.exists():
        write_json(meta / "status.json", {**status, "status": "checkpointed"})
        return 75
    cmd = [sys.executable, str(ROOT / "scripts/eval_scaling_checkpoint.py"),
           "--config", run["config"], "--checkpoint", str(checkpoint),
           "--expected-updates", str(run["optimizer_updates"]),
           "--tokens-per-update", str(run["batch_tokens"]),
           "--out", str(meta / "evaluation.json")]
    for name, path in manifest["evals"].items():
        cmd += ["--eval", f"{name}={path}"]
    with (meta / "eval.log").open("a") as log:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc:
        write_json(meta / "status.json", {**status, "status": "eval_failed", "returncode": rc})
        return rc
    done = {**status, "status": "complete", "finished_unix": time.time(),
            "evaluation": str(meta / "evaluation.json")}
    write_json(meta / "DONE.json", done)
    write_json(meta / "status.json", done)
    print(f"Completed {run['name']}", flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--task", type=int)
    p.add_argument("--tasks", help="comma-separated array indices, sequential execution")
    p.add_argument("--minutes", type=float, default=0, help="checkpoint after this many training minutes per invocation")
    args = p.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    indices = [args.task] if args.task is not None else [int(i) for i in args.tasks.split(",")] if args.tasks else range(len(manifest["runs"]))
    for index in indices:
        rc = run_one(manifest["runs"][index], manifest, args.minutes)
        if rc:
            raise SystemExit(rc)


if __name__ == "__main__":
    main()
