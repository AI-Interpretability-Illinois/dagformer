"""Keep local GPUs busy; move only held, pending tasks from this study's Slurm array."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


class ExistingRunner:
    def __init__(self, pid, metadata):
        self.pid, self.metadata = pid, Path(metadata)

    def poll(self):
        try:
            os.kill(self.pid, 0)
            stat = Path(f"/proc/{self.pid}/stat").read_text()
            if stat.rsplit(")", 1)[1].split()[0] != "Z":
                return None
        except ProcessLookupError:
            pass
        except FileNotFoundError:
            pass
        self.returncode = 0 if (self.metadata / "DONE.json").exists() else 75
        return self.returncode

    def wait(self):
        while self.poll() is None:
            time.sleep(2)
        return self.returncode


def remote(host, *args):
    return subprocess.check_output(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                    host, shlex.join(args)], text=True, timeout=60)


def take_pending(host, array, index):
    job = f"{array}_{index}"
    state = remote(host, "scontrol", "show", "job", job, "-o")
    fresh = lambda s: (re.search(r"\bJobState=PENDING\b", s)
                       and re.search(r"\bRestarts=0(?:\s|$)", s)
                       and re.search(r"\bRunTime=00:00:00(?:\s|$)", s))
    if not fresh(state):
        return False
    remote(host, "scontrol", "hold", job)
    state = remote(host, "scontrol", "show", "job", job, "-o")
    if (fresh(state)
            and re.search(r"\bReason=JobHeldUser\b", state)
            and re.search(rf"\bArrayTaskId={index}(?:\s|$)", state)):
        remote(host, "scancel", job)
        return True
    remote(host, "scontrol", "release", job)
    return False


def sync_report(args, root):
    snapshot = root / "delta_snapshot"
    snapshot.mkdir(exist_ok=True)
    rsync = ["rsync", "-a", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15"]
    filters = ["--include=/manifest.json", "--include=/runs/", "--include=/runs/*/",
               "--include=*.json", "--include=*.csv", "--include=*.yaml", "--exclude=*"]
    report = root / "report"
    report.mkdir(exist_ok=True)
    sync_path = report / "remote_sync.json"
    sync = json.loads(sync_path.read_text()) if sync_path.exists() else {"last_successful_pull_unix": None}
    sync.update(checked_unix=time.time(), remote_available=False, error=None)
    try:
        subprocess.run([*rsync, *filters, f"{args.host}:{args.remote_root}/", str(snapshot) + "/"],
                       check=True, timeout=90)
    except (subprocess.SubprocessError, OSError) as exc:
        sync["error"] = str(exc)
        print(f"Remote pull unavailable; refreshing local results: {exc}", flush=True)
    else:
        sync.update(remote_available=True, last_successful_pull_unix=time.time())
    sync_path.write_text(json.dumps(sync, indent=2) + "\n")
    collect = [sys.executable, str(ROOT / "scripts/collect_six_axis.py"),
               "--manifest", str(root / "manifest.json")]
    if (snapshot / "manifest.json").exists():
        collect += ["--manifest", str(snapshot / "manifest.json")]
    subprocess.run([*collect, "--legacy", str(root / "legacy"), "--out", str(report)],
                   check=True, timeout=90)
    if not sync["remote_available"]:
        return False
    subprocess.run([*rsync, str(report) + "/", f"{args.host}:{args.remote_root}/report/"],
                   check=True, timeout=90)
    subprocess.run([*rsync, *filters, str(root) + "/", f"{args.host}:{args.remote_root}/local_snapshot/"],
                   check=True, timeout=90)
    subprocess.run([*rsync, "--include=*/", "--include=*.json", "--include=*.yaml", "--include=*.npz", "--exclude=*",
                    str(root / "legacy") + "/", f"{args.host}:{args.remote_root}/legacy/"], check=True, timeout=90)
    if args.remote_checkpoints:
        manifest = json.loads((root / "manifest.json").read_text())
        for run in manifest["runs"]:
            meta = Path(run["metadata_dir"])
            if not (meta / "DONE.json").exists() or (meta / "UPLOADED.json").exists():
                continue
            destination = f"{args.remote_checkpoints}/{run['name']}"
            remote(args.host, "mkdir", "-p", destination)
            subprocess.run([*rsync, "--include=*.pt", "--exclude=*",
                            run["checkpoint_dir"] + "/", f"{args.host}:{destination}/"],
                           check=True, timeout=600)
            (meta / "UPLOADED.json").write_text(json.dumps({
                "destination": destination, "uploaded_unix": time.time()}) + "\n")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--array", type=int, required=True)
    p.add_argument("--host", default="delta")
    p.add_argument("--remote-root", required=True)
    p.add_argument("--gpus", default="3,1", help="First GPU prefers dense; second prefers FourWay")
    p.add_argument("--training-code", required=True)
    p.add_argument("--remote-checkpoints", help="Upload completed local checkpoint groups here")
    p.add_argument("--small-second-gpu", action="store_true",
                   help="Limit routed tasks on the second GPU to L<=6, width<=512")
    args = p.parse_args()
    root = Path(args.manifest).resolve().parent
    manifest = json.loads(Path(args.manifest).read_text())
    lock = (root / "dispatcher.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    taken_path = root / "local_tasks.json"
    taken = json.loads(taken_path.read_text()) if taken_path.exists() else [0, 1]
    attempted, active, failed = set(), {}, {}
    previous = root / "dispatcher_status.json"
    if previous.exists():
        for gpu, entry in json.loads(previous.read_text()).get("active", {}).items():
            cmdline = Path(f"/proc/{entry['pid']}/cmdline")
            if gpu not in args.gpus.split(",") or not cmdline.exists():
                continue
            if b"run_six_axis.py" not in cmdline.read_bytes():
                continue
            index = entry["task"]
            proc = ExistingRunner(entry["pid"], manifest["runs"][index]["metadata_dir"])
            active[gpu] = (index, proc, (root / f"runner_gpu{gpu}.log").open("a"))
            attempted.add(index)
            print(f"GPU {gpu}: adopted task {index}, PID {proc.pid}", flush=True)
    last_sync = 0
    remote_available = True
    while not (root / "STOP_DISPATCH").exists():
        for gpu, (index, proc, log) in list(active.items()):
            if proc.poll() is not None:
                log.close()
                print(f"GPU {gpu}: task {index} exited {proc.returncode}", flush=True)
                if proc.returncode:
                    failed[gpu] = {"task": index, "returncode": proc.returncode}
                del active[gpu]
        for slot, gpu in enumerate(args.gpus.split(",")):
            if gpu in active or gpu in failed:
                continue
            candidates = sorted(range(len(manifest["runs"])), key=lambda i: (i % 2 != slot % 2, i))
            for index in candidates:
                run = manifest["runs"][index]
                if (slot == 1 and args.small_second_gpu and run["family"] == "fourway_corrected"
                        and (run["layers"] > 6 or run["width"] > 512)):
                    continue
                if index in attempted or (Path(run["metadata_dir"]) / "DONE.json").exists():
                    continue
                if index not in taken:
                    if not remote_available:
                        continue
                    try:
                        if not take_pending(args.host, args.array, index):
                            continue
                    except (subprocess.SubprocessError, OSError) as exc:
                        print(f"Cannot claim task {index}: {exc}", flush=True)
                        if (isinstance(exc, subprocess.TimeoutExpired)
                                or getattr(exc, "returncode", None) == 255):
                            remote_available = False
                            break
                        continue
                    taken.append(index)
                    taken_path.write_text(json.dumps(taken) + "\n")
                attempted.add(index)
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
                log = (root / f"runner_gpu{gpu}.log").open("a")
                proc = subprocess.Popen([sys.executable, str(Path(args.training_code) / "scripts/run_six_axis.py"),
                                         "--manifest", args.manifest, "--task", str(index)],
                                        env=env, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = (index, proc, log)
                print(f"GPU {gpu}: started task {index}, PID {proc.pid}", flush=True)
                break
        if time.time() - last_sync > 300:
            try:
                remote_available = sync_report(args, root)
            except (subprocess.SubprocessError, OSError) as exc:
                print(f"Report sync: {exc}", flush=True)
            last_sync = time.time()
        (root / "dispatcher_status.json").write_text(json.dumps({
            "updated_unix": time.time(), "array": args.array, "local_tasks": taken,
            "failed": failed,
            "active": {gpu: {"task": i, "pid": proc.pid} for gpu, (i, proc, _) in active.items()}}, indent=2) + "\n")
        report = root / "report/results.json"
        uploaded = not args.remote_checkpoints or all(
            (Path(r["metadata_dir"]) / "UPLOADED.json").exists()
            for r in manifest["runs"] if (Path(r["metadata_dir"]) / "DONE.json").exists())
        if uploaded and report.exists() and all(r["status"] == "complete" for r in json.loads(report.read_text())["runs"]):
            break
        time.sleep(30)
    for index, proc, log in active.values():
        (Path(manifest["runs"][index]["metadata_dir"]) / "STOP_REQUEST").touch()
        proc.wait()
        log.close()


if __name__ == "__main__":
    main()
