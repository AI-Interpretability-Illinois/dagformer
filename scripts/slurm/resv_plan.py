#!/usr/bin/env python
"""Decide which filler jobs must give way to real jobs in the reservation.

Three classes of job live in the reservation:
  real     anything not named resv-filler-*/resv-dryrun-*/resv-anchor-* (any user)
  filler   resv-filler-*  1-GPU SFT jobs (scripts/slurm/resv_filler.slurm)
  dryrun   resv-dryrun-*  1-GPU heartbeat fallback (scripts/slurm/resv_dryrun.slurm)

Pass 1: real jobs pending for Resources/Priority, in priority order: find the
cheapest set of nodes where each fits once some holders leave (GPUs, CPUs and
memory all checked; dry-runs leave before fillers, youngest first). Pass 2: the
same for pending fillers, which may displace dry-runs only. Running real jobs are
never disturbed. A pending job that still has not started STUCK_AFTER_S seconds
after holders first yielded for it is assumed blocked by something else and is
ignored from then on.

    python resv_plan.py --me $SLURM_JOB_ID    exit 10 if this job should yield
    python resv_plan.py --status              human-readable view
    python resv_plan.py --claimed             nodes whose pending holders must stay held
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

RESV = os.environ.get("RESV", "sup-30781")
FILLER_PREFIX = os.environ.get("FILLER_PREFIX", "resv-filler")
DRYRUN_PREFIX = os.environ.get("DRYRUN_PREFIX", "resv-dryrun")
IGNORE_PREFIXES = tuple(os.environ.get("PLAN_IGNORE_PREFIXES", "resv-filler,resv-dryrun,resv-anchor").split(","))
STATE_DIR = os.environ.get("STATE_DIR", os.path.expanduser("~/.resv_keepalive"))
STUCK_AFTER_S = int(os.environ.get("STUCK_AFTER_S", "1800"))
ELIGIBLE = {"Resources", "Priority", "None"}


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout


def parse_mem_mb(s: str, cpus_per_node: int) -> float:
    """'64G' | '200G' | '2000M' | '1000Mc' (per cpu) | '0' -> MB per node."""
    m = re.match(r"^\s*([\d.]+)([KMGT]?)(c?)\s*$", s or "0")
    if not m:
        return 0.0
    v, unit, per_cpu = float(m.group(1)), m.group(2), m.group(3)
    v *= {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024, "": 1}[unit]
    return v * cpus_per_node if per_cpu else v


def parse_gpus(tres: str) -> int:
    m = re.search(r"gres/gpu(?::[a-z0-9_]+)?[:=](\d+)", tres or "")
    return int(m.group(1)) if m else 0


def parse_tres(s: str) -> dict:
    out = {"cpu": 0, "mem": 0.0, "gpu": 0}
    for kv in (s or "").split(","):
        if kv.startswith("cpu="):
            out["cpu"] = int(kv[4:])
        elif kv.startswith("mem="):
            out["mem"] = parse_mem_mb(kv[4:], 1)
        elif kv.startswith("gres/gpu"):
            out["gpu"] = parse_gpus(kv)
    return out


def hostnames(expr: str) -> list[str]:
    return sh(["scontrol", "show", "hostnames", expr]).split() if expr else []


def reservation_nodes() -> list[str]:
    out = sh(["scontrol", "show", "reservation", RESV])
    m = re.search(r"Nodes=(\S+)", out)
    return hostnames(m.group(1)) if m else []


def node_state(nodes: list[str]) -> dict[str, dict]:
    st = {}
    for line in sh(["scontrol", "show", "node", "-o", ",".join(nodes)]).splitlines():
        name = re.search(r"NodeName=(\S+)", line)
        cfg = re.search(r"CfgTRES=(\S+)", line)
        alloc = re.search(r"AllocTRES=(\S*)", line)
        if not name:
            continue
        c, a = parse_tres(cfg.group(1) if cfg else ""), parse_tres(alloc.group(1) if alloc else "")
        # MemSpecLimit is held back for the OS and never allocatable (gpua038: 8450M), so a job
        # that looks like it fits in CfgTRES-AllocTRES can still pend on Resources
        spec = re.search(r"MemSpecLimit=(\d+)", line)
        if spec:
            c["mem"] -= float(spec.group(1))
        st[name.group(1)] = {"free": {k: c[k] - a[k] for k in c}, "cfg": c}
    return st


def jobs() -> list[dict]:
    rows = []
    fmt = "%i|%T|%j|%Q|%D|%C|%m|%b|%n|%r|%N|%u"
    for line in sh(["squeue", "-h", "-R", RESV, "-o", fmt]).splitlines():
        f = line.split("|")
        if len(f) < 12:
            continue
        nodes = int(str(f[4]).split("-")[0] or 1)
        cpus_total = int(f[5] or 0)
        cpn = max(1, cpus_total // max(nodes, 1))
        rows.append({"id": f[0], "state": f[1], "name": f[2], "prio": int(f[3] or 0), "nodes": nodes,
                     "cpn": cpn, "mem": parse_mem_mb(f[6], cpn), "gpn": parse_gpus(f[7]),
                     "req": hostnames(f[8]) if f[8] else [], "reason": f[9],
                     "nodelist": hostnames(f[10]) if f[10] else [], "user": f[11]})
    return rows


def load_state() -> dict:
    try:
        with open(os.path.join(STATE_DIR, "plan_yielded.json")) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(st: dict) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    p = os.path.join(STATE_DIR, "plan_yielded.json")
    tmp = f"{p}.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, p)


def _place(j: dict, nodes: list[str], free: dict, holders: dict, yield_set: set, may_yield, notes: list,
           claimed: set | None = None) -> bool:
    """Fit pending job j; mark the holders that must leave. Returns True if placed.
    Nodes where a multi-GPU job (or one short of memory/CPUs, not GPUs) still needs holders to leave go into
    ``claimed``: holders there leave one by one, and backfill would restart a pending holder in each freed GPU
    before the job fits (2026-09-30)."""
    need = {"gpu": j["gpn"], "cpu": j["cpn"], "mem": j["mem"]}
    options = []
    for n in (j["req"] or nodes):
        if n not in free:
            continue
        f = dict(free[n]["free"])
        to_yield = []
        for h in holders[n]:
            if all(f[k] >= need[k] for k in need):
                break
            if h["id"] in yield_set or not may_yield(h):
                continue
            to_yield.append(h)
            f["gpu"] += h["gpn"]; f["cpu"] += h["cpn"]; f["mem"] += h["mem"]
        if all(f[k] >= need[k] for k in need):
            options.append((len(to_yield), -f["gpu"], n, to_yield))
    options.sort()
    if len(options) < j["nodes"]:
        notes.append(f"job {j['id']} {j['name']}: cannot fit even without fillers, ignored")
        return False
    for _, _, n, to_yield in options[:j["nodes"]]:
        # claim the node (pending holders held) for every real job placed there, whether holders must yield
        # or the GPUs are already free: a pending holder outranks a low-priority real job (owner's rule of
        # 2026-10-05 that MoE jobs at priority 1 outrank holders at ~250) and would take the slot first.
        # Until 2026-10-05 only multi-GPU and memory/CPU-short jobs claimed (2026-10-01: a 64G lie_depth_phi
        # lost a filler's 24G to the next filler); two races on 2026-10-05 (a yielded GPU and a freed one both
        # refilled under a 1-GPU MoE job) showed every placement needs it. The stuck timeout below releases
        # a claim whose job never starts, so holders are never held longer than STUCK_AFTER_S.
        if claimed is not None:
            claimed.add(n)
        for h in to_yield:
            yield_set.add(h["id"])
            for k, kk in (("gpu", "gpn"), ("cpu", "cpn"), ("mem", "mem")):
                free[n]["free"][k] += h[kk]
        for k in need:
            free[n]["free"][k] -= need[k]
        notes.append(f"job {j['id']} {j['name']} ({j['user']}, gpu={need['gpu']} cpu={need['cpu']} "
                     f"mem={need['mem']:.0f}M) -> {n}, yield {[x['id'] for x in to_yield] or 'nothing'}")
    return True   # placed; the caller starts the stuck timer at first placement, yield or not


def plan(verbose: bool = False, claimed: set | None = None) -> tuple[set[str], list[str]]:
    notes: list[str] = []
    nodes = reservation_nodes()
    if not nodes:
        return set(), ["reservation not found"]
    free = node_state(nodes)
    all_jobs = jobs()
    for j in all_jobs:
        j["cls"] = ("filler" if j["name"].startswith(FILLER_PREFIX) else
                    "dryrun" if j["name"].startswith(DRYRUN_PREFIX) else
                    "ignored" if j["name"].startswith(IGNORE_PREFIXES) else "real")
    holders = {n: [] for n in nodes}          # running fillers/dry-runs per node
    for j in all_jobs:
        if j["cls"] in ("filler", "dryrun") and j["state"] in ("RUNNING", "COMPLETING") and j["nodelist"]:
            holders[j["nodelist"][0]].append(j)
    for n in holders:                          # dry-runs first, then youngest first
        holders[n].sort(key=lambda j: (0 if j["cls"] == "dryrun" else 1, -int(j["id"])))

    def eligible(cls: str) -> list[dict]:
        # "ReqNodeNotAvail, May be reserved for other job" is what a multi-GPU reservation job shows while
        # another real job holds part of its node (2026-09-30: 4-GPU sft-ddp behind a 1-GPU sft-eval);
        # without it the planner ignores the job and fillers backfill each GPU as it frees.
        p = [j for j in all_jobs if j["state"] == "PENDING" and j["cls"] == cls
             and (j["reason"] in ELIGIBLE or j["reason"].startswith("ReqNodeNotAvail"))]
        p.sort(key=lambda j: (-j["prio"], int(j["id"])))
        return p

    state = load_state()
    now = time.time()
    pend_ids = {j["id"] for j in all_jobs if j["state"] == "PENDING"}
    state = {k: v for k, v in state.items() if k in pend_ids}       # forget started/gone jobs
    yield_set: set[str] = set()
    for cls, may_yield in (("real", lambda h: True), ("filler", lambda h: h["cls"] == "dryrun")):
        for j in eligible(cls):
            first = state.get(j["id"])
            if first and now - first > STUCK_AFTER_S:
                notes.append(f"job {j['id']} {j['name']}: still pending {int((now-first)/60)} min after placement, ignored")
                continue
            if _place(j, nodes, free, holders, yield_set, may_yield, notes,
                      claimed if cls == "real" else None) and j["id"] not in state:
                state[j["id"]] = now
    save_state(state)
    if verbose:
        for n in nodes:
            fr = free[n]["free"]
            notes.append(f"{n}: after plan free gpu={fr['gpu']} cpu={fr['cpu']} mem={fr['mem']:.0f}M; "
                         f"holders={[(h['id'], h['cls']) for h in holders[n]]}")
    return yield_set, notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--me", default="", help="this filler's job id; exit 10 if it must yield")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--claimed", action="store_true",
                    help="print nodes where a pending multi-GPU real job still needs holders to leave (hold pending holders there)")
    args = ap.parse_args()
    claimed: set[str] = set()
    ys, notes = plan(verbose=args.status or bool(args.me), claimed=claimed)
    if args.claimed:
        for n in sorted(claimed):
            print(n)
        return 0
    if args.status or not args.me:
        print("yield:", sorted(ys) or "nobody")
        for n in notes:
            print(" ", n)
        return 0
    if args.me in ys:
        print(f"yield {args.me}: " + "; ".join(notes))
        return 10
    return 0


if __name__ == "__main__":
    sys.exit(main())
