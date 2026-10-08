#!/bin/bash
# Queue one 4-GPU reservation job (scripts/slurm/sft_ddp.slurm) for every SFT task that needs more
# than one GPU (the 1B models: min_gpus in the task list), is not DONE, is not held by a running job,
# and has no sft-ddp job queued yet. Jobs are submitted in task priority order. Safe to call often:
# the fillers, the DDP jobs themselves and the GPU coordinator all call it.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/resv_lib.sh"
REPO_DIR=${REPO_DIR:-/u/xiaocong/dagformer}
cd "$REPO_DIR"
[ -f "$STOP_FILE" ] && exit 0
# One submitter at a time: fillers on both nodes and the coordinator call this concurrently, and two
# callers that both read "nothing queued" would queue every task twice (happened 2026-09-30 14:38).
# mkdir is atomic on the shared home filesystem; a lock older than 10 min is from a dead caller.
LOCKD="$STATE_DIR/submit_sft_ddp.lock"
mkdir -p "$STATE_DIR"
if ! mkdir "$LOCKD" 2>/dev/null; then
  age=$(( $(date +%s) - $(stat -c %Y "$LOCKD" 2>/dev/null || date +%s) ))
  [ "$age" -gt 600 ] || exit 0
  rm -rf "$LOCKD"; mkdir "$LOCKD" 2>/dev/null || exit 0
fi
trap 'rm -rf "$LOCKD"' EXIT
PY=/u/xiaocong/anaconda3/envs/modularity/bin/python
# the calling sft-ddp job (re-queuing its own interrupted task on the way out) does not count as queued
# optional discover.ddp_nodelist in configs/sft/tasks.yaml pins the 4-GPU jobs to one node (2026-10-02: gpua038,
# so gpua047 stays free for 1-GPU analysis jobs)
ddp_node=$($PY -c 'import yaml; print((yaml.safe_load(open("configs/sft/tasks.yaml")).get("discover") or {}).get("ddp_nodelist") or "")' 2>/dev/null)
NODE_OPT=(); [ -n "$ddp_node" ] && NODE_OPT=(--nodelist="$ddp_node")
queued=$(squeue -h -u "$USER" -o "%i %j" 2>/dev/null | awk -v me="${SLURM_JOB_ID:-none}" '$1 != me && $2 ~ /^sft-ddp-/ {sub(/^sft-ddp-/, "", $2); print $2}')
HF_HUB_OFFLINE=1 $PY scripts/sft/sft_train.py --list-json 2>/dev/null | $PY -c '
import json, os, re, sys, yaml
hold = (yaml.safe_load(open("configs/sft/tasks.yaml")).get("discover") or {}).get("ddp_hold") or {}
for line in sys.stdin:
    try: t = json.loads(line)
    except Exception: continue
    if any(re.search(k, t["name"]) and not os.path.exists(v) for k, v in hold.items()):
        continue
    st = t["status"]
    if st == "RUNNING@" + os.environ.get("SLURM_JOB_ID", "none"):   # our own task, checkpointed on the way out
        st = "RESUMABLE"
    if t["min_gpus"] > 1 and (st in ("todo", "RESUMABLE", "STALE-LOCK")):
        print(t["name"], t["priority"])' | while read -r task prio; do
  grep -qx "$task" <<<"$queued" && continue
  limit=$(MAX_HOURS=24 resv_time_limit "$(date +%s)") || { log "sft-ddp: reservation nearly over, not submitting $task"; continue; }
  # task priority -> SLURM nice, so a higher-priority task submitted later (e.g. a 10B run that finishes
  # after a 5B one is already queued) still starts first: (priority-20)*60, i.e. p20 -> 0, p25 -> 300
  nice=$(( prio > 20 ? (prio - 20) * 60 : 0 ))
  mkdir -p logs
  jid=$(sbatch_with_fallback --parsable --reservation="$RESV" "${NODE_OPT[@]}" --time="$limit" --nice="$nice" --job-name="sft-ddp-$task" \
        --chdir="$REPO_DIR" --output="logs/sft_ddp_${task}_%j.out" --error="logs/sft_ddp_${task}_%j.out" \
        "$HERE/sft_ddp.slurm" "$task") && log "sft-ddp: queued $task as job $jid (--time=$limit --nice=$nice)"
done
exit 0
