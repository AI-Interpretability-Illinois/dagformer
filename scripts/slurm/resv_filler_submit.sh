#!/bin/bash
# Keep exactly one PENDING filler job queued for a reserved node.
#
#   resv_filler_submit.sh NODE                  queue a filler for NODE if none is pending
#   resv_filler_submit.sh NODE --begin-in SECS  ... eligible to start only after SECS
#   resv_filler_submit.sh NODE --tasks FILE     ... run this task list instead of configs/sft/tasks.yaml
#                                               (not inherited by the replacement fillers it queues)
#
# Fillers are 1-GPU jobs (see resv_filler.slurm) submitted with a large --nice so
# every real job outranks them; SLURM backfills them onto idle GPUs. When one
# starts or ends it calls this script again, so a pending filler is always ready
# to take the next idle GPU.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/resv_lib.sh"
REPO_DIR=${REPO_DIR:-/u/xiaocong/dagformer}
JOB_SCRIPT=${JOB_SCRIPT:-$HERE/resv_filler.slurm}
# nice 200 puts fillers ~200 below the group's real jobs (~800) yet inside the top
# bf_max_job_test=5000 of the cluster queue; at nice 100000 backfill never tested them.
FILLER_NICE=${FILLER_NICE:-200}
# Short links: backfill only starts a low-priority job if it cannot delay a queued real
# job's projected start, so a 4h filler fits into gaps that a 48h one would not. Fillers
# are resumable and re-queue themselves, so short links cost only a checkpoint reload.
export MAX_HOURS=${FILLER_MAX_HOURS:-1}

NODE=${1:?usage: $0 NODE [--begin-in SECONDS] [--tasks FILE]}; shift
BEGIN_IN=0; TASKS_ARGS=()
while [ $# -gt 0 ]; do
  case $1 in
    --begin-in) BEGIN_IN=$2; shift 2 ;;
    --tasks)    TASKS_ARGS=(--tasks "$2"); shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

if [ -f "$STOP_FILE" ]; then log "stop file $STOP_FILE exists, no filler for $NODE" >&2; exit 0; fi
pending=$(fillers_for "$NODE" PD)
if [ -n "$pending" ]; then log "filler for $NODE already pending ($(tr '\n' ' ' <<<"$pending"))" >&2; exit 0; fi

limit=$(resv_time_limit $(( $(date +%s) + BEGIN_IN )) "$(holder_cap_s "$NODE" "$BEGIN_IN")") || {
  case $? in 1) log "reservation $RESV not found" >&2; exit 1;; *) log "reservation $RESV nearly over, no more fillers" >&2; exit 0;; esac; }
OPTS=()
[ "$BEGIN_IN" -gt 0 ] && OPTS+=(--begin="now+${BEGIN_IN}")
# a pending multi-GPU real job needs this node: queue held (sync_holder_holds releases it later)
claimed_nodes | grep -qx "$NODE" && OPTS+=(--hold)
mkdir -p "$REPO_DIR/logs"
log "sbatch filler for $NODE --time=$limit ${OPTS[*]:-}" >&2
sbatch_with_fallback --parsable --export=NONE --job-name="$FILLER_PREFIX-$NODE" --reservation="$RESV" \
  --nodelist="$NODE" --time="$limit" --nice="$FILLER_NICE" --chdir="$REPO_DIR" \
  --output="logs/filler_${NODE}_%j.out" --error="logs/filler_${NODE}_%j.out" \
  "${OPTS[@]}" "$JOB_SCRIPT" "${TASKS_ARGS[@]}"
