#!/bin/bash
# Keep exactly one PENDING dry-run (heartbeat) job queued for a reserved node.
#
#   resv_dryrun_submit.sh NODE [--begin-in SECS]
#
# Dry-runs are the fallback that holds an idle GPU when no SFT filler is using
# it: 1 GPU, 2 CPUs, 4G, nice 300 (below the fillers' 200, so a filler that fits
# is always preferred). They yield to real jobs and to pending fillers.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/resv_lib.sh"
REPO_DIR=${REPO_DIR:-/u/xiaocong/dagformer}
JOB_SCRIPT=${JOB_SCRIPT:-$HERE/resv_dryrun.slurm}
DRYRUN_NICE=${DRYRUN_NICE:-300}
export MAX_HOURS=${DRYRUN_MAX_HOURS:-1}

NODE=${1:?usage: $0 NODE [--begin-in SECONDS]}; shift
BEGIN_IN=0
while [ $# -gt 0 ]; do
  case $1 in
    --begin-in) BEGIN_IN=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

if [ -f "$STOP_FILE" ]; then log "stop file $STOP_FILE exists, no dry-run for $NODE" >&2; exit 0; fi
pending=$(holders_for "$DRYRUN_PREFIX" "$NODE" PD)
if [ -n "$pending" ]; then log "dry-run for $NODE already pending ($(tr '\n' ' ' <<<"$pending"))" >&2; exit 0; fi

limit=$(resv_time_limit $(( $(date +%s) + BEGIN_IN )) "$(holder_cap_s "$NODE" "$BEGIN_IN")") || {
  case $? in 1) log "reservation $RESV not found" >&2; exit 1;; *) log "reservation $RESV nearly over, no more dry-runs" >&2; exit 0;; esac; }
OPTS=()
[ "$BEGIN_IN" -gt 0 ] && OPTS+=(--begin="now+${BEGIN_IN}")
# a pending multi-GPU real job needs this node: queue held (sync_holder_holds releases it later)
claimed_nodes | grep -qx "$NODE" && OPTS+=(--hold)
mkdir -p "$REPO_DIR/logs"
log "sbatch dry-run for $NODE --time=$limit ${OPTS[*]:-}" >&2
sbatch_with_fallback --parsable --export=NONE --job-name="$DRYRUN_PREFIX-$NODE" --reservation="$RESV" \
  --nodelist="$NODE" --time="$limit" --nice="$DRYRUN_NICE" --chdir="$REPO_DIR" \
  --output="logs/dryrun_${NODE}_%j.out" --error="logs/dryrun_${NODE}_%j.out" \
  "${OPTS[@]}" "$JOB_SCRIPT"
