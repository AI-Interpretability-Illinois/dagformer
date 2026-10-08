#!/bin/bash
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
# job-name, nodelist, reservation, time, nice and output are set by resv_dryrun_submit.sh

# Dry-run fallback for reservation sup-30781: holds one otherwise idle GPU with a
# light heartbeat (scripts/keepalive_gpu.py) so the reservation is never idle
# (PURGE_COMP=30min) even if the SFT fillers fail to submit, schedule or run.
# Every CHECK_INTERVAL seconds it asks scripts/slurm/resv_plan.py whether a real
# job or a pending filler needs its GPU and exits at once if so, re-queuing itself
# with a 5-minute delay. If the planner itself is broken it falls back to a crude
# rule: leave when any non-dry-run job is pending for resources in the reservation.
# Deliberately depends on nothing but bash, squeue/scontrol and (optionally) torch.
#
#   Start:  for n in gpua046 gpua047; do scripts/slurm/resv_dryrun_submit.sh $n; done
#   Stop:   touch ~/.stop_resv_keepalive; scancel -u $USER -n resv-dryrun-gpua046 -n resv-dryrun-gpua047
set -uo pipefail
REPO_DIR=${REPO_DIR:-/u/xiaocong/dagformer}
cd "$REPO_DIR" || exit 1
source scripts/slurm/resv_lib.sh || { RESV=sup-30781; DRYRUN_PREFIX=resv-dryrun; STOP_FILE=$HOME/.stop_resv_keepalive; log() { echo "[$(date '+%F %T')] $*"; }; }
PYTHON=${PYTHON:-/u/xiaocong/anaconda3/envs/modularity/bin/python}
CHECK_INTERVAL=${CHECK_INTERVAL:-60}
YIELD_BEGIN_IN=${YIELD_BEGIN_IN:-300}
NODE=$SLURMD_NODENAME

log "dry-run $SLURM_JOB_ID on $NODE gpu=${CUDA_VISIBLE_DEVICES:-?} time=$(squeue -h -j "$SLURM_JOB_ID" -o %l)"
bash scripts/slurm/resv_dryrun_submit.sh "$NODE"        # keep one pending replacement queued

hb_pid=""
start_heartbeat() {
  if [ -x "$PYTHON" ] && [ -f scripts/keepalive_gpu.py ]; then
    "$PYTHON" scripts/keepalive_gpu.py --interval 2 --size 4096 --duty 0.1 --log-every 3600 &   # ~10% busy: visibly held, negligible power
  else
    log "no heartbeat python available; holding the allocation idle"; sleep infinity &
  fi
  hb_pid=$!
}
leaving=0
leave() {   # REASON BEGIN_IN
  [ "$leaving" = 1 ] && return; leaving=1
  log "leaving: $1"
  [ -n "$hb_pid" ] && { kill "$hb_pid" 2>/dev/null; wait "$hb_pid" 2>/dev/null; }
  [ -f "$STOP_FILE" ] && { squeue -h -u "$USER" -n "$DRYRUN_PREFIX-$NODE" -t PD -o %i | xargs -r scancel; exit 0; }
  bash scripts/slurm/resv_dryrun_submit.sh "$NODE" --begin-in "$2"
  exit 0
}
trap 'leave "time limit approaching" 0' USR1 TERM

# crude fallback used only when the planner cannot run: any non-dry-run job waiting for
# resources in the reservation gets our GPU.
crude_should_yield() {
  squeue -h -R "$RESV" -t PD -o "%j|%r" 2>/dev/null | grep -v "^$DRYRUN_PREFIX" | grep -qE "\|(Resources|Priority|None)$"
}

start_heartbeat
while :; do
  sleep "$CHECK_INTERVAL" & wait $! 2>/dev/null
  [ -f "$STOP_FILE" ] && leave "stop file present" 0
  export MAX_HOURS=${MAX_HOURS:-1}; refresh_pending_holders
  guard_tick                                   # idle alert for both reserved nodes
  scontrol show reservation "$RESV" >/dev/null 2>&1 || { log "reservation gone"; kill "$hb_pid" 2>/dev/null; exit 0; }
  if out=$("$PYTHON" scripts/slurm/resv_plan.py --me "$SLURM_JOB_ID" 2>&1); then :; else
    rc=$?
    if [ $rc -eq 10 ]; then leave "GPU wanted: $out" "$YIELD_BEGIN_IN"; fi
    log "planner failed rc=$rc ($out); using crude rule"
    crude_should_yield && leave "crude rule: a job is pending for resources" "$YIELD_BEGIN_IN"
  fi
  if ! kill -0 "$hb_pid" 2>/dev/null; then log "heartbeat exited; restarting"; start_heartbeat; fi
done
