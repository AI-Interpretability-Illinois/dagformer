#!/bin/bash
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@300
# job-name, nodelist, reservation, time, nice and output are set by resv_filler_submit.sh

# GPU filler for reservation sup-30781. Runs on any single idle GPU of a reserved
# node and (a) trains the instruction-tuning tasks in configs/sft/tasks.yaml with
# scripts/sft/sft_train.py, or, when no task is left, a light GPU heartbeat, so the
# reservation is never idle (PURGE_COMP=30min); (b) every CHECK_INTERVAL seconds
# asks scripts/slurm/resv_plan.py whether a real job is waiting for the GPU/CPU/
# memory it holds and, if so, checkpoints within GRACE seconds and exits. A
# pending replacement is always queued (one per node) so the GPU is refilled as
# soon as it is idle again.
#
#   Start:   for n in gpua046 gpua047; do scripts/slurm/resv_filler_submit.sh $n; done
#   Stop:    touch ~/.stop_resv_keepalive; scancel -u $USER -n resv-filler-gpua046 -n resv-filler-gpua047
#   Status:  python scripts/slurm/resv_plan.py --status ; squeue -R sup-30781
#   Tasks:   ls /work/hdd/bfqt/xiaocong/dagformer_sft/runs/*/{DONE,LOCK} 2>/dev/null
set -uo pipefail
REPO_DIR=${REPO_DIR:-/u/xiaocong/dagformer}
cd "$REPO_DIR"
source scripts/slurm/resv_lib.sh
PYTHON=${PYTHON:-/u/xiaocong/anaconda3/envs/modularity/bin/python}
CHECK_INTERVAL=${CHECK_INTERVAL:-60}
GRACE=${GRACE:-240}                 # seconds the trainer gets to checkpoint after USR1
YIELD_BEGIN_IN=${YIELD_BEGIN_IN:-300}
TASKS=configs/sft/tasks.yaml
while [ $# -gt 0 ]; do case $1 in --tasks) TASKS=$2; shift 2 ;; *) echo "unknown arg $1" >&2; shift ;; esac; done
NODE=$SLURMD_NODENAME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

log "filler $SLURM_JOB_ID on $NODE gpu=${CUDA_VISIBLE_DEVICES:-?} cpus=$SLURM_CPUS_PER_TASK time=$(squeue -h -j "$SLURM_JOB_ID" -o %l) tasks=$TASKS"
bash scripts/slurm/resv_filler_submit.sh "$NODE"      # keep one pending replacement queued
bash scripts/slurm/submit_sft_ddp.sh 2>&1 | tail -n 2     # a node just freed: queue 4-GPU SFT jobs for finished 1B runs first

worker_pid=""; mode="none"; hb_since=0
RETRY_TASKS_S=${RETRY_TASKS_S:-1800}   # in heartbeat mode, re-read the task list this often
start_trainer() {
  "$PYTHON" scripts/sft/sft_train.py --tasks "$TASKS" --job-id "$SLURM_JOB_ID" &
  worker_pid=$!; mode="train"; log "trainer started pid $worker_pid"
}
start_heartbeat() {
  "$PYTHON" scripts/keepalive_gpu.py --interval 2 --size 4096 --duty 0.1 --log-every 3600 &
  worker_pid=$!; mode="heartbeat"; hb_since=$(date +%s); log "no SFT work left: heartbeat started pid $worker_pid"
}
stop_worker() {   # graceful: USR1 -> wait GRACE -> TERM -> KILL
  [ -n "$worker_pid" ] && kill -0 "$worker_pid" 2>/dev/null || { worker_pid=""; return; }
  kill -USR1 "$worker_pid" 2>/dev/null
  local t=0
  while kill -0 "$worker_pid" 2>/dev/null && [ $t -lt "$GRACE" ]; do sleep 2; t=$((t+2)); done
  if kill -0 "$worker_pid" 2>/dev/null; then log "worker still alive after ${GRACE}s, TERM"; kill -TERM "$worker_pid"; sleep 20; kill -KILL "$worker_pid" 2>/dev/null; fi
  wait "$worker_pid" 2>/dev/null; worker_pid=""
}
leaving=0
leave() {   # REASON BEGIN_IN
  [ "$leaving" = 1 ] && return; leaving=1
  log "leaving: $1"
  stop_worker
  [ -f "$STOP_FILE" ] && { fillers_for "$NODE" PD | xargs -r scancel; log "stop file: pending fillers for $NODE cancelled"; exit 0; }
  bash scripts/slurm/resv_filler_submit.sh "$NODE" --begin-in "$2"
  exit 0
}
trap 'leave "time limit approaching" 0' USR1
trap 'leave "cancelled (TERM)" 0' TERM        # scancel: let the trainer checkpoint first

start_trainer
crashes=0
while :; do
  sleep "$CHECK_INTERVAL" & wait $! 2>/dev/null   # interruptible by the USR1 trap
  [ -f "$STOP_FILE" ] && leave "stop file present" 0
  export MAX_HOURS=${MAX_HOURS:-1}; refresh_pending_holders
  guard_tick                                   # idle alert for both reserved nodes
  scontrol show reservation "$RESV" >/dev/null 2>&1 || { log "reservation gone"; stop_worker; exit 0; }
  if out=$("$PYTHON" scripts/slurm/resv_plan.py --me "$SLURM_JOB_ID" 2>&1); then :; else
    rc=$?; [ $rc -eq 10 ] && leave "real job needs this GPU: $out" "$YIELD_BEGIN_IN"
    log "planner error rc=$rc: $out"
  fi
  if [ "$mode" = heartbeat ] && [ $(( $(date +%s) - hb_since )) -ge "$RETRY_TASKS_S" ]; then
    log "heartbeat mode for ${RETRY_TASKS_S}s: re-checking the task list"; stop_worker; crashes=0; start_trainer
  fi
  if ! kill -0 "$worker_pid" 2>/dev/null; then
    wait "$worker_pid"; rc=$?; worker_pid=""
    if [ "$mode" = train ]; then
      case $rc in
        0) log "trainer exited cleanly (rc 0), checking for more work"
           bash scripts/slurm/submit_sft_evals.sh 2>&1 | tail -n 2  # benchmark finished SFT groups (300M six, 1B pairs)
           bash scripts/slurm/submit_sft_ddp.sh 2>&1 | tail -n 2    # queue 4-GPU jobs for finished 1B runs
           start_trainer ;;
        3) bash scripts/slurm/submit_sft_ddp.sh 2>&1 | tail -n 2
           start_heartbeat ;;
        *) crashes=$((crashes+1)); log "trainer crashed rc=$rc (crash $crashes)"
           if [ $crashes -ge 2 ]; then start_heartbeat; else sleep 30; start_trainer; fi ;;
      esac
    else
      log "heartbeat exited rc=$rc; restarting"; start_heartbeat
    fi
  fi
done
