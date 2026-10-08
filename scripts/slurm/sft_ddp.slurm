#!/bin/bash
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
# One multi-GPU SFT task (the 1B models): torchrun over the node's 4 GPUs, same recipe and resumable
# ckpt as the 1-GPU fillers. Submitted by scripts/slurm/submit_sft_ddp.sh (job name sft-ddp-<task>);
# on the time-limit warning the ranks checkpoint, and the job re-queues the task if it is not DONE.
#   args: TASK
set -uo pipefail
TASK=${1:?task name}
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
PY=/u/xiaocong/anaconda3/envs/modularity/bin
log() { echo "[$(date '+%F %T')] $*"; }
log "sft-ddp $SLURM_JOB_ID on $(hostname -s) task $TASK gpus=${CUDA_VISIBLE_DEVICES:-?}"
tr_pid=""
on_usr1() {   # time limit near: ask the rank processes (torchrun's children) to checkpoint and stop
  log "USR1: asking ranks to checkpoint"
  [ -n "$tr_pid" ] && for p in $(pgrep -P "$tr_pid"); do kill -USR1 "$p" 2>/dev/null; done
}
trap on_usr1 USR1
$PY/torchrun --nproc_per_node=4 --master_port=$((29500 + RANDOM % 400)) scripts/sft/sft_train.py --only "$TASK" --job-id "$SLURM_JOB_ID" &
tr_pid=$!
wait "$tr_pid"; rc=$?
while kill -0 "$tr_pid" 2>/dev/null; do wait "$tr_pid"; rc=$?; done   # wait was interrupted by the trap
log "torchrun exited rc=$rc"
bash scripts/slurm/submit_sft_ddp.sh 2>&1 | tail -n 3      # re-queue anything unfinished (incl. this task)
bash scripts/slurm/submit_sft_evals.sh 2>&1 | tail -n 3    # benchmark finished pairs
