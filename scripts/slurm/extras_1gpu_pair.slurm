#!/bin/bash
#SBATCH --job-name=extras1g_pair
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=40:00:00
#SBATCH --output=logs/extras1g_pair_%j.out
#SBATCH --signal=B:USR1@600
# Two small pretraining runs as co-tenants of ONE GPU (post-hoc NAACL extras, D4 75M learning-rate pair, Delta fallback
# for the timan108 runs after that host went down on 2026-10-05). Same contract as extras_1gpu.slurm, twice: each
# trainer resumes from the latest checkpoint in its save_dir; USR1 asks both to checkpoint and stop; DONE per run.
#   sbatch --nodelist=gpua038 --export=ALL,CFG1=...,TRAINER1=...,CFG2=...,TRAINER2=...,EVAL=... scripts/slurm/extras_1gpu_pair.slurm
set -uo pipefail
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}" OMP_NUM_THREADS=4
CFG1=${CFG1:?config 1}; TRAINER1=${TRAINER1:-scripts/pretrain_baseline.py}
CFG2=${CFG2:?config 2}; TRAINER2=${TRAINER2:-scripts/pretrain_dagformer.py}
EVAL=${EVAL:-}
sd() { sed -n 's/^save_dir:[[:space:]]*"\{0,1\}\([^"]*\)"\{0,1\}.*/\1/p' "$1"; }
d1=$(sd "$CFG1"); d2=$(sd "$CFG2")
echo "[$(date '+%F %T')] job $SLURM_JOB_ID on $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-?}"
echo "  run1: cfg=$CFG1 trainer=$TRAINER1 save_dir=$d1"
echo "  run2: cfg=$CFG2 trainer=$TRAINER2 save_dir=$d2"
pids=()
for i in 1 2; do
  eval "cfg=\$CFG$i; tr=\$TRAINER$i; d=\$d$i"
  mkdir -p "$d"
  [ -n "$EVAL" ] && [ ! -f "$d/eval_cache.pt" ] && cp "$EVAL" "$d/eval_cache.pt"
  rm -f "$d/STOP_REQUEST"
  if [ -f "$d/DONE" ]; then echo "run$i already DONE"; continue; fi
  torchrun --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 "$tr" --config "$cfg" > "$d/train_${SLURM_JOB_ID}.log" 2>&1 &
  pids+=($!); echo "  run$i pid ${pids[-1]} log $d/train_${SLURM_JOB_ID}.log"
done
trap 'echo "[$(date "+%F %T")] USR1: requesting checkpoints"; touch "$d1/STOP_REQUEST" "$d2/STOP_REQUEST"' USR1
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=$?; done
for d in "$d1" "$d2"; do
  [ -f "$d/DONE" ] && continue
  if [ ! -f "$d/STOP_REQUEST" ] && ls "$d"/checkpoint_step*.pt >/dev/null 2>&1; then
    last=$(ls "$d"/checkpoint_step*.pt | sed 's/.*step\([0-9]*\).pt/\1/' | sort -n | tail -1)
    tot=$(sed -n 's/^total_steps:[[:space:]]*\([0-9]*\).*/\1/p' "$(grep -l "$d" "$CFG1" "$CFG2" | head -1)")
    [ "$last" = "$tot" ] && { touch "$d/DONE"; echo "[$(date '+%F %T')] DONE $d"; }
  fi
done
echo "exit rc=$rc at $(date)"
exit $rc
