#!/bin/bash
#SBATCH --job-name=extras1g
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=40:00:00
#SBATCH --output=logs/extras1g_%j.out
#SBATCH --signal=B:USR1@600
# One-GPU pretraining run for the post-hoc NAACL extras (D1/D2 at 150M). The trainer resumes from the latest
# checkpoint in save_dir, so a requeue or a second submission of the same config continues the run.
#   sbatch --export=ALL,CFG=configs/extras/150m_muddformer_delta_dolma12bp.yaml,TRAINER=scripts/pretrain_muddformer.py,\
#          EVAL=/work/hdd/bfqt/xiaocong/dagformer_extras/eval_cache_150m_timan12b.pt scripts/slurm/extras_1gpu.slurm
set -uo pipefail
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-offline}" OMP_NUM_THREADS=4
CFG=${CFG:?config yaml}; TRAINER=${TRAINER:-scripts/pretrain_baseline.py}; EVAL=${EVAL:-}
save_dir=$(sed -n 's/^save_dir:[[:space:]]*"\{0,1\}\([^"]*\)"\{0,1\}.*/\1/p' "$CFG")
echo "[$(date '+%F %T')] job $SLURM_JOB_ID on $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-?} cfg=$CFG trainer=$TRAINER save_dir=$save_dir"
mkdir -p "$save_dir"
[ -n "$EVAL" ] && [ ! -f "$save_dir/eval_cache.pt" ] && cp "$EVAL" "$save_dir/eval_cache.pt"
[ -f "$save_dir/DONE" ] && { echo "already DONE"; exit 0; }
# ask the trainer to checkpoint and stop 10 min before the time limit (it polls save_dir/STOP_REQUEST)
trap 'echo "[$(date "+%F %T")] USR1: requesting checkpoint"; touch "$save_dir/STOP_REQUEST"' USR1
rm -f "$save_dir/STOP_REQUEST"
torchrun --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 "$TRAINER" --config "$CFG" &
wait $!
rc=$?
if [ $rc = 0 ] && [ ! -f "$save_dir/STOP_REQUEST" ]; then touch "$save_dir/DONE"; echo "[$(date '+%F %T')] DONE"; fi
echo "exit rc=$rc at $(date)"
exit $rc
