#!/bin/bash
#SBATCH --job-name=prune_ft
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/prune_ft_%j.out
#SBATCH --error=logs/prune_ft_%j.err
#SBATCH --signal=B:USR1@120

# Prune-during-finetune, one run per job. Sweep via env vars:
#   CONFIG=configs/prune/150m_dagformer_math.yaml sbatch scripts/slurm/prune_finetune.slurm
#   CONFIG=... OVERRIDES="target_sparsity=0.7 save_dir=/work/.../150m_dagformer_math_s70" \
#       sbatch scripts/slurm/prune_finetune.slurm
# Preemption / time limit: SIGUSR1 triggers a save to <save_dir>/latest and the
# job re-submits itself, resuming from that checkpoint.

set -u
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
mkdir -p logs

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export WANDB_MODE="${WANDB_MODE:-disabled}"

CONFIG="${CONFIG:?set CONFIG=configs/prune/<run>.yaml}"
OVERRIDES="${OVERRIDES:-}"
AUTORESUBMIT="${AUTORESUBMIT:-1}"

args=()
for ov in $OVERRIDES; do args+=(--override "$ov"); done

trap 'echo "SIGUSR1: trainer saves latest/; resubmitting"; \
      if [ "$AUTORESUBMIT" = "1" ]; then sbatch --dependency=afterany:$SLURM_JOB_ID --export=ALL "$0"; fi' SIGUSR1

echo "config=$CONFIG overrides=[$OVERRIDES] job=$SLURM_JOB_ID node=$(hostname) start=$(date)"
python3 scripts/prune_finetune.py --config "$CONFIG" "${args[@]}" &
wait $!
rc=$?
echo "exit rc=$rc at $(date)"
exit $rc
