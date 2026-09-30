#!/bin/bash
#SBATCH --job-name=prune_smoke
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=00:45:00
#SBATCH --output=logs/prune_smoke_%j.out
#SBATCH --error=logs/prune_smoke_%j.err

# GPU smoke of the prune-during-finetune trainer on the real 75M pair:
# 60 steps, pruning at 20..40, tiny eval. Checks loading, speed, memory and
# that the pruned checkpoint round-trips, before the sweep is submitted.
set -u
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
mkdir -p logs
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 WANDB_MODE=disabled
ROOT=/work/hdd/bfqt/xiaocong/dagformer_pruning/checkpoints/smoke
rc=0
for fam in baseline dagformer; do
    echo "=== $fam ($(date)) ==="
    python3 scripts/prune_finetune.py --config configs/prune/75m_${fam}_math.yaml \
        --override total_steps=60 --override prune_start_step=20 --override prune_end_step=40 \
        --override prune_every=10 --override warmup_steps=5 --override eval_every=30 \
        --override eval_max_batches=8 --override save_every=0 \
        --override save_dir=$ROOT/75m_${fam} || rc=$?
done
echo "smoke rc=$rc ($(date))"
exit $rc
