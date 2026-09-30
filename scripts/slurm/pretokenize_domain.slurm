#!/bin/bash
#SBATCH --job-name=pretok_domain
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=01:30:00
#SBATCH --output=logs/pretok_domain_%j.out
#SBATCH --error=logs/pretok_domain_%j.err

# Build the domain finetuning corpora for the pruning experiment (all from the
# local HF cache, fully offline). One GPU is requested only because this
# account cannot submit CPU-only jobs.
#
#   sbatch scripts/slurm/pretokenize_domain.slurm
#   RECIPES="gsm8k" sbatch scripts/slurm/pretokenize_domain.slurm

set -u
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
mkdir -p logs

export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

DATA_ROOT="${DATA_ROOT:-/work/hdd/bfqt/xiaocong/dagformer_pruning/data}"
RECIPES="${RECIPES:-mathinstruct gsm8k wikitext2}"

rc=0
for r in $RECIPES; do
    echo "=== $r ($(date)) ==="
    python3 scripts/pretokenize_domain.py --recipe "$r" --out-dir "$DATA_ROOT/$r" || rc=$?
done
echo "done rc=$rc ($(date))"
exit $rc
