#!/bin/bash
#SBATCH --job-name=lmeval-reasoning
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --output=logs/lmeval_reasoning_%j.out
#SBATCH --error=logs/lmeval_reasoning_%j.err

# Reasoning-first lm-eval sweep over the shared DAGFormer / baseline checkpoints.
#
#   sbatch slurm_lmeval_reasoning.slurm
#   MODELS="300m" SUITE=math GEN_LIMIT=500 sbatch slurm_lmeval_reasoning.slurm
#
# Run `python prefetch_data.py --suite all` from a login node first: this job
# sets HF_HUB_OFFLINE=1 so a missing dataset fails immediately instead of
# hanging on a compute node that may have no outbound network.

set -uo pipefail
cd /u/xiaocong/dagformer/experiments/results/lmeval
mkdir -p logs reasoning

PYTHON=/u/xiaocong/anaconda3/envs/modularity/bin/python

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODELS="${MODELS:-all}"
SUITE="${SUITE:-reasoning}"
GEN_LIMIT="${GEN_LIMIT:-200}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OUT_DIR="${OUT_DIR:-reasoning}"
EXTRA_ARGS="${EXTRA_ARGS:---save-samples --overwrite}"

echo "host:   $(hostname)"
echo "models: $MODELS"
echo "suite:  $SUITE"
echo "start:  $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# shellcheck disable=SC2086
$PYTHON run_eval.py \
    --model $MODELS \
    --suite "$SUITE" \
    --batch-size "$BATCH_SIZE" \
    --gen-limit "$GEN_LIMIT" \
    --out-dir "$OUT_DIR" \
    $EXTRA_ARGS
rc=$?

echo "=================== comparison ==================="
$PYTHON compare.py --dir "$OUT_DIR" --write || true

echo "finished: $(date) rc=$rc"
exit "$rc"
