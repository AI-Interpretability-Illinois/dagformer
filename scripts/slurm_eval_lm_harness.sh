#!/bin/bash
#SBATCH --job-name=lmeval
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lmeval_%j.out
#SBATCH --error=logs/lmeval_%j.err

set -euo pipefail
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs experiments/results/lmeval

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

# Usage:
#   sbatch scripts/slurm_eval_lm_harness.sh <config.yaml> <ckpt.pt> <output_name>
# Example:
#   sbatch scripts/slurm_eval_lm_harness.sh \
#       configs/pretrain_300m_baseline_5k.yaml \
#       checkpoints/pretrain_300m_baseline/checkpoint_step8000.pt \
#       300m_baseline_step8000

CONFIG=${1:?config required}
CKPT=${2:?checkpoint required}
NAME=${3:?output name required}
TASKS=${4:-default}
BATCH=${5:-8}

OUT=experiments/results/lmeval/${NAME}.json
echo "Config:  $CONFIG"
echo "Ckpt:    $CKPT"
echo "Output:  $OUT"
echo "Tasks:   $TASKS"
echo "Started: $(date)"

/usr/bin/python3.9 scripts/eval_lm_harness.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --tasks "$TASKS" \
    --batch_size "$BATCH" \
    --output_path "$OUT"
