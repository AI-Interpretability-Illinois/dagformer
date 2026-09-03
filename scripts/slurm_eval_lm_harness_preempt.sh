#!/bin/bash
#SBATCH --job-name=lmeval
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4-preempt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lmeval_%j.out
#SBATCH --error=logs/lmeval_%j.err

set -eo pipefail
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs experiments/results/lmeval

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:${PYTHONPATH:-}
export PATH=$HOME/.local/bin:$PATH

CONFIG=${1:?config required}
CKPT=${2:?checkpoint required}
NAME=${3:?output name required}
TASKS=${4:-default}
BATCH=${5:-8}
LIMIT=${6:-}

OUT=experiments/results/lmeval/${NAME}.json
echo "Config:  $CONFIG"
echo "Ckpt:    $CKPT"
echo "Output:  $OUT"
echo "Tasks:   $TASKS"
echo "Started: $(date)"

CMD="/usr/bin/python3.9 scripts/eval_lm_harness.py --config $CONFIG --ckpt $CKPT --tasks $TASKS --batch_size $BATCH --output_path $OUT"
if [ -n "$LIMIT" ]; then CMD="$CMD --limit $LIMIT"; fi
echo "$CMD"
eval $CMD
