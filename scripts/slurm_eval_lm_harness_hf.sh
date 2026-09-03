#!/bin/bash
#SBATCH --job-name=lmeval-hf
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4-preempt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/lmeval_hf_%j.out
#SBATCH --error=logs/lmeval_hf_%j.err

set -eo pipefail
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs experiments/results/lmeval

export HF_HOME=/work/hdd/bfqt/yurenh2/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:${PYTHONPATH:-}
export PATH=$HOME/.local/bin:$PATH

# Usage: sbatch scripts/slurm_eval_lm_harness_hf.sh <hf_model> <revision> <output_name> [tasks] [batch]
HF_MODEL=${1:?hf_model required}
REVISION=${2:?revision required}
NAME=${3:?output name required}
TASKS=${4:-default}
BATCH=${5:-4}

OUT=experiments/results/lmeval/${NAME}.json
echo "HF model: $HF_MODEL @ $REVISION"
echo "Output:   $OUT"
echo "Tasks:    $TASKS"
echo "Started:  $(date)"

/usr/bin/python3.9 scripts/eval_lm_harness.py \
    --hf_model "$HF_MODEL" \
    --hf_revision "$REVISION" \
    --tasks "$TASKS" \
    --batch_size "$BATCH" \
    --output_path "$OUT"
