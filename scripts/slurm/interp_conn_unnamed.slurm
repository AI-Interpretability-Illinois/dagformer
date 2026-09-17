#!/bin/bash
#SBATCH --job-name=interp_connU
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/interp_connU_%j.out
#SBATCH --error=logs/interp_connU_%j.err
set -u
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export HF_HUB_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export TRANSFORMERS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
GROUP=${1:?group index}
HEADS=$(sed -n "$((GROUP + 1))p" experiments/results/interp/token_effects/unnamed3_groups.txt)
python3 scripts/interp_conn_effects.py --heads "$HEADS" --shard "u$GROUP" --batch 16
rc=$?; echo "conn group $GROUP exited rc=$rc"; exit "$rc"
