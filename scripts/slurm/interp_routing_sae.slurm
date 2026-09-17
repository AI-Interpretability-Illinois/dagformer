#!/bin/bash
#SBATCH --job-name=interp_rsae
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=02:30:00
#SBATCH --output=logs/interp_rsae_%j.out
#SBATCH --error=logs/interp_rsae_%j.err

set -u
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export HF_HUB_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export TRANSFORMERS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

python3 scripts/interp_routing_sae.py


rc=$?
echo "editing exited rc=$rc"
exit "$rc"
