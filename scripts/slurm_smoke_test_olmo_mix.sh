#!/bin/bash
#SBATCH --job-name=smoke-olmomix
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_olmomix_%j.out
#SBATCH --error=logs/smoke_olmomix_%j.err

set -eo pipefail
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs

export HF_HOME=/work/hdd/bfqt/yurenh2/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:${PYTHONPATH:-}
export PATH=$HOME/.local/bin:$PATH
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_DATASETS_TIMEOUT=300
export PYTHONUNBUFFERED=1

echo "Started: $(date)"
/usr/bin/python3.9 scripts/smoke_test_olmo_mix.py
echo "Done:    $(date)"
