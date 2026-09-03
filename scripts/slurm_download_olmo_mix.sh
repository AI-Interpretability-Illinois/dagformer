#!/bin/bash
#SBATCH --job-name=dl-olmomix
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4-preempt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dl_olmomix_%j.out
#SBATCH --error=logs/dl_olmomix_%j.err

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
# Disable hf_xet: its CAS chunk cache fills up mid-download and errors with
# "CAS service error : IO Error: Disk quota exceeded" (it's NOT a /work quota,
# it's xet's internal cache).
export HF_XET_NATIVE=0
export HF_HUB_DISABLE_XET=1

echo "Started: $(date)"
/usr/bin/python3.9 scripts/download_olmo_mix_local.py
echo "Done:    $(date)"
df -h /work | tail -1
du -sh /work/hdd/bfqt/data/olmo-mix-1124 2>&1
