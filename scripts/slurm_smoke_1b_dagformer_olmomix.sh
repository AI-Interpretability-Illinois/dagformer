#!/bin/bash
#SBATCH --job-name=smoke-1b-4way
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=1800G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke_1b_dagformer_%j.out
#SBATCH --error=logs/smoke_1b_dagformer_%j.err

set -eo pipefail

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs /work/hdd/bfqt/dagformer_checkpoints/smoke_1b_dagformer_olmomix

export HF_HOME=/work/hdd/bfqt/yurenh2/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:${PYTHONPATH:-}
export PATH=$HOME/.local/bin:$PATH
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_DATASETS_TIMEOUT=300
export HF_XET_NATIVE=0
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

CONFIG=configs/fourway_1b_dagformer_olmomix_smoke.yaml

echo "Config:    $CONFIG"
echo "Save dir:  /work/hdd/bfqt/dagformer_checkpoints/smoke_1b_dagformer_olmomix"
echo "Started:   $(date)"
echo "Job ID:    $SLURM_JOB_ID"

torchrun \
    --nproc_per_node=8 \
    --master_port=29523 \
    scripts/pretrain_dagformer.py \
    --config "$CONFIG"

echo "Smoke run completed at $(date)."
