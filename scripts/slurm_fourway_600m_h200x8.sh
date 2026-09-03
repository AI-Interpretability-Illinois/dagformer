#!/bin/bash
#SBATCH --job-name=600m-4way
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=1800G
#SBATCH --time=36:00:00
#SBATCH --output=logs/fourway_600m_%j.out
#SBATCH --error=logs/fourway_600m_%j.err
#SBATCH --signal=B:USR1@120

set -euo pipefail

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs /work/hdd/bfqt/dagformer_checkpoints/fourway_600m_chinchilla

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:${PYTHONPATH:-}
export PATH=$HOME/.local/bin:$PATH

CONFIG=configs/fourway_600m_chinchilla.yaml

echo "Config: $CONFIG"
echo "Starting at $(date)"

torchrun \
    --nproc_per_node=8 \
    --master_port=29519 \
    scripts/pretrain_dagformer.py \
    --config "$CONFIG"
