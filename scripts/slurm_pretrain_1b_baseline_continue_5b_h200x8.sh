#!/bin/bash
#SBATCH --job-name=1b-base-5b
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=1800G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pretrain_1b_baseline_5b_%j.out
#SBATCH --error=logs/pretrain_1b_baseline_5b_%j.err
#SBATCH --signal=B:USR1@120

set -euo pipefail

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs checkpoints/pretrain_1b_baseline_continue_5b_h200x8

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

CONFIG=configs/pretrain_1b_baseline_continue_5b_h200x8.yaml

echo "Config: $CONFIG"
echo "Starting at $(date)"
echo "Continuing from /projects/bfqt/xiaocong/dagformer_ckpts/pretrain_1b_baseline/checkpoint_step9500.pt"
echo "Target: actual ~5.238B tokens, final total_steps=13309"

torchrun \
    --nproc_per_node=8 \
    --master_port=29517 \
    scripts/pretrain_baseline.py \
    --config "$CONFIG"
