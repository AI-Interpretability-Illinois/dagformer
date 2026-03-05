#!/bin/bash
#SBATCH --job-name=pt-300m
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=30:00:00
#SBATCH --output=logs/pretrain_300m_%j.out
#SBATCH --error=logs/pretrain_300m_%j.err
#SBATCH --signal=B:USR1@120

# Mini OLMo-2 300M baseline pretraining
# 10000 steps × 524K tokens/step ≈ 5.2B tokens
# Estimated wall time: ~8-12h on 4× A40

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    scripts/pretrain_baseline.py \
    --config configs/pretrain_300m.yaml
