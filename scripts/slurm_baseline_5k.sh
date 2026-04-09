#!/bin/bash
#SBATCH --job-name=bl-5k
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=30:00:00
#SBATCH --output=logs/baseline_5k_%j.out
#SBATCH --error=logs/baseline_5k_%j.err
#SBATCH --signal=B:USR1@120

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

torchrun \
    --nproc_per_node=4 \
    --master_port=29504 \
    scripts/pretrain_baseline.py \
    --config configs/pretrain_300m_baseline_5k.yaml
