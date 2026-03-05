#!/bin/bash
#SBATCH --job-name=pt-dagformer
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=30:00:00
#SBATCH --output=logs/pretrain_dagformer_%j.out
#SBATCH --error=logs/pretrain_dagformer_%j.err
#SBATCH --signal=B:USR1@120

# DAGFormer 300M pretraining (static A or self-embed)
# Usage: sbatch scripts/slurm_pretrain_dagformer.sh configs/pretrain_300m_static.yaml
#    or: sbatch scripts/slurm_pretrain_dagformer.sh configs/pretrain_300m_selfembed.yaml

CONFIG=${1:-configs/pretrain_300m_static.yaml}

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

echo "Config: $CONFIG"
echo "Starting at $(date)"

torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    scripts/pretrain_dagformer.py \
    --config "$CONFIG"
