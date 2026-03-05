#!/bin/bash
#SBATCH --job-name=ddp-test
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200g
#SBATCH --time=00:30:00
#SBATCH --output=logs/ddp_test_%j.out
#SBATCH --error=logs/ddp_test_%j.err

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TRANSFORMERS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export HF_HUB_CACHE=/projects/bfqt/users/yurenh2/hf_cache/hub
export HF_DATASETS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs checkpoints/p2_ddp_test

echo "=== DDP Test: 4x A40, 100 steps ==="
echo "Nodes: $SLURM_NNODES, GPUs: $SLURM_GPUS_ON_NODE, Tasks: $SLURM_NTASKS"

torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    scripts/train.py --config configs/p2_ddp_test.yaml
