#!/bin/bash
#SBATCH --job-name=d6-reverse
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=200g
#SBATCH --time=16:00:00
#SBATCH --output=logs/d6_reverse_%j.out
#SBATCH --error=logs/d6_reverse_%j.err
#SBATCH --signal=B:USR1@120

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TRANSFORMERS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export HF_HUB_CACHE=/projects/bfqt/users/yurenh2/hf_cache/hub
export HF_DATASETS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs checkpoints/d6_reverse_init

echo "=== D6: Reverse init_logit=-15, A~0, learn to turn ON, 4x A40 ==="
torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 \
    --master_addr=localhost --master_port=29500 \
    scripts/train.py --config configs/p2_reverse_init.yaml
