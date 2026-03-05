#!/bin/bash
#SBATCH --job-name=grad-ckpt
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=00:30:00
#SBATCH --output=logs/grad_ckpt_%j.out
#SBATCH --error=logs/grad_ckpt_%j.err

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TRANSFORMERS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export HF_HUB_CACHE=/projects/bfqt/users/yurenh2/hf_cache/hub
export HF_DATASETS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer

echo "=== Gradient analysis at A=1 on fine-tuned OLMo (step 5000) ==="

python scripts/gradient_analysis.py \
    --revision stage1-step1907359-tokens4001B \
    --checkpoint checkpoints/p2_ddp_long/checkpoint_step5000 \
    --eval-dataset allenai/dolmino-mix-1124 \
    --eval-dataset-name dolmino_mix \
    --eval-skip 10000 \
    --eval-size 50 \
    --batch-size 4
