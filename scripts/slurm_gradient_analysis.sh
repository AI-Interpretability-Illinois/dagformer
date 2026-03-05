#!/bin/bash
#SBATCH --job-name=grad-analysis
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=01:00:00
#SBATCH --output=logs/grad_analysis_%j.out
#SBATCH --error=logs/grad_analysis_%j.err

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TRANSFORMERS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export HF_HUB_CACHE=/projects/bfqt/users/yurenh2/hf_cache/hub
export HF_DATASETS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs

echo "=== Gradient Analysis: Stage 1 Final on Dolma v1.7 ==="
python3 -u scripts/gradient_analysis.py --revision stage1-step1907359-tokens4001B

echo ""
echo "=== Gradient Analysis: Stage 1 Final on Dolmino ==="
python3 -u scripts/gradient_analysis.py --revision stage1-step1907359-tokens4001B \
    --eval-dataset allenai/dolmino-mix-1124 --eval-dataset-name dolmino_mix
