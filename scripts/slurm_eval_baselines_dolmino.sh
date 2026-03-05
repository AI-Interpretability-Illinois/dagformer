#!/bin/bash
#SBATCH --job-name=eval-dolmino
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_dolmino_%j.out
#SBATCH --error=logs/eval_dolmino_%j.err

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TRANSFORMERS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export HF_HUB_CACHE=/projects/bfqt/users/yurenh2/hf_cache/hub
export HF_DATASETS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer

# Eval on Dolmino mix (same data we'll train on)
echo "=== Stage 1 Final on Dolmino ==="
python scripts/eval_baseline.py --revision stage1-step1907359-tokens4001B \
    --eval-dataset allenai/dolmino-mix-1124 --eval-dataset-name dolmino_mix

echo ""
echo "=== Stage 2 Final on Dolmino ==="
python scripts/eval_baseline.py --revision stage2-ingredient3-step23852-tokens51B \
    --eval-dataset allenai/dolmino-mix-1124 --eval-dataset-name dolmino_mix
