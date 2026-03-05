#!/bin/bash
#SBATCH --job-name=eval-baselines
#SBATCH --partition=gpuA40x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_baselines_%j.out
#SBATCH --error=logs/eval_baselines_%j.err

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TRANSFORMERS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export HF_HUB_CACHE=/projects/bfqt/users/yurenh2/hf_cache/hub
export HF_DATASETS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

echo "=== Stage 1 Final (pretrain only, 4T tokens) ==="
python scripts/eval_baseline.py --revision stage1-step1907359-tokens4001B

echo ""
echo "=== Stage 2 Final (midtrain, ingredient3, 51B tokens) ==="
python scripts/eval_baseline.py --revision stage2-ingredient3-step23852-tokens51B

echo ""
echo "=== Main (post-trained: SFT+DPO+GRPO) ==="
python scripts/eval_baseline.py
