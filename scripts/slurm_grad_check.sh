#!/bin/bash
#SBATCH --job-name=grad_chk
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/grad_check_%j.out
#SBATCH --error=logs/grad_check_%j.err

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH

echo "============================================"
echo "Small model (4L 4H 128D) — fast, definitive"
echo "============================================"
python3 scripts/gradient_and_reference_check.py

echo ""
echo "============================================"
echo "Full size (12L 16H 1024D) — slower, real arch"
echo "============================================"
python3 scripts/gradient_and_reference_check.py --full_size
