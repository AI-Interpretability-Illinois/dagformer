#!/bin/bash
#SBATCH --job-name=vnorm_san
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/verify_vnorm_%j.out
#SBATCH --error=logs/verify_vnorm_%j.err

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

echo "=========================================="
echo "Baseline: identity init WITHOUT v_norm (should PASS)"
echo "=========================================="
python3 scripts/verify_identity_init.py --dtype float32 --batch 2 --seq_len 128

echo ""
echo "=========================================="
echo "With v_norm enabled: identity init (expect small drift)"
echo "=========================================="
python3 scripts/verify_identity_init.py --dtype float32 --batch 2 --seq_len 128 --use_v_norm
