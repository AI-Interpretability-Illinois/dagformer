#!/bin/bash
#SBATCH --job-name=fw150m_lite
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=160g
#SBATCH --time=18:00:00
#SBATCH --output=logs/fw150m_lite_%j.out
#SBATCH --error=logs/fw150m_lite_%j.err

set -u
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export HF_HUB_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export TRANSFORMERS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer

torchrun --standalone --nproc_per_node=4 scripts/pretrain_dagformer.py \
    --config configs/fourway_150m_lite_mmap.yaml
rc=$?
echo "torchrun exited rc=$rc"
exit "$rc"
