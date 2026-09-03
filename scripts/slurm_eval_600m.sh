#!/bin/bash
#SBATCH --job-name=eval-600m
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/eval_600m_%j.out
#SBATCH --error=logs/eval_600m_%j.err
set -uo pipefail
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs experiments/results/lmeval/matched_mmap
export HF_HOME=/work/hdd/bfqt/yurenh2/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:${PYTHONPATH:-}
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export HF_XET_NATIVE=0
export HF_HUB_DISABLE_XET=1
/usr/bin/python3.9 scripts/eval_lm_harness.py \
    --model-type dense --config configs/pretrain_600m_baseline.yaml \
    --ckpt /work/hdd/bfqt/dagformer_checkpoints/pretrain_600m_baseline_mmap/checkpoint_step22900.pt \
    --tasks wikitext --batch_size 4 \
    --output_path experiments/results/lmeval/matched_mmap/dense_600m_mmap_s22900.json \
  && echo "OK dense_600m" || echo "FAILED dense_600m"
