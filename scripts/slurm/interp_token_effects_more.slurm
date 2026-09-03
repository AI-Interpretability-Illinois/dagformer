#!/bin/bash
#SBATCH --job-name=interp_tokfx2
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/interp_tokfx2_%j.out
#SBATCH --error=logs/interp_tokfx2_%j.err
set -u
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export HF_HUB_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export TRANSFORMERS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
GROUP=${1:?group index}
HEADS=$(sed -n "$((GROUP + 1))p" experiments/results/interp/token_effects/unnamed_groups.txt)
SAVE=""; [ "$GROUP" = "0" ] && SAVE="--save-corpus"
python3 scripts/interp_token_effects.py \
    --config configs/fourway_300m_dagformer_mmap.yaml \
    --ckpt checkpoints/fourway_300m_dagformer_mmap/checkpoint_step9000.pt \
    --n-seq 4000 --offset-seq 1000 --seed 42 --block-size 1024 --skip-positions 5500000 \
    --batch 16 --heads "$HEADS" --tag _more $SAVE
rc=$?; echo "tokfx2 group $GROUP exited rc=$rc"; exit "$rc"
