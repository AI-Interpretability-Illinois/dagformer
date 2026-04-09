#!/bin/bash
#SBATCH --job-name=eval_bm
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_benchmark_%j.out
#SBATCH --error=logs/eval_benchmark_%j.err

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

python3 scripts/eval_sanity_benchmark.py \
    --fourway_ckpt checkpoints/fourway_corrected_joint_causal/checkpoint_step5000.pt \
    --fourway_config configs/fourway_corrected_joint_causal.yaml \
    --dense_ckpt checkpoints/pretrain_300m_baseline_5k/checkpoint_step5000.pt \
    --n_batches 20
