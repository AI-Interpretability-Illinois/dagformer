#!/bin/bash
#SBATCH --job-name=diff_te
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/diff_train_eval_%j.out
#SBATCH --error=logs/diff_train_eval_%j.err

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

# Default: use the best known FourWay checkpoint
CKPT=${1:-checkpoints/fourway_corrected_joint_causal/checkpoint_step5000.pt}
CONFIG=${2:-configs/fourway_corrected_joint_causal.yaml}

echo "Checkpoint: $CKPT"
echo "Config: $CONFIG"

python3 scripts/diff_train_eval_predictor.py \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --batch 2 \
    --seq_len 256
