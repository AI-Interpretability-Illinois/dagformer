#!/bin/bash
#SBATCH --job-name=diag
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/diagnose_%j.out
#SBATCH --error=logs/diagnose_%j.err

CHECKPOINT=${1:-checkpoints/fourway_joint_causal/checkpoint_step5000.pt}
CONFIG=${2:-configs/fourway_joint_causal.yaml}

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs experiments/routing_diagnosis

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"

python3 scripts/diagnose_routing.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --output_dir experiments/routing_diagnosis
