#!/bin/bash
#SBATCH --job-name=eval_id
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=logs/eval_force_identity_%j.out
#SBATCH --error=logs/eval_force_identity_%j.err

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

# Best FourWay checkpoint (step 5000, eval NLL 5.49)
CKPT=${1:-checkpoints/fourway_corrected_joint_causal/checkpoint_step5000.pt}
CONFIG=${2:-configs/fourway_corrected_joint_causal.yaml}

echo "Checkpoint: $CKPT"
echo "Config: $CONFIG"
echo ""

python3 scripts/eval_force_identity.py \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --max_batches 50
