#!/bin/bash
#SBATCH --job-name=fw_1gpu
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=logs/fw_1gpu_%j.out
#SBATCH --error=logs/fw_1gpu_%j.err

CONFIG=${1:-configs/fourway_1gpu_test.yaml}

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs

export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

echo "Config: $CONFIG"
echo "Single GPU — no DDP"
echo ""

# Single process, no torchrun
python3 scripts/pretrain_dagformer.py --config "$CONFIG"
