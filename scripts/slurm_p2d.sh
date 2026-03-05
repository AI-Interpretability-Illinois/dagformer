#!/bin/bash
#SBATCH --signal=SIGUSR1@120
export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TRANSFORMERS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export HF_HUB_CACHE=/projects/bfqt/users/yurenh2/hf_cache/hub
export HF_DATASETS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs checkpoints/p2d
echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -u scripts/train.py --config configs/p2d_low_tau.yaml
