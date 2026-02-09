#!/bin/bash
export HF_HOME=/projects/bfqt/users/yurenh2/hf_cache
export TRANSFORMERS_CACHE=/projects/bfqt/users/yurenh2/hf_cache/transformers
export HF_HUB_CACHE=/projects/bfqt/users/yurenh2/hf_cache/hub

export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH

cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer

echo "=== Python version ==="
python3 --version

echo ""
echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "=== Running ALL 6 sanity checks ==="
python3 scripts/sanity_check.py --device cuda --checks 1 2 3 4 5 6
