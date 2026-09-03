#!/bin/bash
#SBATCH --job-name=diag-nan
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=1:00:00
#SBATCH --output=logs/diag_nan_%j.out
#SBATCH --error=logs/diag_nan_%j.err
set -uo pipefail
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
export HF_HOME=/work/hdd/bfqt/yurenh2/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
echo "########## A: v_norm ON (the suspected NaN trigger) ##########"
/usr/bin/python3.9 scripts/diagnose_600m_nan.py --config configs/fourway_600m_chinchilla.yaml \
    --device cuda --data mmap --vnorm on  2>&1 | tail -25
echo ""
echo "########## B: v_norm OFF (the proposed fix) ##########"
/usr/bin/python3.9 scripts/diagnose_600m_nan.py --config configs/fourway_600m_chinchilla.yaml \
    --device cuda --data mmap --vnorm off 2>&1 | tail -25
