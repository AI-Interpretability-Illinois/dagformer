#!/bin/bash
#SBATCH --job-name=gc_bf16
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:10:00
#SBATCH --output=logs/gc_bf16_%j.out --error=logs/gc_bf16_%j.err
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
export PYTHONPATH=$PWD:$PYTHONPATH
python3 scripts/gradcheck_bf16.py
