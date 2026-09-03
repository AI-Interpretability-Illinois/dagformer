#!/bin/bash
#SBATCH --job-name=off_gc
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1 --gpus-per-node=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=00:15:00
#SBATCH --output=logs/official_gc_%j.out --error=logs/official_gc_%j.err
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
export PYTHONPATH=$PWD:$PYTHONPATH
python3 scripts/official_gradcheck.py
