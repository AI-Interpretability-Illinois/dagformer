#!/bin/bash
#SBATCH --job-name=throughput
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:45:00
#SBATCH --reservation=sup-30781
#SBATCH --output=logs/throughput_%j.out
# J6: measured training-step time and prefill throughput, dense vs DAGFormer (+ static-predictor variant), 300M and 1B.
cd /u/xiaocong/dagformer
source ~/.bashrc; conda activate modularity 2>/dev/null || true
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
nvidia-smi --query-gpu=name --format=csv,noheader
python scripts/throughput_bench.py --out experiments/topology/throughput.json
echo "exit rc=$? at $(date)"
