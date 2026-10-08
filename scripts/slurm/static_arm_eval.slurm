#!/bin/bash
#SBATCH --job-name=static_eval
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --reservation=sup-30781
#SBATCH --nodelist=gpua047
#SBATCH --output=logs/static_eval_%j.out
# J5 follow-up: evaluate the 75M static-table arm on the common caches and on the locality arms' own cache, refit.
cd /u/xiaocong/dagformer
source ~/.bashrc; conda activate modularity 2>/dev/null || true
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
L=/work/hdd/bfqt/xiaocong/dagformer_pruning/checkpoints/locality
bash scripts/scaling_common_eval.sh delta
python3 scripts/eval_pretrained.py --model /projects/biro/xiaocong/dagformer_locality/static --config configs/locality/75m_static.yaml --name static \
  --model $L/global --config configs/locality/75m_global.yaml --name global \
  --eval own=$L/global/eval_cache.pt --out experiments/coherence/results/locality/static_vs_global_own.json
python3 scripts/scaling_collect.py --machine delta && python3 scripts/scaling_fit.py
echo "exit rc=$? at $(date)"
