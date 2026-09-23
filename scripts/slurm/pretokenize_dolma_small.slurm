#!/bin/bash
#SBATCH --job-name=pretok_dolma
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --output=logs/pretok_dolma_%j.out
#SBATCH --error=logs/pretok_dolma_%j.err

# Rebuild a Dolma v1.7 slice in mmap format (the original 12B-token shards were
# deleted from /work/hdd/bfqt/data/pretok). Sized for the 75M router-locality
# ablation: 5 arms x 3000 steps x 524K tokens = 1.57B training tokens each,
# all arms reading the same shards; the eval windows are split off afterwards
# by scripts/split_eval_from_mmap.py. Compute nodes have outbound network
# (verified 2026-09-23 on gpua001), so HF streaming works here.
set -u
cd "${REPO_DIR:-/u/xiaocong/dagformer}"
mkdir -p logs
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_DOWNLOAD_TIMEOUT=300 HF_HUB_ENABLE_HF_TRANSFER=0 HF_DATASETS_TIMEOUT=300
export HF_XET_NATIVE=0 HF_HUB_DISABLE_XET=1
OUT_DIR="${OUT_DIR:-/work/hdd/bfqt/xiaocong/dagformer_pruning/data/dolma_v1_7_1p7b}"
TOKEN_BUDGET="${TOKEN_BUDGET:-1700000000}"
if [ -f "$OUT_DIR/index.json" ]; then echo "already built: $OUT_DIR"; exit 0; fi
python3 scripts/pretokenize.py --dataset allenai/dolma --dataset-version v1_7 \
    --out-dir "$OUT_DIR" --token-budget "$TOKEN_BUDGET" --seq-len 1024 --seed 0 \
    --tokenizer-id /work/hdd/bfqt/shared/dagformer-models/tokenizer
echo "pretok rc=$? ($(date))"
