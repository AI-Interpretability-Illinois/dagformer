#!/bin/bash
#SBATCH --job-name=pretok
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pretok_%j.out
#SBATCH --error=logs/pretok_%j.err

# One-time offline pretokenizer: stream a source ONCE, pack, write uint32 mmap
# shards + index.json to a local dir. Parameterized via env vars so the same
# script builds either the olmo-mix (1B run) or Dolma (600M run) shard set:
#   sbatch --export=ALL,DATASET_VERSION=olmo_mix,OUT_DIR=...,TOKEN_BUDGET=...,SEED=0 scripts/slurm_pretokenize.sh
# Runs on a non-preempt partition because pretokenize.py has no mid-job resume
# (a preemption would restart it from scratch). 1 GPU is requested only because
# this account cannot submit CPU-only jobs; tokenization itself is CPU-bound.

set -eo pipefail
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs

export HF_HOME=/work/hdd/bfqt/yurenh2/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:${PYTHONPATH:-}
export PATH=$HOME/.local/bin:$PATH
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_DATASETS_TIMEOUT=300
export PYTHONUNBUFFERED=1
# hf_xet CAS cache fills up and errors with a spurious "Disk quota exceeded".
export HF_XET_NATIVE=0
export HF_HUB_DISABLE_XET=1

DATASET_VERSION=${DATASET_VERSION:-olmo_mix}
OUT_DIR=${OUT_DIR:-/work/hdd/bfqt/data/pretok/olmo_mix_21b}
TOKEN_BUDGET=${TOKEN_BUDGET:-21000000000}
SEQ_LEN=${SEQ_LEN:-1024}
SEED=${SEED:-0}

echo "=============================================="
echo "Pretokenize: version=$DATASET_VERSION"
echo "  out-dir:      $OUT_DIR"
echo "  token-budget: $TOKEN_BUDGET"
echo "  seq-len:      $SEQ_LEN   seed: $SEED"
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Started:      $(date)"
echo "=============================================="

/usr/bin/python3.9 scripts/pretokenize.py \
    --dataset allenai/dolma \
    --dataset-version "$DATASET_VERSION" \
    --out-dir "$OUT_DIR" \
    --token-budget "$TOKEN_BUDGET" \
    --seq-len "$SEQ_LEN" \
    --seed "$SEED"

echo "Done: $(date)"
df -h /work | tail -1
du -sh "$OUT_DIR" 2>&1 || true
