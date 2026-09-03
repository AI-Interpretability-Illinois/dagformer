#!/bin/bash
#SBATCH --job-name=eval-matched
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/eval_matched_%j.out
#SBATCH --error=logs/eval_matched_%j.err

# Eval the MATCHED-DATA dense + DAGFormer runs (trained on the SAME dolma mmap as
# the DenseFormer/HC reproductions) on wikitext BPB. Combined with the already-
# evaluated DenseFormer/HC-mmap numbers, this gives a confounder-free method
# comparison at 75/150M — everyone on identical data/tokenizer/tokens/setup.

set -uo pipefail
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs experiments/results/lmeval/matched_mmap

export HF_HOME=/work/hdd/bfqt/yurenh2/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:${PYTHONPATH:-}
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export HF_XET_NATIVE=0
export HF_HUB_DISABLE_XET=1

OUT=experiments/results/lmeval/matched_mmap
run() {
  local mt=$1 cfg=$2 ckpt=$3 name=$4
  if [ -f "$ckpt" ]; then
    echo "================ eval $name ($mt) ================"
    /usr/bin/python3.9 scripts/eval_lm_harness.py \
        --model-type "$mt" --config "$cfg" --ckpt "$ckpt" \
        --tasks wikitext --batch_size 8 --output_path "$OUT/$name.json" \
      && echo "OK: $name" || echo "FAILED: $name"
  else
    echo "skip $name — no ckpt ($ckpt)"
  fi
}

echo "Started: $(date)"
run dense   configs/pretrain_75m_baseline_mmap.yaml   checkpoints/pretrain_75m_baseline_mmap/checkpoint_step3000.pt   dense_75m_mmap
run dense   configs/pretrain_150m_baseline_mmap.yaml  checkpoints/pretrain_150m_baseline_mmap/checkpoint_step6000.pt  dense_150m_mmap
run fourway configs/pretrain_75m_dagformer_mmap.yaml  checkpoints/pretrain_75m_dagformer_mmap/checkpoint_step3000.pt  dag_75m_mmap
run fourway configs/pretrain_150m_dagformer_mmap.yaml checkpoints/pretrain_150m_dagformer_mmap/checkpoint_step6000.pt dag_150m_mmap
echo "ALL MATCHED EVALS DONE $(date)"
