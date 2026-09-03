#!/bin/bash
#SBATCH --job-name=eval-300m
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/eval_300m_%j.out
#SBATCH --error=logs/eval_300m_%j.err

# Extend the confounder-free (identical dolma-mmap data) method comparison to 300M.
# All four methods evaluated at step 9000 for a fair iso-token comparison — EXCEPT
# Hyper-Connections, whose run only reached step 8000 (its point is labelled
# separately and must NOT be read as iso-token with the others).

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
        --tasks wikitext --batch_size 4 --output_path "$OUT/$name.json" \
      && echo "OK: $name" || echo "FAILED: $name"
  else
    echo "skip $name — no ckpt ($ckpt)"
  fi
}

echo "Started: $(date)"
run dense           configs/pretrain_300m_baseline_mmap.yaml      checkpoints/pretrain_300m_baseline_mmap/checkpoint_step9000.pt      dense_300m_mmap_s9000
run fourway         configs/fourway_300m_dagformer_mmap.yaml      checkpoints/fourway_300m_dagformer_mmap/checkpoint_step9000.pt      dag_300m_mmap_s9000
run denseformer     configs/pretrain_300m_denseformer.yaml        checkpoints/pretrain_300m_denseformer/checkpoint_step9000.pt        df_300m_s9000
run hyperconnection configs/pretrain_300m_hyperconnection.yaml    checkpoints/pretrain_300m_hyperconnection/checkpoint_step8000.pt    hc_300m_s8000
echo "ALL 300M EVALS DONE $(date)"
