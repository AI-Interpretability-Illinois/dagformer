#!/bin/bash
#SBATCH --job-name=eval-repro
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_repro_%j.out
#SBATCH --error=logs/eval_repro_%j.err

# Eval completed DenseFormer/HyperConnection reproduction checkpoints on wikitext
# to get bits-per-byte, comparable to all our other frontier points. Uses the
# new --model-type loaders in eval_lm_harness.py (build the ACTUAL method model +
# load routing params). Skips any checkpoint not yet written (still-running runs).

set -uo pipefail
cd /projects/bfqt/users/yurenh2/ml-projects/DAGFormer
mkdir -p logs experiments/results/lmeval/reproductions

export HF_HOME=/work/hdd/bfqt/yurenh2/huggingface_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/projects/bfqt/users/yurenh2/ml-projects/DAGFormer:${PYTHONPATH:-}
export PATH=$HOME/.local/bin:$PATH
export PYTHONUNBUFFERED=1
export HF_XET_NATIVE=0
export HF_HUB_DISABLE_XET=1

OUT=experiments/results/lmeval/reproductions

run() {
  local mt=$1 cfg=$2 ckpt=$3 name=$4
  if [ -f "$ckpt" ]; then
    echo "================ eval $name ($mt) ================"
    /usr/bin/python3.9 scripts/eval_lm_harness.py \
        --model-type "$mt" --config "$cfg" --ckpt "$ckpt" \
        --tasks wikitext --batch_size 8 \
        --output_path "$OUT/$name.json" \
      && echo "OK: $name" || echo "FAILED: $name"
  else
    echo "skip $name — checkpoint not present yet ($ckpt)"
  fi
}

echo "Started: $(date)"
run denseformer     configs/pretrain_75m_denseformer.yaml     checkpoints/pretrain_75m_denseformer/checkpoint_step3000.pt     df_75m
run denseformer     configs/pretrain_150m_denseformer.yaml    checkpoints/pretrain_150m_denseformer/checkpoint_step6000.pt    df_150m
run denseformer     configs/pretrain_300m_denseformer.yaml    checkpoints/pretrain_300m_denseformer/checkpoint_step12000.pt   df_300m
run hyperconnection configs/pretrain_75m_hyperconnection.yaml  checkpoints/pretrain_75m_hyperconnection/checkpoint_step3000.pt hc_75m
run hyperconnection configs/pretrain_150m_hyperconnection.yaml checkpoints/pretrain_150m_hyperconnection/checkpoint_step6000.pt hc_150m
run hyperconnection configs/pretrain_300m_hyperconnection.yaml checkpoints/pretrain_300m_hyperconnection/checkpoint_step12000.pt hc_300m
echo "ALL EVALS DONE $(date)"
